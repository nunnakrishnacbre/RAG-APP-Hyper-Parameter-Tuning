import os
import re
import time
import json
import random
import joblib
import traceback
import logs
import torch
import transformers
from bert_score import score
from itertools import product
from datetime import datetime
from typing import List, Dict, Any
from configuration import Configuration
from rag_scripts.llm.llmResponse import GROQLLM
from pinecone import Pinecone, PineconeException
from rag_scripts.rag_pipeline import RAGPipeline
from sentence_transformers import SentenceTransformer, util
from rag_scripts.embedding.vector_db.faiss_db import FAISSVectorDB
from rag_scripts.documents_processing.chunking import PyMuPDFChunker
from rag_scripts.embedding.embedder import SentenceTransformerEmbedder
from rag_scripts.embedding.vector_db.chroma_db import chromaDBVectorDB
from rag_scripts.embedding.vector_db.pinecone_db import PineconeVectorDB

VECTOR_DB_CONSTRUCTORS = {
    "chroma": chromaDBVectorDB,
    "faiss": FAISSVectorDB,
    "pinecone": PineconeVectorDB
}

class RAGEvaluator:
    def __init__(self, eval_data_path: str = Configuration.EVAL_DATA_PATH,
                 pdf_path: str = Configuration.FULL_PDF_PATH):
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation data not found at: {eval_data_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")
        
        with open(eval_data_path, 'r') as f:
            self.eval_queries = json.load(f)

        self.pdf_path = pdf_path
        self.embedder = SentenceTransformer(Configuration.DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        logs.logger.info(f"RAG Evaluator initialized with {len(self.eval_queries)} evaluation queries")

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-z0-9]', '-', name.lower())
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')
        return sanitized[:45].rstrip('-') if len(sanitized) > 45 else sanitized
    
    def _calculate_retrieval_relevance(self, query: str, retrieved_docs: List[Dict[str, Any]],
                                       expected_answer: str =None) -> float:
        if not retrieved_docs:
            logs.logger.warning(f"No retrieved documents for query: {query}")
            return 0.0
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        reference_embedding = [query_embedding]

        score_cosine = []
        for doc in retrieved_docs:
            doc_content = doc.get('content','')
            if not doc_content.strip():
                continue
            doc_embedding = self.embedder.encode(doc_content, convert_to_tensor=True, normalize_embeddings=True)
            max_sim_to_reference = 0.0
            for ref_embedding in reference_embedding:
                sim = util.cos_sim(ref_embedding, doc_embedding)
                if  isinstance(sim,torch.Tensor):
                    sim_value = sim.item()
                else:
                    sim_value = sim
                if sim_value > max_sim_to_reference:
                    max_sim_to_reference = sim_value
            score_cosine.append(max_sim_to_reference)
        cosine_score = sum(score_cosine) / len(score_cosine) if score_cosine else 0.0

        logs.logger.info(f"Max cosine score for query '{query}': {cosine_score}")
        return cosine_score
    
    def _calculate_response_groundedness(self, response: Dict[str,Any], retrieved_docs: List[Dict[str, Any]]) -> float:
        try:
            if not retrieved_docs:
                logs.logger.warning(f"No retrieved documents for groundedness evaluation")
                return 0.0
            response_text = response.get("summary","")
            if not isinstance(response_text,str):
                response_text = str(response_text)

            response_segments = re.split(r'[\u25cf\u25cb]\s*|\n\s*\n', response_text.strip())
            response_segments = [seg.strip() for seg in response_segments if seg.strip()]

            groundedness_scores = []
            for doc in retrieved_docs:
                doc_content = doc.get('content','')
                if not doc_content.strip():
                    continue
                doc_embedding = self.embedder.encode(doc_content,
                                            convert_to_tensor=True,
                                            normalize_embeddings=True)
                for segment in response_segments:
                    if segment:
                        segment_embedding = self.embedder.encode(segment,
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)

                        segment_similarity = util.cos_sim(segment_embedding,
                                                          doc_embedding).item()
                        groundedness_scores.append(segment_similarity)

                full_response_embedding = self.embedder.encode(response_text,
                                                        convert_to_tensor=True,
                                                        normalize_embeddings=True)
                full_response_similarity = util.cos_sim(full_response_embedding,doc_embedding).item()
                groundedness_scores.append(full_response_similarity)

            groundedness = max(
                groundedness_scores) if groundedness_scores else 0.0  # Use max to handle structured responses
            logs.logger.info(f"Max groundedness score for response: {groundedness:.2f}")
            return groundedness
        except Exception as ex:
            logs.logger.error(f"Exception in calculating groundedness: {ex}")
            traceback.print_exc()
            return 0.0
    
    def _calculate_response_relevance(self, query: str, response: Dict[str,Any],expected_answer: str=None) -> float:
        try:
            response_text = response.get("summary","")
            response_text = response_text.replace('\n','')
            if not isinstance(response_text,str) or not response_text.strip():
                return 0.0

            reference_text = expected_answer if expected_answer is not None and  expected_answer.strip() else query

            refernce_embedding = self.embedder.encode(reference_text,convert_to_tensor=True,normalize_embeddings=True)

            response_embedding = self.embedder.encode(response_text, convert_to_tensor=True, normalize_embeddings=True)
            similarity = util.cos_sim(refernce_embedding, response_embedding).item()

            response_length = len(response_text.split())

            reference_length = len(reference_text.split())
            length_penalty_factor = 1.0
            if response_length > (reference_length*1.5) and response_length >50:
                length_penalty_factor = max(0.8,1.0-(response_length-(reference_length*2))/100.0)
            elif response_length < (reference_length*0.5) and response_length >20:
                length_penalty_factor = max(0.5,response_length/(reference_length*0.5))

            adjusted_relevance = similarity * length_penalty_factor
            logs.logger.info(
                f"Relevance score for query '{query}': {adjusted_relevance:.2f} (similarity: {similarity:.2f}, penalty: {length_penalty_factor:.2f})")
            return adjusted_relevance
        except Exception as ex:
            logs.logger.info.error(f"Exception in relevance score calculation: {ex}")
            traceback.print_exc()
            return 0.0
    
    def _calculate_bert_score(self, response: Dict[str,Any], reference: str) -> float:
        try:
            response_text = response.get("summary", "")
            if not isinstance(response_text, str):
                logs.logger.warning("Response text is not a string")
                return 0.0
            if not response_text:
                logs.logger.error(f"No retrieved documents for BERT score evaluation")
                return 0.0
            if not reference.strip():
                logs.logger.warning(f"No reference answer provided for BERTScore calculation")
                return 0.0
            transformers.utils.logging.set_verbosity_error()
            _, _, f1 = score([response_text], [reference], lang="en", model_type="roberta-large")
            transformers.utils.logging.set_verbosity_warning()
            bert_score_value = f1.item()
            logs.logger.info(f"BERT score: {bert_score_value:.2f}")
            return bert_score_value
        except Exception as ex:
            logs.logger.error(f"Exception in BERT score calculation: {ex}")
            traceback.print_exc()
            return 0.0

    def evaluate_response(self, query: str, response: Dict[str,Any], retrieved_docs: List[Dict[str, Any]],
                         expected_keywords: List[str] = None, expected_answer: str = None) -> Dict[str, Any]:
        try:
            cosine_score = self._calculate_retrieval_relevance(query, retrieved_docs,expected_answer=expected_answer)
            groundedness = self._calculate_response_groundedness(response, retrieved_docs)
            relevance = self._calculate_response_relevance(query, response,expected_answer=expected_answer)
            bert_score = self._calculate_bert_score(response, expected_answer) if expected_answer else 0.0

            observations = []
            if groundedness < 0.6:
                observations.append(f"Response may contain ungrounded information (groundedness score: {groundedness:.2f})")
            else:
                observations.append(f"Response is well grounded (score: {groundedness:.2f})")
            
            if relevance < 0.7:
                observations.append(f"Response may not fully address the question (relevance score: {relevance:.2f})")
            else:
                observations.append(f"Response is highly relevant (score: {relevance:.2f})")
            
            if cosine_score < 0.5:
                observations.append(f"Low similarity between query and retrieved documents")
            
            if bert_score < 0.7 and expected_answer:
                observations.append(f"Low BERT score, semantic mismatch with reference answer")
            
            return {
                "cosine_score": round(cosine_score, 2),
                "groundedness": round(groundedness, 2),
                "relevance": round(relevance, 2),
                "bert_score": round(bert_score, 2),
                "observations": "; ".join(observations)
            }
        
        except Exception as ex:
            logs.logger.error(f"Exception in calculating response scores: {ex}")
            traceback.print_exc()
            return {
                "cosine_score": 0.0,
                "groundedness": 0.0,
                "relevance": 0.0,
                "bert_score": 0.0,
                "observations": f"Error in evaluation: {ex}"
            }

    def evaluate_combined_params_grid(self,
                                     chunk_size_to_test: List[int],
                                     chunk_overlap_to_test: List[int],
                                     embedding_models_to_test: List[str],
                                     vector_db_types_to_test: List[str],
                                      re_ranker_model: List[str],
                                     llm_model_name: str = Configuration.DEFAULT_GROQ_LLM_MODEL,
                                     search_type: str = "grid",
                                     n_iter: int = 50) -> Dict[str, Any]:
        logs.logger.info("\n--- Starting the evaluation of best parameters ---")
        best_score = -1.0
        best_params = {}
        best_metrics = {}
        results = []
        
        param_combination = [(c_size, c_overlap, embed_model, db_type,re_ranker)
                             for c_size, c_overlap, embed_model, db_type,re_ranker in product(
                                 chunk_size_to_test, chunk_overlap_to_test,
                                 embedding_models_to_test, vector_db_types_to_test,re_ranker_model)
                             if c_overlap < c_size]
        
        param_to_test = (random.sample(param_combination, min(n_iter, len(param_combination)))
                         if search_type.lower() == 'random' else param_combination)
        logs.logger.info(f"Testing {len(param_to_test)} {'random' if search_type.lower() == 'random' else 'all'} "
                    f"combinations out of {len(param_combination)}")

        for idx, (c_size, c_overlap, embed_model, db_type,re_ranker) in enumerate(param_to_test, 1):
            logs.logger.info('-'*50)
            logs.logger.info(f"\nIteration {idx}/{len(param_to_test)} \nchunk_size: {c_size} \nchunk_overlap: {c_overlap} "
                        f"\nembed_model: {embed_model} \nvector_db: {db_type}")

            current_params_str = f"Chunk: {c_size}-{c_overlap}- Embed- {embed_model}- DB-{db_type}-{re_ranker}"
            embed_model = embed_model.replace('_', '-')
            temp_collection_name = self._sanitize_collection_name(
                f"{Configuration.COLLECTION_NAME}-{search_type}-{c_size}-{c_overlap}-{embed_model}-{db_type}")
            temp_db_path = os.path.join(Configuration.DATA_DIR, f"{db_type}_temp_{search_type}_{c_size}_{c_overlap}_{embed_model}_{embed_model}")
            os.makedirs(temp_db_path, exist_ok=True)

            vector_db_instance = None
            try:
                embedder = SentenceTransformerEmbedder(model_name=embed_model)
                db_constructor = VECTOR_DB_CONSTRUCTORS.get(db_type.lower())
                if not db_constructor:
                    logs.logger.error(f"Unsupported vector DB type: {db_type}")
                    continue
                vector_db_instance = db_constructor(embedder=embedder,
                                                    db_path=temp_db_path,
                                                    collection_name=temp_collection_name)
                                                    
                
                chunker_instance = PyMuPDFChunker(pdf_path=self.pdf_path,
                                                 chunk_size=c_size,
                                                 chunk_overlap=c_overlap)
                
                llm_instance = GROQLLM(model_name=llm_model_name)

                pipeline = RAGPipeline(
                    document_path=self.pdf_path,
                    chunker=chunker_instance,
                    embedder=embedder,
                    vector_db=vector_db_instance,
                    llm=llm_instance,
                    chunk_size=c_size,
                    chunk_overlap=c_overlap,
                    embedding_model_name=embed_model,
                    llm_model_name=llm_model_name,
                    db_path=temp_db_path,
                    collection_name=temp_collection_name,
                    re_ranker_model_name=re_ranker
                )
                
                pipeline.build_index()
                if vector_db_instance.count_documents() == 0:
                    logs.logger.warning(f"No documents foundin vector DB after build for "
                                   f"{current_params_str}. skipping evaluation for this combination.")
                    continue

                cosine_scores = []
                groundedness_scores = []
                relevance_scores = []
                bert_scores = []

                for eval_item in self.eval_queries:
                    query = eval_item['query']
                    expected_answer = eval_item.get('expected_answer_snippet', '')

                    expected_keywords = eval_item.get('expected_keywords', [])
                    retrieved_docs = pipeline.retrieve_raw_documents(query, k=3)
                    response = pipeline.query(query, k=3)
                    logs.logger.debug(f"Query: {query} \n Response: {json.dumps(response,indent=2, ensure_ascii=False)}")
                    if not expected_answer.strip():
                        expected_answer = self._syntesize_raw_reference(retrieved_docs)


                    eval_result = self.evaluate_response(query, response,
                                            retrieved_docs,
                                            expected_answer=expected_answer,
                                            expected_keywords=expected_keywords)
                    
                    cosine_scores.append(eval_result['cosine_score'])
                    groundedness_scores.append(eval_result['groundedness'])
                    relevance_scores.append(eval_result['relevance'])
                    bert_scores.append(eval_result['bert_score'])

                average_cosine_score = round(sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0, 2)
                average_groundedness = round(sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0.0, 2)
                average_relevance = round(sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0, 2)
                average_bert_score = round(sum(bert_scores) / len(bert_scores) if bert_scores and any(bert_scores) else 0.0, 2)

                average_score = round((0.2* average_cosine_score +
                                       0.35* average_groundedness +
                                      0.35* average_relevance +
                                       0.15*average_bert_score), 2)
                
                results.append({
                    "iteration": idx,
                    "chunk_size": c_size,
                    "chunk_overlap": c_overlap,
                    "embedding_model": embed_model,
                    "vector_db_type": db_type,
                    "average_retrieval_relevance_score": average_score,
                    "average_cosine_score": average_cosine_score,
                    "average_groundedness": average_groundedness,
                    "average_relevance": average_relevance,
                    "average_bert_score": average_bert_score
                })
                
                logs.logger.info(f"Average retrieval relevance score: {average_score}")
                logs.logger.info(f"Average cosine score: {average_cosine_score}")
                logs.logger.info(f"Average groundedness: {average_groundedness}")
                logs.logger.info(f"Average relevance: {average_relevance}")
                logs.logger.info(f"Average BERT score: {average_bert_score}")

                if average_score > best_score:
                    best_score = average_score
                    best_params = {
                        "iteration":idx,
                        "chunk_size": c_size,
                        "chunk_overlap": c_overlap,
                        "embedding_model": embed_model,
                        "vector_db_type": db_type,
                        "re_ranker_model":re_ranker
                    }
                    best_metrics = {
                        "average_retrieval_relevance_score": average_score,
                        "average_cosine_score": average_cosine_score,
                        "average_groundedness": average_groundedness,
                        "average_relevance": average_relevance,
                        "average_bert_score": average_bert_score,
                        "re_ranker_model":re_ranker
                    }

                    logs.logger.info(f"Best score: {best_score}")
                    logs.logger.info(f"Best params: {best_params}")
            
            except (PineconeException, ValueError) as ex:
                logs.logger.error(f"Exception in grid search for chunk_size={c_size}, "
                            f"chunk_overlap={c_overlap}, embed_model={embed_model}, vector_db={db_type}: {ex}")
                traceback.print_exc()
            finally:
                if 'vector_db_instance' in locals() and vector_db_instance is not None:
                    try:
                        vector_db_instance.delete_collection(temp_collection_name)
                        if isinstance(vector_db_instance, chromaDBVectorDB):
                            del vector_db_instance.client
                            time.sleep(5)
                        if isinstance(vector_db_instance, PineconeVectorDB):
                            pc = Pinecone(api_key=Configuration.PINECONE_API_KEY)
                            if temp_collection_name in pc.list_indexes().names():
                                pc.delete_index(temp_collection_name)
                                logs.logger.info(f"Pinecone index: {temp_collection_name} deleted")
                    except Exception as cleanup_ex:
                        logs.logger.error(f"Error during cleanup: {cleanup_ex}")
                        traceback.print_exc()

        logs.logger.info("\n---- Evaluation completed ----")
        logs.logger.info(f"Best parameters: {best_params}")
        logs.logger.info(f"Best score: {best_score:.2f}")
        logs.logger.info(f"Total iterations evaluated: {len(results)}")

        pkl_directory = os.path.join(Configuration.DATA_DIR, "eval_results")
        os.makedirs(pkl_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        pkl_file = os.path.join(pkl_directory, f"eval_results_{search_type}_{timestamp}.pkl")

        try:
            joblib.dump({
                "best_params": best_params,
                "best_score": best_score,
                "results": results,
                "best_metrics": best_metrics
            }, pkl_file)
            logs.logger.info(f"Results saved to {pkl_file}")
        except Exception as ex:
            logs.logger.error(f"Exception in saving results to {pkl_file}: {ex}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "results": results,
            "best_metrics": best_metrics,
            "pkl_file": pkl_file
        }


    def _evaluate_with_llm(self, query:str, response_summary: str,retrieved_docs_contents: List[str]) -> Dict[str,Any]:
        try:
            context = "\n".join(retrieved_docs_contents)
            system_message = f"""
                You are an expert, Impartial judge for evaluating Retrieval Augmented Generation (RAG) system message.
                Your ONLY task is to output a JSON object containing the evaluation score and reasoning.
                DO NOT include any other text, explanations, conversational remarks or markdown code blocks(```json).
                Strictly adhere to the requested JSON format.
                """
            prompt = f"""
                You are evaluating a RAG system's response.
                
                Query: "{query}"
                Retrieved context: --- {context} ---
                RAG Systems Response: "{response_summary}"
                      
                please provide a score for Groundedness and Relevance on a scale of 1 to 5,
                where 5 is excellent and 1 is very poor.
                
                Groundedness: is how the response is supported *only* to the Retrieved context with  no hallucinations ?
                1. Contains significant information not supported by the context or contradicts it.
                2. Contains some unsupported information.
                3. Mostly grounded, but might have minor deviations or additions.
                4. Almost entirely grounded in the context.
                5. Fully and accurately, using only information from the context.
                
                Relevance: how well the response is directly and comprehensively answers the query based on the context ?
                1. Does not answer the query at all, or answers a different question.
                2. Addresses the query partially but misses significant part or is off-topic.
                3. Answer the query reasonably well, but could be more complete or focused.
                4. Answer the query well, covering most relevant aspects.
                5. Answer the query completely, accurately and concisely, directly addressing all aspects.
                
                output your assessment ONLY in the following JSON format. no other text.
                {{
                    "groundedness_score": <int> out of 5,
                    "relevance_score": <int> out of 5,
                    "reasoning": "Brief explanation for the scores."
                }}                                              
            """

            eval_with_llm = GROQLLM(model_name="llama-3.3-70b-versatile")
            llm_response =  eval_with_llm.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500)

            print("\n -- LLM Judge Raw Response --")
            print(llm_response)
            print('-'*50)

            eval_scores = json.loads(llm_response)
            return {
                "Groundedness score": eval_scores.get("groundedness_score",0),
                "Relevance score": eval_scores.get("relevance_score",0),
                "Reasoning": eval_scores.get("reasoning","")
            }
        except Exception as ex:
            logs.logger.error(f"Exception {ex}")
            logs.logger.error(traceback.print_exc())

            return {
                "Groundedness score": 0,
                "Relevance score": 0,
                "Reasoning": f"Exception in LLM evaluation: {ex}"
            }


    def _syntesize_raw_reference(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        try:
            if not retrieved_docs:
                return  ""
            raw_content_snippets = [doc.get('content','') for doc in retrieved_docs if doc.get('content')]
            raw_answer = " ".join(raw_content_snippets)

            raw_answer = " ".join(raw_answer.split()).join(sorted(list(set(raw_answer.split()))))

            return raw_answer
        except Exception as ex:
            logs.logger.error(f"Exception {ex}")
            logs.logger.error(traceback.print_exc())
            return ""
