import streamlit
import os
import sys
import json
import argparse
import warnings
import traceback
import logs
import chromadb
import hashlib
import sqlite3
import regex as re
from pinecone import Pinecone
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer, util

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings("ignore")

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'src')))
from sentence_transformers import SentenceTransformer
from configuration import Configuration
from rag_scripts.rag_pipeline import RAGPipeline
from rag_scripts.documents_processing.chunking import PyMuPDFChunker
from rag_scripts.embedding.embedder import SentenceTransformerEmbedder
from rag_scripts.embedding.vector_db.chroma_db import chromaDBVectorDB
from rag_scripts.embedding.vector_db.faiss_db import FAISSVectorDB
from rag_scripts.embedding.vector_db.pinecone_db import PineconeVectorDB
from rag_scripts.llm.llmResponse import GROQLLM
from rag_scripts.evaluation.evaluator import RAGEvaluator

class RAGOperations:
    VALID_VECTOR_DB = {'chroma','faiss','pinecone'}

    @staticmethod
    def check_db(vector_db_type: str, db_path: str, collection_name: str) -> bool:
        try:
            if vector_db_type not in RAGOperations.VALID_VECTOR_DB:
                logs.logger.info(f"Invalid Vector DB: {vector_db_type}")
                raise
            if vector_db_type.lower() == 'pinecone':
                pc = Pinecone(api_key=Configuration.PINECONE_API_KEY)
                return collection_name in pc.list_indexes().names()
            elif vector_db_type.lower() == 'chroma':
                return os.path.exists(db_path) and os.listdir(db_path)
            elif vector_db_type.lower() == "faiss":
                faiss_index_file = os.path.join(db_path,f"{collection_name}.faiss")
                faiss_doc_store_file = os.path.join(db_path,f"{collection_name}_docs.pkl")
                return os.path.exists(faiss_index_file) and os.path.exists(faiss_doc_store_file)
        except Exception as ex:
            logs.logger.info(f"Exception in checking {vector_db_type} existence")
            logs.logger.info(traceback.print_exc())
            return False

    @staticmethod
    def get_pipeline_params(args: argparse.Namespace, use_tuned: bool = False) -> Dict[str,Any]:
        try:
            best_param_path = os.path.join(Configuration.DATA_DIR,'best_params.json')
            params = {
                'document_path':Configuration.FULL_PDF_PATH,
                'chunk_size':args.chunk_size,
                'chunk_overlap':args.chunk_overlap,
                'embedding_model_name':args.embedding_model,
                'vector_db_type':args.vector_db_type,
                'llm_model_name':args.llm_model,
                'db_path': None,
                'collection_name': Configuration.COLLECTION_NAME,
                'vector_db': None,
                'temperature': args.temperature,
                'top_p':args.top_p,
                'max_tokens':args.max_tokens,
                're_ranker_model':args.re_ranker_model
            }

            if os.path.exists(best_param_path):
                with open(best_param_path,'rb') as f:
                    best_params = json.load(f)
                logs.logger.info(f"Best params: {best_params} from the file {best_param_path}")

                params.update({
                    'vector_db_type': best_params.get('vector_db_type',params['vector_db_type']),
                    'embedding_model_name': best_params.get('embedding_model',params['embedding_model_name']),
                    'chunk_overlap': best_params.get('chunk_overlap',params['chunk_overlap']),
                    'chunk_size': best_params.get('chunk_size',params['chunk_size']) ,
                    're_ranker_model': best_params.get('re_ranker_model',params['re_ranker_model']) })
                use_tuned = True

            if use_tuned:
                tuned_db_type = params['vector_db_type']
                params['db_path'] = os.path.join(Configuration.DATA_DIR,'TunedDB',tuned_db_type) if tuned_db_type != 'pinecone' else ""
                params['collection_name'] = 'tuned-'+Configuration.COLLECTION_NAME
                if tuned_db_type in  ['chroma','faiss']:
                    os.makedirs(params['db_path'],exist_ok=True)
                logs.logger.info(f"Tuned db path: {params['db_path']}")
            else:
                params['db_path'] = ( Configuration.CHROMA_DB_PATH if params['vector_db_type'] == 'chroma'
                                      else Configuration.FAISS_DB_PATH if params['vector_db_type'] == 'faiss'
                                      else "")
                if params['vector_db_type'] in ['chroma', 'faiss']:
                    os.makedirs(params['db_path'],exist_ok=True)
                    logs.logger.info(f"Created directory for {params['vector_db_type']} at {params['db_path']}")

            return params
        except Exception as ex:
            logs.logger.info(f"Exception in get_pipeline_params: {ex}")
            logs.logger.info(traceback.print_exc())
            sys.exit(1)


    @staticmethod
    def check_embedding_dimension(vector_db_type: str,db_path: str,
                                  collection_name: str, embedding_model: str) -> bool:
        if vector_db_type !='chroma':
            return True
        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(collection_name)
            model = SentenceTransformer(embedding_model)
            sample_embedding = model.encode(["test"])[0]
            try:
                expected_dim = collection._embedding_function.dim
            except AttributeError:
                peek_result = collection.peek(limit=1)
                if 'embedding' in peek_result and peek_result['embedding']:
                    expected_dim = len(peek_result['embedding'][0])
                else:
                    return False
            actual_dim = len(sample_embedding)
            logs.logger.info(f"Expected dimension: {expected_dim} Actual dimension: {actual_dim}")
            return expected_dim == actual_dim
        except Exception as ex:
            logs.logger.info(f"Error checking embedding dimension: {ex}")
            return False


    @staticmethod
    def initialize_pipeline(params: dict[str,Any]) -> RAGPipeline:
        try:
            embedder = SentenceTransformerEmbedder(model_name=params['embedding_model_name'])
            chunkerObj = PyMuPDFChunker(
                pdf_path=params['document_path'],
                chunk_size=params['chunk_size'],
                chunk_overlap=params['chunk_overlap'])
            llm_model = params['llm_model_name']
            vector_db = None
            if params['vector_db_type'] == 'chroma':
                vector_db = chromaDBVectorDB(embedder=embedder,
                                db_path=params['db_path'],
                                collection_name=params['collection_name'])
            elif params['vector_db_type'] == 'faiss':
                vector_db = FAISSVectorDB(embedder=embedder,
                                db_path=params['db_path'],
                                collection_name=params['collection_name'] )
            elif params['vector_db_type'] == 'pinecone':
                vector_db = PineconeVectorDB(embedder=embedder,
                                db_path=params['db_path'],
                                collection_name=params['collection_name'])
            else:
                raise ValueError(f"Unknown vector_db_type: {params['vector_db_type']}")

            return RAGPipeline( document_path=params['document_path'],
                chunker=chunkerObj, embedder=embedder,
                vector_db=vector_db,
                llm=GROQLLM(model_name= llm_model),
                re_ranker_model_name=params['re_ranker_model'] if params['re_ranker_model'] else Configuration.DEFAULT_RERANKER,)
        except Exception as ex:
            logs.logger.info(f"Exception in pipeline initialize: {ex}")
            traceback.print_exc()
            sys.exit(1)

    @staticmethod
    def run_build_job(args: argparse.Namespace) -> None:
        try:
            params = RAGOperations.get_pipeline_params(args)
            pipeline = RAGOperations.initialize_pipeline(params)
            pipeline.build_index()
            logs.logger.info(f"RAG Build JOB completed")
        except Exception as ex:
            logs.logger.info(f"Exception in run build job: {ex}")
            traceback.print_exc()
            sys.exit(1)


    @staticmethod
    def run_search_job(args: argparse.Namespace,user_info: Dict[str,str]) -> None:
        try:
            params = RAGOperations.get_pipeline_params(args, use_tuned=args.use_tuned)
            vector_db_type = params['vector_db_type']
            db_path = params['db_path']
            collection_name = params['collection_name']

            pipeline = RAGOperations.initialize_pipeline(params)
            db_exists = RAGOperations.check_db(vector_db_type,db_path,collection_name)

            if args.use_rag:
                if not db_exists:
                    pipeline.build_index()
                elif pipeline.vector_db.count_documents() == 0:
                    pipeline.build_index()
                elif not RAGOperations.check_embedding_dimension(vector_db_type,db_path,
                                                                 collection_name,params['embedding_model_name'] ):
                    logs.logger.info(f"Embedding dimension mismatch. rebuilding the index")
                    pipeline.vector_db.delete_collection(collection_name)
                    pipeline.build_index()

                else:
                    logs.logger.info(f"Using existing {vector_db_type} database with collection: {collection_name}")

                if pipeline.vector_db.count_documents() == 0:
                    logs.logger.info(f"No Documents found in vector database after re-build")
                    sys.exit(1)

            evaluator = RAGEvaluator(eval_data_path=Configuration.EVAL_DATA_PATH,
                                     pdf_path=Configuration.FULL_PDF_PATH)

            user_query = args.query if args.query else  (
                            input("Enter your Query: "))
            if user_query.lower() == 'exit':
                return
            user_context = {"role": user_info['role'],
                            "location": user_info['location'],
                            "department":user_info['department'] }

            expected_answers = None
            expected_keywords = []
            query_found = False
            try:
                with open(Configuration.EVAL_DATA_PATH, 'r') as f:
                    eval_data = json.load(f)
                for item in eval_data:
                    if item.get('query').strip().lower() == user_query.strip().lower():
                        expected_keywords = item.get('expected_keywords',[])
                        expected_answers = item.get('expected_answer_snippet',"")
                        query_found = True
                        break
                if not expected_keywords and not expected_answers:
                    logs.logger.info(f"No evaluation data found for query in json")
            except Exception as ex:
                logs.logger.info(f"No json file : {ex}")
            retrieved_documents = []
            if args.raw:
                retrieved_documents = pipeline.retrieve_raw_documents(
                                        user_query, k=args.k*2)
                logs.logger.info("Raw documents retrieved")
                logs.logger.info(json.dumps(retrieved_documents, indent=4))
                if not retrieved_documents:
                    response ={"summary":"No relevant documents found",
                               "sources":[]}
                else:

                    query_embedding = evaluator.embedder.encode(user_query,
                                            convert_to_tensor=True,normalize_embeddings=True)
                    similarities = [(doc, util.cos_sim(query_embedding,
                                                       evaluator.embedder.encode(doc['content'],
                                                                    convert_to_tensor=True,
                                                                    normalize_embeddings=True)).item())
                                    for doc in retrieved_documents]
                    similarities.sort(key=lambda x: x[1], reverse=True)

                    top_docs = similarities[:min(3, len(similarities))]

                    truncated_content = []
                    for doc, sim in top_docs:
                        content_paragraphs = re.split(r'\n\s*\n', doc['content'].strip())
                        para_sims = [(para, util.cos_sim(query_embedding,
                                                         evaluator.embedder.encode(para.strip(), convert_to_tensor=True,
                                                                                   normalize_embeddings=True)).item())
                                     for para in content_paragraphs if para.strip()]
                        para_sims.sort(key=lambda x: x[1], reverse=True)

                        top_paras = [para for para, para_sim in para_sims[:2] if para_sim >= 0.3]
                        if len(top_paras) < 1:  # Fallback to at least one paragraph
                            top_paras = [para for para, _ in para_sims[:1]]
                        truncated_content.append('\n\n'.join(top_paras))

                    response = {
                    "summary": "\n".join(truncated_content),
                    "sources":[{ "document_id":f"DOC {idx+1}",
                        "page": str(doc['metadata'].get("page_number","NA")),
                        "section": doc['metadata'].get("section","NA"),
                        "clause": doc['metadata'].get("clause","NA")}
                        for idx,(doc,_) in enumerate(top_docs)] }

            else:
                logs.logger.info("LLM+RAG")
                response = pipeline.query(user_query, k=args.k,
                                          include_metadata=True,
                                          user_context=user_context
                                          )
                retrieved_documents = pipeline.retrieve_raw_documents(
                    user_query, k=args.k)


            final_expected_answer = expected_answers if expected_answers is not None else ""
            additional_eval_metrices = {}
            if not query_found:
                logs.logger.info(f"No query found in eval_Data.json: {user_query}")
                raw_reference_for_score = evaluator._syntesize_raw_reference(retrieved_documents)
                if not final_expected_answer.strip():
                    final_expected_answer = raw_reference_for_score

                retrieved_documents_content = [doc.get('content','') for doc in retrieved_documents]
                llm_as_judge = evaluator._evaluate_with_llm(user_query, response.get('summary',''),retrieved_documents_content)
                if llm_as_judge:
                    additional_eval_metrices.update(llm_as_judge)
                    output = {"query": user_query, "response": response, "evaluation": additional_eval_metrices}
                    logs.logger.info(json.dumps(output, indent=4))
                    return json.dumps(output)
                else:
                    output = { "query": user_query, "response":response, "evaluation":llm_as_judge }
                    logs.logger.info(json.dumps(output, indent=4))
                    return json.dumps(output)

            else:

                eval_result = evaluator.evaluate_response(user_query, response, retrieved_documents,
                                                expected_keywords, expected_answers)
                output = { "query": user_query, "response":response, "evaluation":eval_result }
                logs.logger.info(json.dumps(output,indent=2,ensure_ascii=False))

                return json.dumps(output)


        except Exception as ex:
            logs.logger.info(f"Exception in run search job {ex}")
            traceback.print_exc()

    @staticmethod
    def run_hypertune_job(args: argparse.Namespace) -> None:
        try:
            evaluator = RAGEvaluator(eval_data_path=Configuration.EVAL_DATA_PATH,
                                     pdf_path=Configuration.FULL_PDF_PATH)

            result = evaluator.evaluate_combined_params_grid(
                chunk_size_to_test=[512,1024,2048],
                chunk_overlap_to_test=[100,200,400],
                embedding_models_to_test=["all-MiniLM-L6-v2",
                                         "all-mpnet-base-v2",
                                         "paraphrase-MiniLM-L3-v2",
                                         "multi-qa-mpnet-base-dot-v1" ],
                vector_db_types_to_test=['pinecone','chroma','faiss'],
                llm_model_name=args.llm_model,
                re_ranker_model = [ "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "cross-encoder/ms-marco-TinyBERT-L-2"],
                search_type='random', n_iter=3 )
            
            best_parameter = result['best_params']
            best_score = result['best_score']
            pkl_file = result['pkl_file']
            best_metrics = result['best_metrics']

            best_param_path = os.path.join(Configuration.DATA_DIR,'best_params.json')

            with open(best_param_path, 'w') as f:
                json.dump(best_parameter, f, indent=4)

            tuned_db = best_parameter['vector_db_type']
            tuned_path = os.path.join(Configuration.DATA_DIR,'TunedDB',tuned_db)
            if tuned_db != 'pinecone':
                os.makedirs(tuned_path, exist_ok=True)
            tuned_collection_name = "tuned-"+Configuration.COLLECTION_NAME

            tuned_params = {
                'document_path': Configuration.FULL_PDF_PATH,
                'chunk_size': best_parameter.get('chunk_size', Configuration.DEFAULT_CHUNK_SIZE),
                'chunk_overlap': best_parameter.get('chunk_overlap',Configuration.DEFAULT_CHUNK_OVERLAP),
                'embedding_model_name': best_parameter.get('embedding_model',Configuration.DEFAULT_SENTENCE_TRANSFORMER_MODEL),
                'vector_db_type': tuned_db,
                'llm_model_name':args.llm_model,
                'db_path':tuned_path if tuned_db !='pinecone' else "",
                'collection_name':tuned_collection_name,
                'vector_db': None,
                're_ranker_model':best_parameter.get('re_ranker', Configuration.DEFAULT_RERANKER)
            }

            if 're_ranker_model' in best_parameter:
                tuned_params['re_ranker_model'] = best_parameter['re_ranker_model']
            else:
                tuned_params['re_ranker_model'] = Configuration.DEFAULT_RERANKER

            tuned_pipeline = RAGOperations.initialize_pipeline(tuned_params)
            tuned_pipeline.build_index()

        except Exception as ex:
            logs.logger.info(f"Exception in hypertune: {ex} ")
            traceback.print_exc()


    @staticmethod
    def run_llm_with_prompt(args: argparse.Namespace,run_type: str) -> None:
        try:
            params = RAGOperations.get_pipeline_params(args,
                                        use_tuned=args.use_tuned)
            pipeline = RAGOperations.initialize_pipeline(params)


            evaluator = RAGEvaluator(eval_data_path=Configuration.EVAL_DATA_PATH,
                                     pdf_path=Configuration.FULL_PDF_PATH)

            system_message = (
                "You are an expert assistant for Flykite Airlines HR Policy Queries."
                "Provide concise, accurate and policy-specific answers based solely on the the provided context."
                "Structured your response clearly, using bullet points, newlines if applicable. "
                "If the context lacks information, state that clearly and speculation."
            ) if run_type == 'prompting' else None

            user_query = input("Enter your query: ")
            expected_answer = None
            expected_keywords = []
            try:
                with open(Configuration.EVAL_DATA_PATH, 'r') as f:
                    eval_data= json.load(f)
                for item in eval_data:
                    expected_answer = item.get('expected_answer_snippet',"")
                    expected_keywords = item.get('expected_keywords',[])
                    break
            except Exception as ex:
                logs.logger.info(f"Error loading eval_data.json for query {user_query}: {ex}")

            if run_type == 'prompting':
                prompt = (
                    f"You are an expert assistant for Flykite Airlines HR Policy Queries."
                    f"Answer the following question with a structured response, using bullet points or sections where applicable"
                    f"Base your answer solely on the query and avoid hallucination"
                    f"Question: \n {user_query} \n"
                    f"Answer: ")

            else:
                prompt = user_query

            response = pipeline.llm.generate_response(
                    prompt=prompt,
                    system_message=system_message,
                    temperature = args.temperature,
                    top_p = args.top_p,
                    max_tokens = args.max_tokens
                )
            retreived_documents = []

            eval_result = evaluator.evaluate_response(user_query,
                                                response,
                                                retreived_documents,
                                                expected_keywords,
                                                expected_answer)

            output = {  "query":user_query,
                        "response": {
                             "summary: ":response.strip(),
                            "source: ":["LLM Response Not RAG loaded"]},
                        "evaluation": eval_result }


            logs.logger.info(json.dumps(output, indent=2))

        except Exception as ex:
            logs.logger.info(f"Exception in LLm_prompting response: {ex}")
            traceback.print_exc()
            sys.exit(1)

    @staticmethod
    def login() -> Dict[str,str]:
        username = input("Enter your username: ")
        password = input("Enter your password: ")

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username,jobrole,department,location FROM users WHERE username = ? AND password = ?",
                (username, hashed_password)
            )
            user = cursor.fetchone()
            logs.logger.info(f"{user}")
            conn.close()
            if user:
                return {"username": user[0], "role": user[1],"department": user[2],"location": user[3]}
            else:
                logs.logger.info("Invalid username or password")
                sys.exit(1)

        except sqlite3.Error as ex:

            return {"username": None, "role": None,"department": None,"location": None}





def main():

    user_info = RAGOperations.login()
    print(f"Logged in as {user_info['username']}"
          f"{user_info['department']} at {user_info['location']}")


    # user_info = {
    # "username": "admin",
    # "role": "admin",  # or "admin" if you need to test admin features
    # "department": "admin",
    # "location": "Chennai"
    # }

    parser = argparse.ArgumentParser(description='RAG PIPELINE OPERATIONS')

    parser.add_argument('--job',type=str, required=True,
                        choices=['rag-build','search','eval-hypertune','llm','prompting'],
                        help='job to execute build RAG index, search and hypertune')

    parser.add_argument('--raw', action='store_true',
                        help='dispaly raw retrieved documents in json format')
    parser.add_argument('--chunk_size', type=int, default=Configuration.DEFAULT_CHUNK_SIZE,help="Document chunking")
    parser.add_argument('--chunk_overlap', type=int, default=Configuration.DEFAULT_CHUNK_OVERLAP,help="Document overlap")
    parser.add_argument('--embedding_model', type=str, default=Configuration.DEFAULT_SENTENCE_TRANSFORMER_MODEL,help="Embedding model")
    parser.add_argument('--vector_db_type', type=str,default="chroma",choices=['chroma','faiss','pinecone'],help='vector database type')
    parser.add_argument('--llm_model',type=str, default=Configuration.DEFAULT_GROQ_LLM_MODEL)
    parser.add_argument('--use-tuned',action='store_true',help='Use the tuned DB for search')
    parser.add_argument('--k',type=int, default=5,help='number of doc to retrieve')
    parser.add_argument('--query',type=str, default=None,help='query from user')
    parser.add_argument('--temperature', type=float, default=0.1,help='LLM temperature for response generation')
    parser.add_argument('--top_p', type=float, default=0.95, help='LLM top_p for response generation')
    parser.add_argument('--max_tokens', type=int, default = 1000, help='LLM max tokens for response generation')
    parser.add_argument('--use-rag', type=lambda x: x.lower() =='true',default=True, help ='llm or prompting')
    parser.add_argument('--user-context', type=str, default=None, help='user context as JSON e.g., {"role": "admin", "location":"chennai", "designation":"engineer"}')
    parser.add_argument('--fine-tune', action='store_true', help='use fine-tuned model')
    parser.add_argument('--n-iter',type=int, default=10,help='number of iterations')
    parser.add_argument('--re_ranker_model',type=str,default=Configuration.DEFAULT_RERANKER,help="ReRanking the retrieval documents")
    args = parser.parse_args()

    if args.job == 'eval-hypertune' and user_info['role']!= 'admin':
        print(f"Access denied: Hypertune execution is forbidden")

    if args.job == 'rag-build':
        RAGOperations.run_build_job(args)
    elif args.job == 'search':
        RAGOperations.run_search_job(args,user_info)
    elif args.job == 'eval-hypertune':
        RAGOperations.run_hypertune_job(args)
    elif args.job in ['llm','prompting']:
        RAGOperations.run_llm_with_prompt(args, args.job)


if __name__ == '__main__':
    main()  
