import logs
import traceback
from typing import List, Dict, Any
from configuration import Configuration
from rag_scripts.interfaces import IDocumentChunker, IEmbedder, IVector, ILLM, IRAGPipeline
from sentence_transformers import CrossEncoder
from rag_scripts.documents_processing.chunking import PyMuPDFChunker
from rag_scripts.embedding.embedder import SentenceTransformerEmbedder
from rag_scripts.embedding.vector_db.chroma_db import chromaDBVectorDB
from rag_scripts.embedding.vector_db.faiss_db import FAISSVectorDB
from rag_scripts.embedding.vector_db.pinecone_db import PineconeVectorDB
from rag_scripts.llm.llmResponse import GROQLLM

class RAGPipeline(IRAGPipeline):

    def __init__(self,
                 document_path: str = Configuration.FULL_PDF_PATH,
                 chunker: IDocumentChunker = None,
                 embedder: IEmbedder = None,
                 vector_db: IVector = None,
                 llm: ILLM = None,
                 chunk_size: int = Configuration.DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = Configuration.DEFAULT_CHUNK_OVERLAP,
                 embedding_model_name: str = Configuration.DEFAULT_SENTENCE_TRANSFORMER_MODEL,
                 llm_model_name: str = Configuration.DEFAULT_GROQ_LLM_MODEL,
                 vector_db_type: str = "chroma",
                 db_path: str = None,
                 collection_name: str = None,
                 re_ranker_model_name: str = Configuration.DEFAULT_RERANKER
                 ):
        self.document_path = document_path

        self.chunker = chunker if chunker else PyMuPDFChunker(
            pdf_path = self.document_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.embedder = embedder if embedder else SentenceTransformerEmbedder(model_name = embedding_model_name)

        if vector_db:
            self.vector_db = vector_db
        
        else:
            if not isinstance(vector_db_type, str):
                raise ValueError("vector db type must be string")
            db_path = db_path or (
                Configuration.CHROMA_DB_PATH if vector_db_type.lower() == 'chroma'
                else
                Configuration.FAISS_DB_PATH if vector_db_type.lower() == 'faiss'
                else "" )
            collection_name = collection_name or Configuration.COLLECTION_NAME

            if vector_db_type.lower() == "chroma":
                self.vector_db = chromaDBVectorDB(
                    embedder = self.embedder,
                    db_path=db_path,
                    collection_name=collection_name
                )
            elif vector_db_type.lower() == "faiss":
                self.vector_db = FAISSVectorDB(
                    embedder=self.embedder,
                    db_path=db_path,
                    collection_name=collection_name
                )
            elif vector_db_type.lower() == "pinecone":
                self.vector_db = PineconeVectorDB(
                    embedder=self.embedder,
                    db_path=db_path,
                    collection_name=collection_name
                )
            else:
                raise ValueError("RAG application suppots chroma or faiss db")
            
        self.llm = llm if llm else GROQLLM(
            api_key= Configuration.GROQ_API_KEY,
            model_name=llm_model_name )

        self.re_ranker = None
        if re_ranker_model_name:
            try:
                self.re_ranker = CrossEncoder(re_ranker_model_name=re_ranker_model_name)
                logs.logger.info(f"ReRanker model loaded: {re_ranker_model_name}")
            except Exception as ex:
                logs.logger.info(f"ReRanker model could not be loaded: {re_ranker_model_name}")
                self.re_ranker = None
        
        logs.logger.info("RAG pipeline initialized")

    def build_index(self):
        logs.logger.info(f"building index for {self.document_path}")
        try:
            document_hash = self.chunker.hash_document()
            logs.logger.info(f"{document_hash}")
            existing_chunk_ids = self.vector_db.get_document_hash_ids(document_hash)

            if existing_chunk_ids:
                logs.logger.info(f"Documents {self.document_path} hash: {document_hash[:8]} already present in the vector DB with {len(existing_chunk_ids)} chunks.")
                return
            
            logs.logger.info(f"Chunking starting")
            document_hash, chunks = self.chunker.chunk_documents()

            if not chunks:
                logs.logger.info(f"No chunks generated for {self.document_path} index not built")
                return
            
            self.vector_db.add_chunks(chunks)

            logs.logger.info(f"Index built successfully for the {self.document_path} with {len(chunks)}")


        except FileNotFoundError:
            logs.logger.info(f"Exception Document not at {self.document_path}")
            traceback.print_exc()
        except Exception as ex:
            logs.logger.info(f"Exception in build index {ex}")
            traceback.print_exc()

    def retrieve_raw_documents(self, user_query: str, k: int =5) -> List[Dict[str,Any]]:
        if self.vector_db.count_documents() == 0:
            logs.logger.info("vector database is empty please build index first")
            return []

        query_embedding = user_query
        initial_retrieval_k = k*3 if self.re_ranker else k
        retrieved_docs_with_score = self.vector_db.search(query_embedding, k=initial_retrieval_k)
        if not retrieved_docs_with_score:
            logs.logger.info(f"Retrieval failed for {user_query}")
            return []
        retrieved_docs = [doc for doc in retrieved_docs_with_score]
        if self.re_ranker:
            document_content = [doc['content'] for doc in retrieved_docs]
            rerank_score = self.re_ranker.rerank(document_content)
            doc_with_reRank = []
            for idx, doc in enumerate(retrieved_docs):
                doc_with_reRank.append({
                    'doc': doc,
                    'rerank_score': rerank_score[idx]
                })
            ranked_docs = sorted(doc_with_reRank, key=lambda x: x['rerank_score'], reverse=True)
            final_retrieved_docs = [item['doc'] for item in ranked_docs[:k]]
        else:
            final_retrieved_docs = retrieved_docs
        return final_retrieved_docs

    def query(self, user_query: str, k: int=3,
              include_metadata: bool = True,
              user_context: Dict[str,Any]=None) -> Dict[str,Any]:
        if not user_query.strip():
            return {"summary": "Enter the query", "sources": []}
        
        retrieved_docs = self.retrieve_raw_documents(user_query,k)

        if not retrieved_docs:
            return {"summary": "Unable to find relevant information in the documents for the query asked. "
                               "Please refer contact HR department directly or refer HR policy document",
                    "sources": []}



        context_info =[]
        metadata_info = []

        for indx, doc in enumerate(retrieved_docs):
            context_info.append(f"Document {indx+1} content: {doc['content']}")
            if include_metadata:
                metadata = doc['metadata']
                metadata_info.append({
                 "document_id": f"DOC {indx+1}",
                 "page": str(metadata.get("page_number","NA")),
                "section": metadata.get("section","NA"),
                "clause": metadata.get("clause","NA") })
        
        context_string = "\n".join(context_info)
        user_context = user_context or {"role":"general", "location":"chennai","department":"unknown"}
        context_description = (f" for a {user_context['role']} in"
                               f"{user_context['location']} and {user_context['department']}")



        prompt = (
            f"You are an expert assistant for Flykite Airlines HR Policy queries. "
            f"Answer the question '{user_query}' based solely on the provided context from the Flykite Airlines HR Policy, "
            f"Tailor the answer for a {user_context['role']} in {user_context['location']} and {user_context['department']}. "
            f"Include only the criteria and details that directly address the question, ensuring all relevant points from the context are covered without adding unrelated information or assumptions. "
            f"Present the answer in a concise format using bullet points, a table, or sections for readability, and cite specific sections and clauses from the sources where applicable. "
            f"Cite specific sections and clauses from the sources using the format (soruce: DOC X, Page: Y, Section: Z, Clause: A) at the end of each relevant point or page"
            f"If the query is ambiguous, ask for clarification. If the context does not fully address the question, state what is known and suggest consulting the full HR Policy or HR department. "
            f"Context: \n{context_string}\n\n"
            f"Answer: "  )
        
        llm_response = self.llm.generate_response(prompt)

        if llm_response is None:
            return {"summary": "Unable to find relevant information in the documents for the query asked. ",
                    "sources": [] }

        final_response = {"summary": llm_response.strip(),
                          "sources":metadata_info if include_metadata else []

                          }



        return final_response

        
