
import chromadb
import logs
import traceback
from typing import List, Dict, Any
from chromadb.api.types import EmbeddingFunction
from configuration import Configuration
from rag_scripts.interfaces import IVector
from rag_scripts.interfaces import IEmbedder
from rag_scripts.embedding.embedder import SentenceTransformerEmbedder

class chromaDBEmbeddingFunction(EmbeddingFunction):

    def __init__(self, embedder: IEmbedder):
        self.embedder = embedder
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_texts(texts)

class chromaDBVectorDB(IVector):
    def __init__(self, embedder:IEmbedder, db_path: str = Configuration.CHROMA_DB_PATH, collection_name: str = Configuration.COLLECTION_NAME):
        super().__init__(embedder, db_path, collection_name)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.chroma_embed_function = chromaDBEmbeddingFunction(self.embedder)
        self.collection = self._get_or_create_collection()
        logs.logger.info(f"Chroma DB intialized path = {self.db_path}, collection = {self.collection_name}")

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_embed_function)
    
    
    def add_chunks(self,documents: List[Dict[str,Any]]) -> List[str]:
        try:
            if not documents:
                return []
            ids = [doc['metadata']['chunk_id'] for doc in documents]
            contents = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]

            self.collection.add(documents=contents, metadatas=metadatas,ids=ids)

            logs.logger.info(f"Added {len(ids)} chunks to chroma db")
            return ids
        except Exception as ex:
            logs.logger.info(f"Exception in adding chunks to chromaDB: {ex}")
            traceback.print_exc
            return[]
    
    def get_document_hash_ids(self, document_hash: str) -> List[str]:

        try:
            result = self.collection.get(where={"document_id":document_hash},limit=1)
            return result['ids'] if result and result['ids'] else []
        except Exception as ex:
            logs.logger.info(f"Exception getting document hash {document_hash} from chromadb {ex}")
            traceback.print_exc
            return[]

    def search(self, query: str, k: int=3) -> List[Dict[str, Any]]:
        try:
            search_result = self.collection.query(
                query_texts = [query],
                n_results=k,
                include=['documents','metadatas','distances']
            )
            retrieved_documents =[]
            if search_result and search_result['documents'] and search_result['metadatas']:
                for indx in range(len(search_result['documents'][0])):
                    document_content = search_result['documents'][0][indx]
                    document_metadata = search_result['metadatas'][0][indx]
                    document_distance = search_result['distances'][0][indx]

                    retrieved_documents.append({
                        "content": document_content,
                        "metadata": document_metadata,
                        "distance": document_distance })
            logs.logger.info(f"Rettrieved {len(retrieved_documents)} documents from chroma DB for query: '{query}'")

            return retrieved_documents
        except Exception as ex:
            logs.logger.info(f"Exception in Chroma db search: {ex}")
            traceback.print_exc()
            return []  
        
    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name = collection_name)
            logs.logger.info(f"Chroma DB collection {collection_name} deleted")
            self.collection = self._get_or_create_collection()
        except Exception as ex:
            logs.logger.info(f"Exception in deleting the chroma db collection: {collection_name}")
            traceback.print_exc()
    
    def count_documents(self) -> int:
        try:
            return self.collection.count()
        except Exception as ex:
            logs.logger.info(f"Exception in counting documents in chroma db: {ex}")
            traceback.print_exc()
            return 0
