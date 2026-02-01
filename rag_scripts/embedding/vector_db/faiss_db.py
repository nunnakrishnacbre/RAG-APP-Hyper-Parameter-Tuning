import os
import logs
import faiss
import pickle
import traceback
import numpy as np
from typing import List, Dict, Any

from configuration import Configuration
from rag_scripts.interfaces import IVector, IEmbedder
from rag_scripts.embedding.embedder import SentenceTransformerEmbedder

class FAISSVectorDB(IVector):

    def __init__(self, embedder: IEmbedder, db_path: str= Configuration.FAISS_DB_PATH,
                 collection_name: str = Configuration.COLLECTION_NAME):
        super().__init__(embedder,db_path,collection_name)
        self.index = None
        self.doc_store: List[Dict[str,Any]] =[]

        self.collection_file = os.path.join(self.db_path,f"{self.collection_name}.faiss")
        self.doc_store_file = os.path.join(self.db_path, f"{self.collection_name}_docs.pkl")

        os.makedirs(self.db_path, exist_ok=True)

        self._load_index()

        logs.logger.info(f"FAISSDB Initalized: path ='{self.db_path}, collection ='{self.collection_name}")

    
    def _load_index(self):
        try:
            if os.path.exists(self.collection_file) and os.path.exists(self.doc_store_file):
                self.index = faiss.read_index(self.collection_file)
                with open(self.doc_store_file,'rb') as f:
                    self.doc_store = pickle.load(f)
                logs.logger.info(f"Loaded Existing FAISS index and doc store from {self.collection_file}")
            else:
                logs.logger.info(f"No Existing FAISS index found, need new collection")
        except Exception as ex:
           logs.logger.info(f"Exception in loading FAISS index: {ex}")
           traceback.print_exc()
           self.index = None
           self.doc_store = []
    
    def _save_index(self):
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.collection_file)
                with open(self.doc_store_file, "wb") as f:
                    pickle.dump(self.doc_store,f)
                logs.logger.info(f"FAISS index and doc store saved to {self.collection_file}")

        except Exception as ex:
            logs.logger.info(f"Exception in saving FAISS index: {ex}")
            traceback.print_exc()

    def add_chunks(self, documents: List[Dict[str, Any]]) -> List[str]:
        try:
            if not documents:
                return[]
            
            contents = [doc['content'] for doc in documents]
            embeddings = np.array(self.embedder.embed_texts(contents),dtype='float32')

            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                logs.logger.info(f"Created New FAISS index with dimension {dimension}")

            self.index.add(embeddings)

            for doc in documents:
                self.doc_store.append(doc)
            
            self._save_index()
            chunk_ids = [doc['metadata']['chunk_id'] for doc in documents]
            logs.logger.info(f"Added {len(chunk_ids)} chunks to FAISS index")
            return chunk_ids
        except Exception as ex:
            logs.logger.info(f"Exception in adding chunks to FAISS: {ex}")
            traceback.print_exc()
            return []

    def get_document_hash_ids(self, document_hash: str) ->List[str]:
        try:
            found_ids =[]
            for doc in self.doc_store:
                if doc['metadata'].get('document_id') == document_hash:
                    found_ids.append(doc['metadata']['chunk_id'])
        
            return found_ids
        except Exception as ex:
            logs.logger.info(f"Exception in get document hash {ex}")

    def search(self, query: str, k:int=3) -> List[Dict[str,Any]]:
        try:
            if self.index is None or self.index.ntotal ==0:
                logs.logger.info(f"FAISS index is empty or not intialized")
                return []
            query_embedding = np.array(self.embedder.embed_query(query),dtype='float32').reshape(1,-1)
            distances, indices = self.index.search(query_embedding,k)
            
            retrieved_documents = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.doc_store):
                    doc = self.doc_store[idx]
                    retrieved_documents.append({
                        "content":doc['content'],
                        "metadata":doc['metadata'],
                        "distance":float(dist) })
            logs.logger.info(f"Retrieved {len(retrieved_documents)} documents from FAISS for query: {query}")

            return retrieved_documents


        except Exception as ex:
            logs.logger.info(f"Exception in FAISS db search {ex}")
            traceback.print_exc()
            return []

    def delete_collection(self,collection_name: str):
        try:
            if os.path.exists(self.collection_file):
                os.remove(self.collection_file)
            if os.path.exists(self.doc_store_file):
                os.remove(self.doc_store_file)
            self.index = None
            self.doc_store = []
            logs.logger.info(f"FAISS collection files for {collection_name} deleted")

        except Exception as ex:
            logs.logger.info(f"Error deleting FAISS collection files for {collection_name}: ex")
            traceback.print_exc

    def count_documents(self) -> int:
        try:
            return self.index.ntotal if self.index is not None else 0
        except Exception as ex:
            logs.logger.info(f"Exception in counting document in FAISS: {ex}")
            traceback.print_exc()
            return 0
