
import time
import logs
import traceback
from typing import List, Dict,Any
import re
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from configuration import Configuration
from rag_scripts.interfaces import IVector, IEmbedder

class PineconeVectorDB(IVector):
    def __init__(self, embedder:IEmbedder, db_path: str, collection_name: str="flykite"):
        super().__init__(embedder,db_path,collection_name)

        self.api_key = Configuration.PINECONE_API_KEY
        self.cloud = Configuration.PINECONE_CLOUD
        self.region = Configuration.PINECONE_REGION

        if not self.api_key:
            raise ValueError("Pinecone API KEY not provided in configuration")
        
        self.pc = Pinecone(api_key=self.api_key)
        collection_name = collection_name.replace('_','-')
        self.index_name=collection_name

        logs.logger.info(f"Collection name: {collection_name}")

        self.dim=len(self.embedder.embed_texts(["test"])[0])
        self._create_index()
        self.index=self.pc.Index(self.index_name)

        logs.logger.info(f"Pinecone DB Initialized index= {self.index_name}, cloud= {self.cloud}, region = {self.region}")

    def _create_index(self):
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name in existing_indexes:
                index = self.pc.Index((self.index_name))
                index_info = index.describe_index_stats()
                if index_info.dimension!=self.dim:
                    logs.logger.info(f"Pinecone DB Index already exists {self.index_name} "
                          f"with dimension {index_info.dimension}"
                          f"but the expected dimension is {self.dim}"
                          " so deleting it and recreating again"
                          )
                    self.pc.delete_index(self.index_name)

                    for _ in range(30):
                        if self.index_name not in self.pc.list_indexes().names():
                            break
                        else:
                            time.sleep(2)


                elif index_info.metric!='cosine':
                    self.pc.delete_index(self.index_name)

                    for _ in range(30):
                        if self.index_name in self.pc.list_indexes().names():
                            time.sleep(2)
                        else:
                            break
                else:
                    logs.logger.info(f"Pinecone index already exists {self.index_name} ")
                    return

            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(name=self.index_name,
                    dimension=self.dim,
                    metric='cosine',
                    spec=ServerlessSpec(cloud=self.cloud,region=self.region)
                )
                max_attempts = 90
                for attempt in range(max_attempts):
                    try:
                        index = self.pc.Index(self.index_name)
                        index_info = index.describe_index_stats()
                        if index_info.dimension == self.dim:
                            logs.logger.info(f"Pinecone index {self.index_name} already exists ")
                            return
                    except PineconeException:
                        pass
                    time.sleep(2)
                raise TimeoutError(f"Pinecone index {self.index_name} does not exist")

    
        except PineconeException as ex:

            logs.logger.info(f"Exception creating Pinecone index: {ex}")
            traceback.print_exc()
            raise
        except Exception as ex:
            logs.logger.info(f"Exception creating Pinecone index: {ex}")
            traceback.print_exc()
            raise



    def add_chunks(self, documents: List[Dict[str,Any]]) -> List[str]:
        try:
            if not documents:
                return[]
            vectors_info = []
            for doc in documents:
                content = doc['content']
                embedding = self.embedder.embed_texts([content])[0]
                chunk_id = doc['metadata']['chunk_id']
                metadata= doc['metadata'].copy()
                metadata['content']=content

                vectors_info.append({
                    "id":chunk_id, "values":embedding, "metadata":metadata
                })

            logs.logger.info(f"Upserting {len(vectors_info)} to pincone index")
            self.index.upsert(vectors=vectors_info)
            chunk_ids = [vec['id'] for vec in vectors_info]
            logs.logger.info(f"Added {len(chunk_ids)} chunks to pinecone index {self.index_name}")
            logs.logger.info(f"Added chunk ids {chunk_ids[:5]}")
            return chunk_ids

        except PineconeException as ex:
            logs.logger.info(f"Exception in adding chunks to pinecone db {ex}")
            traceback.print_exc()
            return []

    def get_document_hash_ids(self, document_hash: str) ->List[str]:
        try:
            vector_sample = [0.0]*self.dim
            result = self.index.query(
                vector = vector_sample,
                filter = {"document_id": {"$eq": document_hash}},
                top_k = 10000,
                include_metadata=False )
            
            return [match['id'] for match in result['matches']]
        
        except PineconeException as ex:
            logs.logger.info(f"Exception getting document hash IDs from pinecone: {ex}")
            traceback.print_exc()
            return []
    
    def search(self,query: str, k:int=3) -> List[Dict[str,Any]]:
        try:
            logs.logger.info(f"searching pinecode index {self.index_name} for query {query}")
            query_embedding = self.embedder.embed_query(query)
            result = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )

            retrieved_documents = []
            for match in result['matches']:
                md= match['metadata']
                content = md.pop('content','')

                retrieved_documents.append({
                    "content":content,
                    "metadata":md,
                    "distance":match['score']
                })
            
            logs.logger.info(f"Retrieved {len(retrieved_documents)} documents from pinecone for query: {query}")

            return retrieved_documents


        except PineconeException as ex:
            logs.logger.info(f"Exception in Pinecone search: {ex}")
            traceback.print_exc()
            return[]

    def delete_collection(self, collection_name: str):
        try:
            
            self.pc.delete_index(collection_name)
            logs.logger.info(f"Pinecone index {collection_name} deleted")
        except PineconeException as ex:
            logs.logger.info(f"Exception in deleting pinecone index: {collection_name} {ex}")
            traceback.print_exc()
    
    def count_documents(self) -> int:
        try:
            for attempt in range(19):
                status = self.index.describe_index_stats()
                count = status.get('total_vector_count',0)
                if count > 0:
                    return count
                time.sleep(7)
            logs.logger.info(f"No vectors found in index: {self.index_name}")
            return 0
            
        except PineconeException as ex:
            logs.logger.info(f"EXception in getting document counts: {ex}")
            traceback.print_exc()
            return 0
