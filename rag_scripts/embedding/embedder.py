import os
import logs
import traceback
from typing import List
from sentence_transformers import SentenceTransformer
from traits.trait_types import self

from configuration import Configuration
from rag_scripts.interfaces import IEmbedder

class SentenceTransformerEmbedder(IEmbedder):
    def __init__(self, model_name: str = Configuration.DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(self.model_name)
            logs.logger.info(f'Sentence Transformer loaded {self.model_name}')
        except Exception as ex:
            logs.logger.info(f"Exception in loading Sentence Transformer {self.model_name}")
            traceback.print_exc()
            raise

    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts).tolist()
            return embeddings
        except Exception as ex:
            logs.logger.info(f"Exception in embedding: {ex}")
            traceback.print_exc()
            return [[] for _ in texts]
        
    def embed_query(self,query: str) -> List[float]:
        try:
            logs.logger.info(f"Embedding query: {query}")
            embedding = self.model.encode(query).tolist()
            return embedding
        except Exception as ex:
            logs.logger.info(f"Exception in query embedding: {ex}")
            traceback.print_exc()
            return []

    def rank(selfself,Query: str, documents:List[str]) -> List[float]:
        if not documents:
            return []
        try:
            sentence_paris = [[Query, doc] for doc in documents]
            scores = self.model.predict(sentence_paris)
            return scores.tolist()
        except Exception as ex:
            logs.logger.info(f"Exception in ranking: {ex}")
            traceback.print_exc()
            return []
