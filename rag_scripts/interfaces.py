from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class IDocumentChunker(ABC):
    @abstractmethod
    def hash_document(self) -> str:
        """This function will generate a unique hash for the document"""
        pass
    
    @abstractmethod
    def chunk_documents(self) -> Tuple[str,List[Dict[str,Any]]]:
        '''
        This function will be chunking the document
        return: Tuple[str, List[Dict[str,Any]]]
        this will return the hashed Document ID
        and the dict which contains the content and metadata
        '''
        pass

    

class IEmbedder(ABC):
    @abstractmethod
    def embed_texts(self,texts: List[str]) -> List[List[float]]:
        '''
        This fucntion generates the embeddings for a list of text strings
        Return: List[List[float]]: A list of embedding vectors
        '''
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        '''
        This function generates a embedding for a single query string
        returns: List[float]: A single embedding vector
        '''
        pass

class IVector(ABC):
    def __init__(self, embedder: IEmbedder, db_path: str, collection_name: str):
        self.embedder = embedder
        self.db_path = db_path
        self.collection_name = collection_name
    
    @abstractmethod
    def add_chunks(self, documents: List[Dict[str,Any]]) -> List[str]:
        '''
        This function add the list of documents chunk to the vector database
        Argument:
         documents: List of chunk dictionaries which contains content and metadata
        Returns:
         List[str]: List of Chunk id 
        '''
        pass

    @abstractmethod
    def get_document_hash_ids(self, document_hash: str) -> List[str]:

        '''
        This function check the document is already exist based ont he hash id
        retunr: List[str]: List of chunk IDs associated with the document hash or empty list if not found
        '''
        pass

    @abstractmethod
    def search(self, query: str, k: int=3) -> List[Dict[str,Any]]:
        '''
        This function searches the vector database for relevant documents based on a query.
        Argument:
            query: user query string
            k: The number of top-k results to retrieve
        returns:
            List[Dict]: A List of retrieved document chunk, including content, metadata and distance

        '''
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        '''
        Deletes a sepcified colelction or index from the vector database.
        useful for clean up during evaluations
        '''
        pass

    @abstractmethod
    def count_documents(self) -> int:
        '''
        Returns the number of documents/chunks in the collection
        '''
        pass


class ILLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_message: str = None) -> str:
        '''
        Generates a response from the LLM based on a prompt

        '''
        pass

class IRAGPipeline(ABC):
    @abstractmethod
    def build_index(self):
        '''
        Builds the Rag Pipeline for documents index
        '''
        pass

    @abstractmethod
    def query(self, user_query: str) -> str:
        '''
        Quering the RAG pipeline

        '''
        pass

    @abstractmethod
    def retrieve_raw_documents(self, user_query: str) -> List[Dict]:
        '''
        Retrives the raw documents without LLM generation
        '''
        pass

