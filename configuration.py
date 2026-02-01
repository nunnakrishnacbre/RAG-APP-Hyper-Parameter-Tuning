import os
import logs
from dotenv import load_dotenv

load_dotenv()

class Configuration:
    logs.logger.info("Configuration loaded")
    GROQ_API_KEY = os.getenv('GROQ_API_KEY','default_groq_key')
    OPEN_API_KEY = os.getenv('OPENAI_API_KEY','default_openai_key')
    HF_TOKEN = os.getenv('HF_TOKEN','default_hf_token')

    DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE'))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP'))
    DEFAULT_SENTENCE_TRANSFORMER_MODEL = os.getenv('DEFAULT_SENTENCE_TRANSFORMER_MODEL','all-MiniLM-L6-v2')
    DEFAULT_GROQ_LLM_MODEL = os.getenv('DEFAULT_GROQ_LLM_MODEL','llama-3.3-70b-versatile')
    DEFAULT_RERANKER = os.getenv('DEFAULT_RERANKER')

    PROJECT_ROOT_BASE = os.path.abspath(os.path.dirname(__file__))
    logs.logger.info(f"PROJECT_ROOT {PROJECT_ROOT_BASE}")

    if os.path.basename(PROJECT_ROOT_BASE) == "src":

        PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT_BASE, '..'))
    else:
        PROJECT_ROOT = PROJECT_ROOT_BASE


    DOCUMENTS_DIR = os.path.join(PROJECT_ROOT,'DOCUMENTS')
    logs.logger.info(f"DOCUMENTS_DIR {DOCUMENTS_DIR}")
    DATA_DIR = os.path.join(PROJECT_ROOT,'DATA')


    PDF_FILE_NAME = os.getenv('PDF_FILE_NAME')
    FULL_PDF_PATH = os.path.join(DOCUMENTS_DIR,PDF_FILE_NAME)
    logs.logger.info(f"FULL_PDF_PATH: {FULL_PDF_PATH} ")


    CHROMA_DB_PATH = os.path.join(DATA_DIR,os.getenv('CHROMA_DB_PATH'))
    FAISS_DB_PATH = os.path.join(DATA_DIR,os.getenv('FAISS_DB_PATH'))

    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    EVAL_DATA_PATH = os.path.join(PROJECT_ROOT, os.getenv('EVAL_DATA_PATH'))
    
    PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
    PINECONE_CLOUD=os.getenv('PINECONE_CLOUD','aws')
    PINECONE_REGION=os.getenv('PINECONE_REGION','us-east-1')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR,exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(FAISS_DB_PATH,exist_ok=True)

    if not os.path.exists(FULL_PDF_PATH):
        logs.logger.debug(f"PDF not found in {FULL_PDF_PATH}")
    else:
        logs.logger.debug(f"PDF file found in {FULL_PDF_PATH}")

    if not os.path.exists(EVAL_DATA_PATH):
        logs.logger.debug(f"eval json file not found in {EVAL_DATA_PATH}")
    else:
        logs.logger.debug(f"eval file found in {EVAL_DATA_PATH}")
