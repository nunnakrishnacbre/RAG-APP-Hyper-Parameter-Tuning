---
title: FlyKite Airlines HR Policy
emoji: ✈️ 🤗
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 3.9
app_file: app.py
app_type: streamlit
pinned: false
license: mit
---

# Flykite Airlines HR Policy RAG Bot
This BOT build by hypertuning the Chunk size, overlap, different types of embbedding and different types of vector databases.

chunk_size_to_test= [100,250,500,700]
chunk_overlap_to_test=[0,50,100,150]
embedding_models_to_test= ["all-MiniLM-L6-v2","all-mpnet-base-v2","paraphrase-MiniLM-L3-v2","multi-qa-mpnet-base-dot-v1"]
vector_db_types_to_test=['chroma', 'faiss','pinecone']

Hyper tuning is enabled by choosing either Random search or Grid search of the hyper parameters

To Run the hypertuning
python main.py --job eval-hypertune

## create .env file with below keys and values
GROQ_API_KEY=MY_GROQ_API_KEY
OPEN_API_KEY=MY_OPEN_API_KEY
HF_TOKEN=MY_HF_TOKEN

DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_GROQ_LLM_MODEL=llama-3.3-70b-versatile
CHROMA_DB_PATH=CHROMA_DB
FAISS_DB_PATH=FAISS_DB
PDF_FILE_NAME=Flykite_Airlines_HR_Policy.pdf
COLLECTION_NAME=flykite
EVAL_DATA_PATH=eval_data.json

PINECONE_API_KEY=MY_PINECONE_API_KEY
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
