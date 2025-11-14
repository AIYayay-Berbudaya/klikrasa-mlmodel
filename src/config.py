import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv('DATA_PATH', BASE_DIR / 'data'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', BASE_DIR / 'models'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
FAISS_INDEX_PATH = Path(os.getenv('FAISS_INDEX_PATH', MODEL_PATH / 'suggestion' / 'vector_index.faiss'))
EMBEDDINGS_PATH = Path(os.getenv('EMBEDDINGS_PATH', DATA_PATH / 'embeddings' / 'embeddings.npy'))
KUE_JSONL = Path(os.getenv('KUE_JSONL', DATA_PATH / 'raw' / 'kue_tradisional.jsonl'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SUMMARIZER_MODEL_DIR = Path(os.getenv('SUMMARIZER_MODEL_DIR', MODEL_PATH / 'summarizer'))
STORY_MODEL_DIR = Path(os.getenv('STORY_MODEL_DIR', MODEL_PATH / 'story'))

# Ensure directories exist
for p in [DATA_PATH, MODEL_PATH, MODEL_PATH / 'suggestion', DATA_PATH / 'embeddings']:
    p.mkdir(parents=True, exist_ok=True)