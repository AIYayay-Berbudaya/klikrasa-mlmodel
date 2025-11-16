import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models"))

KUE_JSONL = Path(os.getenv("KUE_JSONL", DATA_PATH / "raw" / "kue_tradisional.jsonl"))

EMBEDDINGS_PATH = Path(os.getenv(
    "EMBEDDINGS_PATH",
    DATA_PATH / "embeddings" / "embeddings.npy"
))

FAISS_INDEX_PATH = Path(os.getenv(
    "FAISS_INDEX_PATH",
    MODEL_PATH / "suggestion" / "vector_index.faiss"
))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Optional
SUMMARIZER_MODEL_DIR = Path(os.getenv("SUMMARIZER_MODEL_DIR", MODEL_PATH / "summarizer"))
STORY_MODEL_DIR = Path(os.getenv("STORY_MODEL_DIR", MODEL_PATH / "story"))

# Ensure common dirs exist
for p in [
    DATA_PATH,
    DATA_PATH / "embeddings",
    DATA_PATH / "raw",
    MODEL_PATH,
    MODEL_PATH / "suggestion"
]:
    p.mkdir(parents=True, exist_ok=True)
