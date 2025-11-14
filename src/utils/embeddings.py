import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from src.config import EMBEDDING_MODEL, EMBEDDINGS_PATH, FAISS_INDEX_PATH

class EmbeddingBuilder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)

    def save_embeddings(self, embeddings, path: Path = EMBEDDINGS_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(self, path: Path = EMBEDDINGS_PATH):
        return np.load(path)

    def build_faiss_index(self, embeddings, dim=None, index_path: Path = FAISS_INDEX_PATH):
        if dim is None:
            dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        return index

    def load_faiss_index(self, index_path: Path = FAISS_INDEX_PATH):
        return faiss.read_index(str(index_path))