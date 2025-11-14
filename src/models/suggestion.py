import json
import numpy as np
import faiss
from pathlib import Path
from src.utils.preprocessing import read_jsonl, kue_to_corpus
from src.utils.embeddings import EmbeddingBuilder
from src.config import KUE_JSONL, EMBEDDINGS_PATH, FAISS_INDEX_PATH

class SemanticSearch:
    def __init__(self, rebuild_index: bool = False):
        # load data
        self.data = read_jsonl(KUE_JSONL)
        self.ids = [d.get('id') for d in self.data]
        self.corpus = [kue_to_corpus(d) for d in self.data]
        self.embedder = EmbeddingBuilder()

        if rebuild_index or not Path(EMBEDDINGS_PATH).exists() or not Path(FAISS_INDEX_PATH).exists():
            print('Building embeddings and faiss index...')
            self._build_index()
        else:
            self.embeddings = self.embedder.load_embeddings(EMBEDDINGS_PATH)
            self.index = self.embedder.load_faiss_index(FAISS_INDEX_PATH)

    def _build_index(self):
        embs = self.embedder.encode(self.corpus)
        embs = np.array(embs, dtype='float32')
        # normalize for cosine similarity using inner product
        self.embeddings = embs
        self.embedder.save_embeddings(embs)
        self.index = self.embedder.build_faiss_index(embs)

    def search(self, query: str, top_k: int = 5, region_filter: str = None):
        q_emb = self.embedder.encode([query])
        q_emb = np.array(q_emb, dtype='float32')
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = self.data[int(idx)].copy()
            item['similarity_score'] = float(score)
            results.append(item)
        return results

    def get_by_id(self, item_id: str):
        try:
            index = self.ids.index(item_id)
            return self.data[index].copy()
        except ValueError:
            return None