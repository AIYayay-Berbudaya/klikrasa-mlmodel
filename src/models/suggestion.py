import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import KUE_JSONL, EMBEDDINGS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class SemanticSearch:
    def __init__(self):
        logger.info("Initializing SemanticSearch...")
        # load dataset
        if not os.path.exists(KUE_JSONL):
            raise FileNotFoundError(f"Dataset not found: {KUE_JSONL}")
        self.dataset = read_jsonl(str(KUE_JSONL))

        # ensure title field exists
        for i, it in enumerate(self.dataset):
            if "id" not in it:
                it["id"] = str(i)
            if "title" not in it:
                if "name" in it:
                    it["title"] = it["name"]
                elif "nama_kue" in it:
                    it["title"] = it["nama_kue"]
                else:
                    it["title"] = ""

        # load embedding model
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # load embeddings and index
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("Embeddings or FAISS index not found. Run scripts/build_embeddings.py first.")

        logger.info("Loading embeddings from %s", EMBEDDINGS_PATH)
        self.embeddings = np.load(str(EMBEDDINGS_PATH)).astype("float32")
        logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
        # faiss.read_index expects string path
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        # normalize embeddings for cosine similarity using inner product
        try:
            faiss.normalize_L2(self.embeddings)
        except Exception:
            pass

    def _item_to_text(self, item):
        parts = []
        if item.get("title"):
            parts.append(str(item["title"]))
        if item.get("description"):
            parts.append(item["description"])
        if item.get("history"):
            parts.append(item["history"])
        if item.get("making_process"):
            parts.append(item["making_process"])
        if item.get("region"):
            parts.append(item["region"])
        return " . ".join(parts)

    def encode(self, texts):
        emb = self.model.encode(texts, convert_to_numpy=True)
        try:
            faiss.normalize_L2(emb)
        except Exception:
            pass
        return emb.astype("float32")

    def search_embeddings(self, query_emb, top_k=5):
        if query_emb.ndim == 1:
            query_emb = np.expand_dims(query_emb, 0)
        scores, indices = self.index.search(query_emb.astype("float32"), top_k)
        return scores, indices

    def search(self, query: str, top_k=5, region_filter: str = None):
        q_emb = self.encode([query])
        scores, indices = self.search_embeddings(q_emb, top_k)
        results = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0 or i >= len(self.dataset):
                continue
            item = dict(self.dataset[i])
            item["similarity_score"] = float(s)
            if region_filter:
                region = (item.get("region") or "").lower()
                if region_filter.lower() not in region:
                    continue
            results.append(item)
        return results

    def search_similar_to(self, kue_id: str, top_k=5):
        idx = next((i for i, it in enumerate(self.dataset) if str(it.get("id")) == str(kue_id)), None)
        if idx is None:
            return []
        query_emb = np.expand_dims(self.embeddings[idx], 0)
        scores, ids = self.index.search(query_emb, top_k + 1)
        results = []
        for s, i in zip(scores[0], ids[0]):
            if i == idx:
                continue
            item = dict(self.dataset[i])
            item["similarity_score"] = float(s)
            results.append(item)
        return results

    def search_by_taste(self, taste: str, top_k=5):
        return self.search(taste, top_k)