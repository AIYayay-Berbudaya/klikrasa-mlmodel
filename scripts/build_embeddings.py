import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import KUE_JSONL, EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, FAISS_INDEX_PATH

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def item_to_text(item):
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

def main():
    print("Loading dataset:", KUE_JSONL)
    data = read_jsonl(str(KUE_JSONL))
    corpus = [item_to_text(d) for d in data]

    print("Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding corpus...")
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    # normalize vectors for cosine (inner product)
    try:
        faiss.normalize_L2(embeddings)
    except Exception:
        pass

    os.makedirs(os.path.dirname(str(EMBEDDINGS_PATH)), exist_ok=True)
    np.save(str(EMBEDDINGS_PATH), embeddings)
    print("Saved embeddings:", EMBEDDINGS_PATH)

    # build FAISS
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    os.makedirs(os.path.dirname(str(FAISS_INDEX_PATH)), exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print("Saved FAISS index:", FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()