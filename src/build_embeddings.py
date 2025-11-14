import json
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

cfg = yaml.safe_load(open("config/settings.yaml"))

jsonl_path = cfg["dataset"]["jsonl"]
index_path = cfg["embeddings"]["index_path"]
docs_path = cfg["embeddings"]["docs_path"]

os.makedirs("data", exist_ok=True)

model = SentenceTransformer(cfg["embeddings"]["model_name"])

docs = []
texts = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        title = obj.get("title", "")
        desc = obj.get("description", "")
        hist = obj.get("history", "")
        process = obj.get("making_process", "")
        region = obj.get("region", "")

        full_text = f"{title} | {region} | {desc} | {hist} | {process}"

        texts.append(full_text)
        docs.append(obj)

embs = model.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embs)

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, index_path)

with open(docs_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("✓ Embeddings & FAISS index created!")