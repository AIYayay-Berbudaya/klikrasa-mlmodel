from dotenv import load_dotenv
import os
import json
import yaml
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Load .env (opsional, untuk kebutuhan lain)
load_dotenv()

# Load config
cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

# Load FAISS Vector Index
index = faiss.read_index(cfg["embeddings"]["index_path"])
docs = json.load(open(cfg["embeddings"]["docs_path"], encoding="utf-8"))

# Load Sentence Transformer
embedder = SentenceTransformer(cfg["embeddings"]["model_name"])


# RAG Retrieval
def retrieve(query, k=None):
    """Mengambil dokumen paling relevan dari FAISS."""
    if k is None:
        k = cfg["embeddings"]["top_k"]

    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]


# Ollama LLM
def ask_llama(prompt):

    url = "http://localhost:11434/api/generate"   # Endpoint default Ollama local

    payload = {
        "model": cfg["llm"]["base_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg["llm"]["temperature"],
            "num_predict": cfg["llm"]["max_tokens"]
        }
    }

    response = requests.post(url, json=payload)

    # Jika Ollama lagi gak aktif
    if response.status_code != 200:
        return f"[ERROR] Ollama tidak merespon: {response.status_code}\n{response.text}"

    data = response.json()

    print("DEBUG RESPONSE =", data)   # untuk debugging

    # Format normal:
    # { "model": "...", "created_at": "...", "response": "..." }
    if "response" in data:
        return data["response"]

    return "[ERROR] Tidak ada field 'response' dalam balasan Ollama."


# Testing
query = "Apa makna budaya dari kue Adee?"

context_docs = retrieve(query)
context_text = "\n\n".join([doc["description"] for doc in context_docs])

prompt = f"""
Gunakan konteks berikut untuk menjawab pertanyaan pengguna.

KONTEKS:
{context_text}

PERTANYAAN:
{query}

JAWABAN:
"""

print("\n=== JAWABAN AI ===")
print(ask_llama(prompt))