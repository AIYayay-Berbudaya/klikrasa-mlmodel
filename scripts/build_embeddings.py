from src.utils.preprocessing import read_jsonl, kue_to_corpus
from src.utils.embeddings import EmbeddingBuilder
from src.config import KUE_JSONL
import numpy as np

if __name__ == '__main__':
    print("Loading dataset...")
    data = read_jsonl(KUE_JSONL)
    corpus = [kue_to_corpus(d) for d in data]

    print("Building embeddings...")
    eb = EmbeddingBuilder()
    embeddings = eb.encode(corpus)

    embeddings = np.array(embeddings, dtype="float32")
    eb.save_embeddings(embeddings)

    print("Building FAISS index...")
    eb.build_faiss_index(embeddings)

    print("DONE: embeddings.npy + vector_index.faiss berhasil dibuat.")