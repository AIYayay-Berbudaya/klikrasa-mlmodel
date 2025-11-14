# KlikRasa Dataset

Folder ini berisi:
- kue_tradisional.csv — dataset asli
- kue_tradisional.jsonl — dataset siap digunakan oleh AI (untuk RAG)

Format JSONL:
Setiap baris adalah objek JSON:
{
  "id": "...",
  "title": "...",
  "region": "...",
  "description": "...",
  "history": "...",
  "making_process": "...",
  "image": "...",
  "source": "..."
}
