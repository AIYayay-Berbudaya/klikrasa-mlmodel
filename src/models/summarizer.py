from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import SUMMARIZER_MODEL_DIR
import os


class TextSummarizer:
    def __init__(self):
        """
        Summarizer hybrid:
        - Jika kamu punya model fine-tuned (di folder models/summarizer/), maka pakai model lokal.
        - Jika tidak ada model, fallback ke extractive summarizer (rule-based).
        """
        self.local_available = False

        if os.path.exists(SUMMARIZER_MODEL_DIR):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_DIR)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_DIR)
                self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
                self.local_available = True
            except Exception as e:
                print("Gagal load summarizer model lokal:", e)
                self.local_available = False

    def _extractive_summarize(self, text: str, max_length: int = 150) -> str:
        """
        Fallback apabila tidak ada model lokal.
        Potong text dan berhenti di akhir kalimat (.)
        """

        # Jika ada model lokal → pakai itu
        if self.local_available:
            try:
                out = self.pipe(text, max_length=max_length, min_length=20, truncation=True)
                return out[0]["summary_text"]
            except Exception as e:
                print("Local summarizer error:", e)

        # Extractive fallback
        text = text.strip()
        if len(text) <= max_length:
            return text

        cut = text[:max_length]
        last_dot = cut.rfind(".")
        if last_dot != -1 and last_dot > 50:
            return cut[:last_dot + 1]

        return cut + "..."

    def summarize_kue(self, kue: dict) -> dict:
        desc = kue.get("description", "")
        hist = kue.get("history", "")

        return {
            "id": kue.get("id"),
            "title": kue.get("title"),
            "region": kue.get("region"),
            "short_description": self._extractive_summarize(desc, max_length=130),
            "short_history": self._extractive_summarize(hist, max_length=150),
            "image": kue.get("image"),
        }

    def batch_summarize(self, kue_list: list) -> list:
        return [self.summarize_kue(item) for item in kue_list]
