import os
from google import generativeai as genai
from src.config import GEMINI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class AIHelper:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")

    def answer_question(self, question: str, kue_context: dict = None):
        ctx = ""
        if kue_context:
            ctx = "\n".join([f"{k}: {v}" for k, v in kue_context.items()])

        prompt = f"""
        Kamu adalah asisten jajanan tradisional Indonesia.
        Gunakan konteks berikut jika tersedia:
        {ctx}

        Pertanyaan: {question}
        Jawab secara singkat dan informatif.
        """

        res = genai.generate_text(model="models/gemini-2.5-flash", prompt=prompt, max_output_tokens=256)
        return res.text

    def compare_kue(self, k1, k2):
        prompt = f"Bandingkan singkat antara {k1} dan {k2}. Fokus pada rasa, bahan, dan asal daerah."
        res = genai.generate_text(model="models/gemini-2.5-flash", prompt=prompt, max_output_tokens=256)
        return res.text