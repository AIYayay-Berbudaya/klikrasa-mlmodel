import google.generativeai as genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)

class AIHelper:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)

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

        resp = self.model.generate_content(prompt)
        return resp.text

    def compare_kue(self, k1, k2):
        prompt = f"Bandingkan singkat antara {k1} dan {k2}. Fokus pada rasa, bahan, dan asal daerah."
        resp = self.model.generate_content(prompt)
        return resp.text