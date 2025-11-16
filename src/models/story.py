from google import generativeai as genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)

class StoryParaphraser:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def paraphrase_story(self, original_story, kue_name, style="gen-z"):
        prompt = f"""
        Parafrase cerita sejarah untuk kue '{kue_name}' menjadi gaya {style}.
        Jangan menambah fakta baru. Buat modern, ringkas, dan menarik.

        Teks asli:
        {original_story}
        """

        resp = self.model.generate_content(prompt)
        return resp.text

    def create_hook(self, kue_name, description):
        prompt = f"""
        Buat 1 kalimat hook singkat dan catchy untuk promosi kue '{kue_name}'.
        Dasarkan kalimat pada deskripsi berikut:
        {description}
        """

        resp = self.model.generate_content(prompt)
        return resp.text