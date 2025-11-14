from src.config import STORY_MODEL_DIR, GEMINI_API_KEY
from google import generativeai as genai
import os

# Setup Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class StoryParaphraser:
    def __init__(self):
        """
        Hybrid paraphraser:
        - Jika ada model lokal → pakai model lokal
        - Jika tidak ada → pakai Gemini Pro
        """
        self.local_available = False

        if STORY_MODEL_DIR and os.path.exists(STORY_MODEL_DIR):
            # (Optional) implementasi local model loading di masa depan
            self.local_available = False

        self.use_gemini = GEMINI_API_KEY is not None

    # ---------------------------------------------------------
    # Paraphrase Story
    # ---------------------------------------------------------
    def paraphrase_story(self, original_story: str, kue_name: str, style: str = "gen-z"):
        """
        Parafrase sejarah menjadi versi modern (Gen Z, millennial, casual).
        """
        if self.use_gemini:
            prompt = f"""
            Parafrase cerita sejarah untuk kue '{kue_name}' menjadi gaya {style}.
            Jangan menambah fakta baru. Buat modern, ringkas, dan menarik.

            Teks asli:
            {original_story}
            """

            res = genai.generate_text(
                model="gemini-pro",
                prompt=prompt,
                max_output_tokens=400
            )

            return res.text

        # fallback
        return original_story[:300] + ("..." if len(original_story) > 300 else "")

    # ---------------------------------------------------------
    # Create Marketing Hook
    # ---------------------------------------------------------
    def create_hook(self, kue_name: str, description: str):
        """
        Buat kalimat pembuka singkat menarik.
        """
        if self.use_gemini:
            prompt = f"""
            Buat 1 kalimat hook singkat dan catchy untuk promosi kue '{kue_name}'.
            Dasarkan kalimat pada deskripsi berikut:
            {description}
            """

            r = genai.generate_text(
                model="models/gemini-pro",
                prompt=prompt,
                max_output_tokens=60
            )
            return r.text

        return f"{kue_name}: {description[:70]}..."
