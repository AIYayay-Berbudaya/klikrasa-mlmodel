import logging
from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)
genai.configure(api_key=GEMINI_API_KEY)

MODEL = GEMINI_MODEL


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def gemini_generate(prompt: str, timeout_sec: int = 15) -> str:
    """
    Robust Gemini wrapper. Retries on failure.
    """
    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt)
        # prefer .text
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        # fallback: inspect choices-like structure
        try:
            # some SDK versions present candidates/content
            cand = getattr(resp, "candidates", None)
            if cand and len(cand) > 0:
                part = cand[0].content[0].text
                return part.strip()
        except Exception:
            pass
        return str(resp).strip()
    except Exception as e:
        logger.exception("Gemini generate failed")
        raise