import logging
from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)
genai.configure(api_key=GEMINI_API_KEY)

MODEL = GEMINI_MODEL

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def gemini_generate(prompt: str, timeout_sec: int = 15) -> str:
    """
    Robust Gemini wrapper with retry mechanism.
    Supports google-generativeai library (current version).
    """
    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt)
        
        # Primary method: access .text directly
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        
        # Fallback 1: Access via candidates structure
        if hasattr(resp, "candidates") and resp.candidates:
            candidate = resp.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                parts = candidate.content.parts
                if parts and hasattr(parts[0], "text"):
                    return parts[0].text.strip()
        
        # Fallback 2: Convert response to string
        logger.warning(f"Unexpected response structure: {type(resp)}")
        return str(resp).strip()
        
    except Exception as e:
        logger.exception(f"Gemini generate failed for prompt: {prompt[:50]}...")
        raise