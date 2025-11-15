from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor

from src.utils.gemini import gemini_generate
from src.models.suggestion import SemanticSearch

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search"])

_executor = ThreadPoolExecutor(max_workers=2)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = SemanticSearch()
    return _engine


class SearchModernRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_modern(req: SearchModernRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong")
    try:
        # 1. expand via Gemini
        expand_prompt = (
            f'Jelaskan makanan / jajanan modern berikut agar bisa dicocokkan dengan jajanan tradisional Indonesia: "{req.query}". '
            "Jelaskan rasa, tekstur, bahan utama, dan cara penyajian dalam 2-3 kalimat."
        )
        expanded = gemini_generate(expand_prompt)

        # 2. search using engine (run in threadpool)
        engine = get_engine()
        from asyncio import get_running_loop
        loop = get_running_loop()
        results = await loop.run_in_executor(_executor, engine.search, expanded, req.top_k, None)

        # 3. reasoning for top item (optional)
        if results:
            top = results[0]
            reason_prompt = (
                f'Jelaskan mengapa kue tradisional "{top.get("title")}" mirip dengan makanan modern "{req.query}". '
                f'Deskripsi modern: {expanded} ; Kue desc: {top.get("description","")}. '
                "Jawab singkat 2-4 kalimat."
            )
            reasoning = gemini_generate(reason_prompt)
        else:
            reasoning = "Tidak ditemukan kue tradisional yang relevan."

        return {
            "query": req.query,
            "expanded_query": expanded,
            "results": results,
            "reasoning": reasoning
        }

    except Exception as ex:
        logger.exception("search_modern error")
        raise HTTPException(status_code=500, detail=str(ex))