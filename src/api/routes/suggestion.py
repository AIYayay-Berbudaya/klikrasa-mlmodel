from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor

from src.models.suggestion import SemanticSearch

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Suggest"])

# executor for blocking tasks
_executor = ThreadPoolExecutor(max_workers=2)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        logger.info("Initializing SemanticSearch engine...")
        _engine = SemanticSearch()
    return _engine


class SuggestionRequest(BaseModel):
    query: str
    top_k: int = 5
    region_filter: Optional[str] = None


class TasteRequest(BaseModel):
    taste: str
    top_k: int = 5


@router.post("/", response_model=List[dict])
async def suggest(req: SuggestionRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong")
    try:
        engine = get_engine()
        # blocking call run in threadpool
        from asyncio import get_running_loop
        loop = get_running_loop()
        results = await loop.run_in_executor(_executor, engine.search, req.query, req.top_k, req.region_filter)
        return results
    except Exception as ex:
        logger.exception("suggest error")
        raise HTTPException(status_code=500, detail=str(ex))


@router.get("/similar/{kue_id}", response_model=List[dict])
async def similar(kue_id: str, top_k: int = 5):
    try:
        engine = get_engine()
        from asyncio import get_running_loop
        loop = get_running_loop()
        results = await loop.run_in_executor(_executor, engine.search_similar_to, kue_id, top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No similar items found")
        return results
    except HTTPException:
        raise
    except Exception as ex:
        logger.exception("similar error")
        raise HTTPException(status_code=500, detail=str(ex))


@router.post("/by-taste", response_model=List[dict])
async def by_taste(req: TasteRequest):
    if not req.taste or not req.taste.strip():
        raise HTTPException(status_code=400, detail="Taste cannot be empty")
    try:
        engine = get_engine()
        from asyncio import get_running_loop
        loop = get_running_loop()
        results = await loop.run_in_executor(_executor, engine.search_by_taste, req.taste, req.top_k)
        return results
    except Exception as ex:
        logger.exception("by_taste error")
        raise HTTPException(status_code=500, detail=str(ex))