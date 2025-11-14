from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.models.suggestion import SemanticSearch

router = APIRouter()

engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SemanticSearch()
    return engine


class SuggestionRequest(BaseModel):
    query: str
    top_k: int = 5
    region_filter: Optional[str] = None


@router.post("/", response_model=List[dict])
async def suggest(req: SuggestionRequest):
    try:
        e = get_engine()
        return e.search(req.query, req.top_k, req.region_filter)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.get("/similar/{kue_id}", response_model=List[dict])
async def similar(kue_id: str, top_k: int = 5):
    try:
        e = get_engine()
        return e.search_similar_to(kue_id, top_k)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.post("/by-taste", response_model=List[dict])
async def by_taste(taste: str, top_k: int = 5):
    try:
        e = get_engine()
        return e.search_by_taste(taste, top_k)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
