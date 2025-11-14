from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.models.story import StoryParaphraser

router = APIRouter()
para = StoryParaphraser()


class ParaphraseRequest(BaseModel):
    original_story: str
    kue_name: str
    style: str = "gen-z"


@router.post("/paraphrase")
async def paraphrase(req: ParaphraseRequest):
    try:
        result = para.paraphrase_story(req.original_story, req.kue_name, req.style)
        return {
            "original": req.original_story,
            "modern_version": result,
            "style": req.style
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


class HookRequest(BaseModel):
    kue_name: str
    description: str


@router.post("/hook")
async def hook(req: HookRequest):
    try:
        result = para.create_hook(req.kue_name, req.description)
        return {"hook": result}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
