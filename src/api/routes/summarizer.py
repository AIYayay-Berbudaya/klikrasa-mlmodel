from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from src.models.summarizer import TextSummarizer

router = APIRouter()
summ = None


def get_sum():
    global summ
    if summ is None:
        summ = TextSummarizer()
    return summ


class SummarizeText(BaseModel):
    text: str
    max_length: int = 150


@router.post("/text")
async def summarize_text(req: SummarizeText):
    try:
        s = get_sum()
        summary = s._extractive_summarize(req.text, req.max_length)
        return {
            "original_length": len(req.text),
            "summary_length": len(summary),
            "summary": summary
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


class SummarizeKue(BaseModel):
    kue_data: Dict


@router.post("/kue")
async def summarize_kue(req: SummarizeKue):
    try:
        s = get_sum()
        return s.summarize_kue(req.kue_data)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.post("/batch")
async def summarize_batch(kue_list: List[Dict]):
    try:
        s = get_sum()
        return s.batch_summarize(kue_list)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
