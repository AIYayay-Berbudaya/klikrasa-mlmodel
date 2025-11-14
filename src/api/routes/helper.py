from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from src.models.helper import AIHelper

router = APIRouter()

helper = None


def get_helper():
    global helper
    if helper is None:
        helper = AIHelper()
    return helper


class ChatRequest(BaseModel):
    question: str
    kue_context: Optional[Dict] = None


@router.post("/ask")
async def ask(req: ChatRequest):
    try:
        h = get_helper()
        answer = h.answer_question(req.question, req.kue_context)
        return {"question": req.question, "answer": answer}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


class CompareRequest(BaseModel):
    kue1_name: str
    kue2_name: str


@router.post("/compare")
async def compare(req: CompareRequest):
    try:
        h = get_helper()
        answer = h.compare_kue(req.kue1_name, req.kue2_name)
        return {"comparison": answer}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
