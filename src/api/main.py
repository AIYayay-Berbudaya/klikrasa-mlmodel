from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.suggestion import router as suggestion_router
from src.api.routes.search import router as search_router
from src.api.routes.helper import router as helper_router
from src.api.routes.summarizer import router as summarizer_router
from src.api.routes.story import router as story_router

app = FastAPI(
    title="Klikrasa ML API",
    version="1.0.0",
    description="Microservice ML untuk proyek Klikrasa"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTERS
app.include_router(suggestion_router, prefix="/api/suggest", tags=["Suggest"])
app.include_router(search_router,      prefix="/api/search",  tags=["Search"])
app.include_router(helper_router,      prefix="/api/chat",    tags=["AI Helper"])
app.include_router(summarizer_router,  prefix="/api/summarize", tags=["Summarizer"])
app.include_router(story_router,       prefix="/api/story",   tags=["Story"])

@app.get("/health")
def health():
    return {"status": "healthy"}