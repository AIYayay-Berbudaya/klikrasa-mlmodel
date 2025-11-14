from fastapi import FastAPI
from src.api.routes import suggestion, helper, summarizer, story

app = FastAPI(
    title="Klikrasa ML API",
    description="Microservice ML untuk proyek Klikrasa",
    version="1.0.0",
)

# Register routes
app.include_router(suggestion.router, prefix="/api/suggest", tags=["Suggestion"])
app.include_router(helper.router, prefix="/api/chat", tags=["AI Helper"])
app.include_router(summarizer.router, prefix="/api/summarize", tags=["Summarizer"])
app.include_router(story.router, prefix="/api/story", tags=["Story"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}