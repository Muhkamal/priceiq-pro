"""PriceIQ Pro — /api/v5/ask endpoint for the AI chatbot."""
from fastapi import APIRouter

def make_ai_router(v5):
    r = APIRouter(prefix="/api/v5", tags=["ai"])
    @r.get("/ask")
    async def ask(q: str):
        from app.services.core.ai_chatbot import ask_ai
        return {"answer": await ask_ai(q, v5)}
    return r
