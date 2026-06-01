"""
PriceIQ Pro — Database Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/database", tags=["database"])


@router.get("/status")
async def db_status():
    from app.services.database import db
    return {"available": db.is_available, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
