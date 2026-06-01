"""
PriceIQ Pro — Auth Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
