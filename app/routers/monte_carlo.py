"""
PriceIQ Pro — monte_carlo Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/monte_carlo", tags=["monte_carlo"])


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
