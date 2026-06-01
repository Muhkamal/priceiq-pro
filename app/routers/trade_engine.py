"""
PriceIQ Pro — Trade Engine Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/trade-engine", tags=["trade-engine"])

# Global trade engine instance
_trade_engine = None

def set_trade_engine(engine):
    global _trade_engine
    _trade_engine = engine


@router.get("/status")
async def engine_status():
    if _trade_engine is None:
        return {"status": "not_initialized"}
    return _trade_engine.get_status()


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
