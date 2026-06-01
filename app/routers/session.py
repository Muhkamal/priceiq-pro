"""
PriceIQ Pro — Session Router v3.3
"""

from fastapi import APIRouter, Query
from datetime import datetime, timezone
import logging

from app.services.session_filter import SessionFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])

session_filter = SessionFilter()


@router.get("/status")
async def get_session_status(
    pair: str = Query("EURUSD", description="Currency pair"),
    strict: bool = Query(False, description="Strict mode"),
):
    """Get current session status for a pair."""
    now = datetime.now(timezone.utc)
    result = session_filter.check(pair, now, strict=strict)
    
    return {
        "pair": pair,
        "current_time_utc": now.isoformat(),
        "is_allowed": result.is_allowed,
        "current_session": result.current_session,
        "active_sessions": result.active_sessions,
        "is_overlap": result.is_overlap,
        "session_quality": result.session_quality,
        "reason": result.reason,
        "next_open": result.next_open.isoformat() if result.next_open else None,
        "strict_mode": strict,
    }


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
