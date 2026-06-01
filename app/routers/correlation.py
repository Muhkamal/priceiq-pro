"""
PriceIQ Pro — Correlation Filter Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/correlation", tags=["correlation"])

# Global correlation filter instance
_correlation_filter = None

def set_correlation_filter(cf):
    global _correlation_filter
    _correlation_filter = cf


@router.get("/status")
async def get_correlation_status():
    if _correlation_filter is None:
        return {"status": "not_initialized"}
    return {
        "open_positions": len(_correlation_filter.get_open_positions()),
        "net_exposure": _correlation_filter.get_net_exposure(),
        "total_risk": _correlation_filter.get_total_risk()
    }


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
