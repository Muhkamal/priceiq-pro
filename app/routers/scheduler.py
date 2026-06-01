"""
PriceIQ Pro — Scheduler Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/status")
async def scheduler_status():
    from app.services.scheduler import get_scheduler
    scheduler = get_scheduler()
    if scheduler:
        return scheduler.get_status()
    return {"status": "not_initialized"}


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
