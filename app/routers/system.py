"""
PriceIQ Pro — System Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone
import logging

from app.core.config import settings
from app.services.scheduler import get_scheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status")
async def system_status():
    scheduler = get_scheduler()
    return {
        "status": "running",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler_running": scheduler._running if scheduler else False,
        "auto_execute": settings.AUTO_EXECUTE_TRADES,
        "paper_trading": settings.OANDA_PAPER_TRADING,
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION
    }


@router.get("/version")
async def version():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/scheduler/status")
async def scheduler_status():
    scheduler = get_scheduler()
    if scheduler:
        return scheduler.get_status()
    return {"status": "not_initialized"}
