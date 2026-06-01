"""
PriceIQ Pro — Signal Router v3.3
"""

from fastapi import APIRouter, Query, BackgroundTasks
from datetime import datetime, timezone
import logging

from app.services.market_analyzer_advanced import MAETradingFormula
from app.services.data_fetcher import data_fetcher
from app.services.session_filter import SessionFilter
from app.services.alerts import AlertManager
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])

mae_engine = MAETradingFormula()
session_filter = SessionFilter()
alert_manager = AlertManager()


@router.get("/")
async def get_signals_root():
    return {"status": "active", "endpoints": ["/latest", "/generate", "/scan/now", "/candles"]}


@router.get("/latest")
async def get_latest_signals(limit: int = 10):
    return {"status": "success", "signals": [], "count": 0}


@router.post("/generate")
async def generate_signal(
    pair: str = "EURUSD",
    timeframe: str = "1h",
    account_balance: float = 10000.0,
    risk_percent: float = 2.0,
):
    now = datetime.now(timezone.utc)
    session_result = session_filter.check(pair, now)
    
    if not session_result.is_allowed:
        return {
            "status": "blocked",
            "blocked_by": "session_filter",
            "reason": session_result.reason,
            "timestamp": now.isoformat()
        }
    
    candles = await data_fetcher.get_candles(pair, timeframe, limit=300)
    signal = mae_engine.generate_signal(candles, pair, timeframe, account_balance, risk_percent)
    
    if not signal:
        return {"status": "no_signal", "message": "No pattern found", "timestamp": now.isoformat()}
    
    return {"status": "success", "signal": signal.dict(), "timestamp": now.isoformat()}


@router.post("/scan/now")
async def scan_now():
    return {"status": "scanning", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/candles")
async def get_candles(pair: str = "EURUSD", timeframe: str = "1h", limit: int = 50):
    candles = await data_fetcher.get_candles(pair, timeframe, limit=limit)
    return {"pair": pair, "timeframe": timeframe, "count": len(candles), "candles": [c.dict() for c in candles]}


@router.get("/health")
async def signal_health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
