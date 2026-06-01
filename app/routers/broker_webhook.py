"""
PriceIQ Pro — Broker Webhook Handler v3.2
Simplified for v3.3 compatibility.
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Callable, Dict, Any

import httpx
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator

from app.services.database import db
from app.services.circuit_breaker import CircuitBreaker
from app.services.correlation_filter import CorrelationFilter
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["webhooks"])


class TradeCloseEvent(BaseModel):
    broker: str
    pair: str
    order_id: str
    exit_price: float
    pnl: float
    result: str
    balance: float
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    trade_id: Optional[str] = None


class TradeOpenEvent(BaseModel):
    broker: str
    pair: str
    order_id: str
    entry_price: float
    direction: str
    units: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None


# Initialize filters
circuit_breaker = CircuitBreaker()
correlation_filter = CorrelationFilter()


@router.post("/trade-close")
async def handle_trade_close(event: TradeCloseEvent, background_tasks: BackgroundTasks):
    """Receive trade close notification."""
    background_tasks.add_task(_process_trade_close, event)
    return {"status": "received", "pair": event.pair, "pnl": event.pnl}


@router.post("/trade-open")
async def handle_trade_open(event: TradeOpenEvent, background_tasks: BackgroundTasks):
    """Receive trade open notification."""
    background_tasks.add_task(_process_trade_open, event)
    return {"status": "received", "pair": event.pair}


@router.post("/oanda/transactions")
async def handle_oanda_transaction(request: Request, background_tasks: BackgroundTasks):
    """Receive OANDA transaction webhook."""
    try:
        body = await request.json()
        transaction = body.get("transaction", body)
        tx_type = transaction.get("type", "")
        
        if tx_type == "ORDER_FILL":
            pair = transaction.get("instrument", "").replace("_", "")
            direction = "buy" if float(transaction.get("units", 0)) > 0 else "sell"
            event = TradeOpenEvent(
                broker="oanda",
                pair=pair,
                order_id=str(transaction.get("id", "")),
                entry_price=float(transaction.get("price", 0)),
                direction=direction,
                units=abs(float(transaction.get("units", 0))),
            )
            background_tasks.add_task(_process_trade_open, event)
            return {"status": "trade_opened", "pair": pair}
        
        elif tx_type in ("TRADE_CLOSE", "STOP_LOSS_ORDER_TRIGGERED", "TAKE_PROFIT_ORDER_TRIGGERED"):
            pair = transaction.get("instrument", "").replace("_", "")
            pnl = float(transaction.get("pl", 0) or transaction.get("realizedPL", 0))
            result = "win" if pnl > 0 else "loss"
            event = TradeCloseEvent(
                broker="oanda",
                pair=pair,
                order_id=str(transaction.get("id", "")),
                exit_price=float(transaction.get("price", 0)),
                pnl=pnl,
                result=result,
                balance=float(transaction.get("accountBalance", 0)),
            )
            background_tasks.add_task(_process_trade_close, event)
            return {"status": "trade_closed", "pair": pair, "pnl": pnl}
        
        return {"status": "ignored", "type": tx_type}
        
    except Exception as e:
        logger.error(f"OANDA webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def webhook_health():
    """Webhook health check."""
    return {
        "status": "healthy",
        "version": "3.3",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def _process_trade_close(event: TradeCloseEvent):
    """Process trade close event."""
    try:
        circuit_breaker.record_trade_outcome(event.pnl, event.result, event.balance)
        correlation_filter.close_position(event.pair)
        logger.info(f"Trade closed: {event.pair} {event.result} PnL={event.pnl:.2f}")
    except Exception as e:
        logger.error(f"Trade close error: {e}")


async def _process_trade_open(event: TradeOpenEvent):
    """Process trade open event."""
    try:
        correlation_filter.add_position(
            pair=event.pair,
            direction=event.direction,
            lots=event.units / 100000,
            entry_price=event.entry_price
        )
        logger.info(f"Trade opened: {event.pair} {event.direction} @ {event.entry_price:.5f}")
    except Exception as e:
        logger.error(f"Trade open error: {e}")
