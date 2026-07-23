"""
PriceIQ Pro — Broker Webhook Handler v1.0

Receives real-time position close events from your broker
instead of detecting them via polling every hour.

Without this:
    Trade closes at 14:23 → detected at 15:01 (next hourly scan)
    Learning loop update delayed 38 minutes
    Equity curve snapshot delayed 38 minutes
    Telegram "trade closed" alert delayed 38 minutes

With this:
    Trade closes at 14:23 → webhook fires at 14:23:02
    All updates immediate

Supports:
    - OANDA webhooks (order fill events)
    - Deriv WebSocket close events
    - Generic JSON webhook (any broker that supports HTTP callbacks)
    - Manual close endpoint (dashboard "close" button)

Webhook payload (generic format — map your broker's format in adapter):
    {
        "event":       "position_closed",
        "pair":        "XAUUSD",
        "direction":   "buy",
        "entry":       1920.50,
        "exit_price":  1935.00,
        "lots":        0.10,
        "pnl_usd":     73.50,
        "stop_loss":   1910.00,
        "tp1":         1935.00,
        "outcome":     "win",
        "broker_id":   "12345678"
    }

Mount in main.py:
    from app.services.api.broker_webhook_v5 import broker_webhook_router
    app.include_router(broker_webhook_router)

Then configure your broker to POST to:
    https://your-render-url.com/api/broker/close
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

broker_webhook_router = APIRouter(prefix="/api/broker", tags=["Broker Webhooks"])

# Webhook secret for signature verification (set in Render env)
WEBHOOK_SECRET = os.environ.get("BROKER_WEBHOOK_SECRET", "")


# ── Payload models ────────────────────────────────────────────

class PositionCloseEvent(BaseModel):
    event:       str = "position_closed"
    pair:        str
    direction:   str              # "buy" | "sell"
    entry:       float
    exit_price:  float
    lots:        float
    pnl_usd:     float
    stop_loss:   float = 0.0
    tp1:         float = 0.0
    outcome:     Optional[str] = None   # auto-computed if not provided
    broker_id:   Optional[str] = None
    agent:       Optional[str] = None   # if broker sends metadata
    regime:      Optional[str] = "unknown"
    session:     Optional[str] = "unknown"
    confidence:  Optional[float] = 0.0
    r_multiple:  Optional[float] = None  # auto-computed if not provided
    management_events: Optional[List[str]] = Field(default_factory=list)


class ManualCloseEvent(BaseModel):
    pair:        str
    current_price: float
    reason:      str = "Manual close via broker"


# ── Signature verification ────────────────────────────────────

def _verify_signature(body: bytes, signature: Optional[str]) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    if not WEBHOOK_SECRET:
        return True   # no secret configured → allow (dev mode)
    if not signature:
        return False
    expected = hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


# ── Helper ────────────────────────────────────────────────────

def _get_v5():
    try:
        from app.services.v5_orchestrator_final import get_v5
        v5 = get_v5()
        if not v5:
            raise HTTPException(status_code=503, detail="V5 not initialised")
        return v5
    except ImportError:
        raise HTTPException(status_code=503, detail="V5 module not found")


def _compute_outcome(pnl_usd: float, entry: float, stop_loss: float) -> str:
    if pnl_usd > 0:
        return "win"
    elif pnl_usd < 0:
        return "loss"
    return "breakeven"


def _compute_r_multiple(entry: float, exit_price: float, stop_loss: float, direction: str) -> float:
    risk_dist = abs(entry - stop_loss)
    if risk_dist == 0:
        return 0.0
    pnl_pts = (exit_price - entry) if direction == "buy" else (entry - exit_price)
    return round(pnl_pts / risk_dist, 3)


# ── Endpoints ─────────────────────────────────────────────────

@broker_webhook_router.post("/close")
async def position_closed_webhook(
    request:   Request,
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
):
    """
    Generic position close webhook.
    Configure your broker to POST here when a position closes.
    """
    body = await request.body()

    # Verify signature
    if not _verify_signature(body, x_signature):
        logger.warning("Broker webhook: invalid signature rejected")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        import json
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Parse and normalise
    try:
        event = PositionCloseEvent(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Payload validation error: {e}")

    # Auto-compute missing fields
    if event.outcome is None:
        event.outcome = _compute_outcome(event.pnl_usd, event.entry, event.stop_loss)

    if event.r_multiple is None and event.stop_loss > 0:
        event.r_multiple = _compute_r_multiple(
            event.entry, event.exit_price, event.stop_loss, event.direction
        )

    r_mult = event.r_multiple or 0.0

    # Dispatch to V5 system
    v5 = _get_v5()
    await v5.on_trade_closed(
        pair=event.pair,
        agent_name=event.agent or _infer_agent(v5, event.pair),
        direction=event.direction,
        entry=event.entry,
        exit_price=event.exit_price,
        stop=event.stop_loss,
        tp1=event.tp1,
        outcome=event.outcome,
        r_multiple=r_mult,
        pnl_usd=event.pnl_usd,
        regime=event.regime or "unknown",
        confidence=event.confidence or 0.0,
        session=event.session or _current_session(),
        management_events=event.management_events or [],
    )

    # Record equity curve snapshot
    if hasattr(v5, "equity_tracker"):
        v5.equity_tracker.record(
            balance=v5.governor.current_balance,
            event="trade_close",
            note=f"{event.pair} {event.direction} {event.outcome} {r_mult:+.2f}R",
        )

    logger.info(
        f"Broker webhook processed: {event.pair} {event.direction} "
        f"{event.outcome} {r_mult:+.2f}R ${event.pnl_usd:+.2f}"
    )

    return {
        "status":    "processed",
        "pair":      event.pair,
        "outcome":   event.outcome,
        "r_multiple": r_mult,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@broker_webhook_router.post("/close/oanda")
async def oanda_position_closed(request: Request):
    """
    OANDA-specific webhook adapter.
    Transforms OANDA's order fill format to our generic format.
    """
    body = await request.body()
    try:
        import json
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Map OANDA payload structure
    # OANDA sends: {"type": "ORDER_FILL", "tradesClosed": [...], "instrument": "XAU_USD"}
    try:
        instrument = payload.get("instrument", "").replace("_", "")
        trades     = payload.get("tradesClosed", [])

        if not trades:
            return {"status": "ignored", "reason": "no trades closed"}

        for trade in trades:
            units      = float(trade.get("units", 0))
            direction  = "buy" if units > 0 else "sell"
            realised_pl = float(trade.get("realizedPL", 0))
            price       = float(payload.get("price", 0))

            v5 = _get_v5()
            await v5.on_trade_closed(
                pair=instrument,
                agent_name=_infer_agent(v5, instrument),
                direction=direction,
                entry=price,   # approximate; OANDA doesn't always send original entry here
                exit_price=price,
                stop=0.0,
                tp1=0.0,
                outcome="win" if realised_pl > 0 else "loss",
                r_multiple=0.0,   # can't compute without original entry/stop
                pnl_usd=realised_pl,
                regime="unknown",
                session=_current_session(),
            )

        return {"status": "processed", "trades": len(trades)}
    except Exception as e:
        logger.error(f"OANDA webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@broker_webhook_router.post("/close/manual")
async def manual_close(event: ManualCloseEvent):
    """
    Trigger a manual force-close from dashboard or Telegram.
    """
    v5 = _get_v5()
    if not hasattr(v5, "trade_manager"):
        raise HTTPException(status_code=404, detail="TradeManager not available")

    positions = v5.trade_manager.get_open_positions()
    pair      = event.pair.upper()

    if pair not in positions:
        raise HTTPException(status_code=404, detail=f"No open position for {pair}")

    await v5.trade_manager.force_close(pair, event.current_price, event.reason)

    return {
        "status":    "closed",
        "pair":      pair,
        "price":     event.current_price,
        "reason":    event.reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@broker_webhook_router.get("/positions/snapshot")
async def get_position_snapshot():
    """
    Return the last saved position snapshot (from shutdown or last tick).
    Useful for reconciliation after restart.
    """
    from app.services.core.shutdown_handler import ShutdownHandler
    snapshot = ShutdownHandler.load_position_snapshot()
    if not snapshot:
        return {"positions": {}, "note": "No snapshot found"}
    return snapshot


@broker_webhook_router.get("/health")
async def broker_webhook_health():
    """Health check for broker webhook connectivity."""
    return {
        "status":           "ok",
        "signature_check":  bool(WEBHOOK_SECRET),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


# ── Helpers ───────────────────────────────────────────────────

def _infer_agent(v5, pair: str) -> str:
    """Try to infer which agent opened this position from trade manager."""
    try:
        pos = v5.trade_manager._positions.get(pair.upper())
        if pos:
            return pos.agent
    except Exception:
        pass
    return "unknown"


def _current_session() -> str:
    """Get current trading session name."""
    try:
        from app.services.market_analyzer import get_session_quality
        name, _ = get_session_quality()
        return name
    except Exception:
        h = datetime.now(timezone.utc).hour
        if 12 <= h < 16:
            return "london_newyork"
        if 7 <= h < 16:
            return "london"
        return "other"
