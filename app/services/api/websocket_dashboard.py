"""
PriceIQ Pro — WebSocket Real-Time Dashboard v1.0

Replaces polling with push updates for the React dashboard.

Without WebSocket:
    Dashboard polls /api/v5/positions every 5s
    = 720 HTTP requests/hour per browser tab
    = stale data between polls

With WebSocket:
    Server pushes updates instantly when:
        - New signal fires
        - Position TP/SL hit
        - Regime changes
        - Risk alert triggers
        - Balance updates
    = 0 polling, real-time data

Message types pushed to client:
    {type: "position_update",  data: {...}}
    {type: "signal_fired",     data: {...}}
    {type: "signal_blocked",   data: {pair, reason}}
    {type: "regime_change",    data: {from, to, pair}}
    {type: "risk_alert",       data: {level, message}}
    {type: "balance_update",   data: {balance, drawdown}}
    {type: "heartbeat",        data: {timestamp}}
    {type: "system_status",    data: {...}}

Mount in main.py:
    from app.services.api.websocket_dashboard import ws_router, ws_broadcaster
    app.include_router(ws_router)
    # Then inject ws_broadcaster into v5 and monitor

Client connection:
    const ws = new WebSocket("wss://your-app.onrender.com/api/v5/ws");
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "position_update") updatePositions(msg.data);
    };
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

ws_router = APIRouter(prefix="/api/v5", tags=["V5 WebSocket"])


class WebSocketBroadcaster:
    """
    Manages all active WebSocket connections and broadcasts messages.
    Thread-safe for asyncio event loop.
    """

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock         = asyncio.Lock()
        self._message_count = 0

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info(f"WS client connected. Total: {len(self._connections)}")

        # Send current state on connect
        try:
            await self._send_to(ws, {"type": "connected",
                                      "data": {"message": "PriceIQ Pro V5 live feed",
                                               "timestamp": datetime.now(timezone.utc).isoformat()}})
        except Exception:
            pass

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)
        logger.info(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message_type: str, data: Any):
        """Broadcast a message to all connected clients."""
        if not self._connections:
            return

        payload = json.dumps({
            "type":      message_type,
            "data":      data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq":       self._message_count,
        })
        self._message_count += 1

        dead = set()
        async with self._lock:
            connections = set(self._connections)

        for ws in connections:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                self._connections -= dead

    # ── Typed broadcast helpers ───────────────────────────────

    async def signal_fired(self, cycle_result):
        await self.broadcast("signal_fired", {
            "pair":        cycle_result.pair,
            "direction":   cycle_result.direction,
            "agent":       cycle_result.agent_used,
            "regime":      cycle_result.regime,
            "confidence":  cycle_result.confidence,
            "fill_price":  cycle_result.fill_price,
            "stop_loss":   cycle_result.stop_loss,
            "tp1":         cycle_result.take_profit_1,
            "tp2":         cycle_result.take_profit_2,
            "lots":        cycle_result.adjusted_lots,
            "ev":          cycle_result.expected_value,
            "session":     cycle_result.session,
        })

    async def signal_blocked(self, pair: str, reason: str, gate: str = ""):
        await self.broadcast("signal_blocked", {
            "pair":   pair,
            "reason": reason,
            "gate":   gate,
        })

    async def position_update(self, positions: Dict):
        await self.broadcast("position_update", positions)

    async def regime_change(self, from_regime: str, to_regime: str, pair: str = ""):
        await self.broadcast("regime_change", {
            "from":  from_regime,
            "to":    to_regime,
            "pair":  pair,
        })

    async def risk_alert(self, level: str, message: str, data: Dict = None):
        await self.broadcast("risk_alert", {
            "level":   level,    # "warn" | "critical" | "emergency"
            "message": message,
            "data":    data or {},
        })

    async def balance_update(self, balance: float, drawdown: float, peak: float):
        await self.broadcast("balance_update", {
            "balance":  round(balance, 2),
            "drawdown": round(drawdown, 4),
            "peak":     round(peak, 2),
        })

    async def heartbeat(self, system_summary: Dict = None):
        await self.broadcast("heartbeat", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary":   system_summary or {},
        })

    async def management_event(self, pair: str, event_type: str, message: str):
        await self.broadcast("management_event", {
            "pair":       pair,
            "event_type": event_type,
            "message":    message,
        })

    @property
    def n_connections(self) -> int:
        return len(self._connections)


# ── Singleton broadcaster ─────────────────────────────────────
ws_broadcaster = WebSocketBroadcaster()


# ── WebSocket endpoint ────────────────────────────────────────

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main real-time WebSocket endpoint.
    Connect from React: new WebSocket("wss://app.onrender.com/api/v5/ws")
    """
    await ws_broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection alive — receive ping/pong or commands
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg  = json.loads(data)

                # Handle client commands
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()}))

                elif msg.get("type") == "subscribe":
                    # Client can request immediate state snapshot
                    try:
                        from app.services.v5_orchestrator_final import get_v5
                        v5     = get_v5()
                        status = v5.get_system_status() if v5 else {}
                        await websocket.send_text(json.dumps({
                            "type": "snapshot", "data": status,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }))
                    except Exception as e:
                        logger.debug(f"WS snapshot error: {e}")

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        await ws_broadcaster.disconnect(websocket)


@ws_router.get("/ws/status")
async def ws_status():
    """Number of active WebSocket connections."""
    return {
        "connections":    ws_broadcaster.n_connections,
        "messages_sent":  ws_broadcaster._message_count,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }
