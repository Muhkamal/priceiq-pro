"""Deriv candle fetcher — uses the permissive binaryws.com endpoint (no auth needed)."""
import asyncio
import json
import logging
import os
from typing import Optional
import websockets
import pandas as pd

logger = logging.getLogger(__name__)

# The binaryws.com domain accepts unauthenticated public requests for ticks_history.
# The derivws.com domain requires per-user auth and blocks datacenter IPs.
WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

class DerivWebSocket:
    """Persistent WebSocket (no auth needed for public candle data)."""
    
    def __init__(self):
        self.ws: Optional[object] = None
        self.lock = asyncio.Lock()
    
    async def ensure_connected(self):
        async with self.lock:
            if self.ws is None or self.ws.closed:
                try:
                    self.ws = await websockets.connect(
                        WS_URL,
                        ping_interval=30,
                        ping_timeout=10,
                        open_timeout=15,
                        close_timeout=5
                    )
                except Exception as e:
                    if self.ws:
                        await self.ws.close()
                    self.ws = None
                    raise
    
    async def fetch_candles(self, symbol: str, count: int, granularity: int) -> list:
        async with self.lock:
            await self.ensure_connected()
            try:
                await self.ws.send(json.dumps({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "style": "candles",
                    "granularity": granularity,
                }))
                resp = json.loads(await self.ws.recv())
                return resp.get("candles", [])
            except Exception as e:
                if self.ws:
                    await self.ws.close()
                self.ws = None
                raise
    
    async def close(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

_deriv_ws = DerivWebSocket()

async def fetch_deriv_m5(symbol: str, limit: int = 1000, granularity: int = 300) -> Optional[pd.DataFrame]:
    """Fetch CLOSED M5 candles for a Deriv synthetic symbol."""
    try:
        candles = await _deriv_ws.fetch_candles(symbol, limit, granularity)
        if not candles:
            logger.warning(f"Deriv empty candles for {symbol}")
            return None
        
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df = df.set_index("datetime").sort_index()
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)
        
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index + pd.Timedelta(seconds=granularity) <= now]
        return df[["open", "high", "low", "close"]] if not df.empty else None
        
    except Exception as e:
        logger.error(f"Deriv fetch failed {symbol}: {e}")
        return None
