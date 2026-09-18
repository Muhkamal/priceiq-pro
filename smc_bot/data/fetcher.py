"""Twelve Data M5 candle fetcher with timeout retry and diagnostic error logging."""
import asyncio
import logging
import os
from typing import Optional
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("TWELVEDATA_API_KEY")
if not API_KEY:
    raise RuntimeError("TWELVEDATA_API_KEY not set - configure in Render Environment and export locally")

BASE_URL = "https://api.twelvedata.com/time_series"
TD_SYMBOLS = {"EURUSD": "EUR/USD", "XAUUSD": "XAU/USD", "GBPUSD": "GBP/USD"}


async def fetch_m5(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch CLOSED M5 candles from Twelve Data; one retry on timeout only."""
    params = {
        "symbol": TD_SYMBOLS.get(symbol, symbol),
        "interval": "5min",
        "outputsize": limit,
        "timezone": "UTC",
        "apikey": API_KEY,
        "format": "JSON",
    }
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(BASE_URL, params=params)
                r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                logger.error(f"TwelveData API error for {symbol}: {data.get('message', 'unknown')}")
                return None
            values = data.get("values")
            if not values:
                logger.warning(f"TwelveData no values for {symbol}")
                return None
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df = df[["open", "high", "low", "close"]]
            for c in df.columns:
                df[c] = df[c].astype(float)
            df["volume"] = 1000
            now = pd.Timestamp.now(tz="UTC")
            df = df[df.index + pd.Timedelta(minutes=5) <= now]
            return df if not df.empty else None
        except (TimeoutError, asyncio.TimeoutError) as e:
            if attempt == 0:
                logger.warning(f"Fetch timeout for {symbol} (attempt 1/2), retrying in 3s")
                await asyncio.sleep(3)
            else:
                logger.error(f"Fetch failed {symbol}: {type(e).__name__}: {e!r}")
                return None
        except Exception as e:
            logger.error(f"Fetch failed {symbol}: {type(e).__name__}: {e!r}")
            return None
    return None
