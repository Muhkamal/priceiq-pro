"""Twelve Data M5 candle fetcher with dummy volume injection."""
import asyncio
import logging
import os
from typing import Optional
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("TWELVEDATA_API_KEY", "6d56ac51c23b4cd5b11c64d0ca34d4cf")
BASE_URL = "https://api.twelvedata.com/time_series"

async def fetch_m5(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch CLOSED M5 candles from Twelve Data."""
    try:
        # Map symbol to Twelve Data format (EURUSD -> EUR/USD)
        td_symbol = symbol
        if symbol == "EURUSD":
            td_symbol = "EUR/USD"
        elif symbol == "XAUUSD":
            td_symbol = "XAU/USD"
        elif symbol == "GBPUSD":
            td_symbol = "GBP/USD"
        
        params = {
            "symbol": td_symbol,
            "interval": "5min",
            "outputsize": limit,
            "timezone": "UTC",
            "apikey": API_KEY,
            "format": "CSV"
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(BASE_URL, params=params)
            r.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), index_col='datetime', parse_dates=True)
        df = df.sort_index()
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Select and rename columns
        df = df[["open", "high", "low", "close"]]
        for c in df.columns:
            df[c] = df[c].astype(float)
        
        # Inject dummy volume for SMC library compatibility
        df["volume"] = 1000
        
        # Drop the forming (current) candle - keep only closed candles
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index + pd.Timedelta(minutes=5) <= now]
        
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"Fetch failed {symbol}: {e}")
        return None
