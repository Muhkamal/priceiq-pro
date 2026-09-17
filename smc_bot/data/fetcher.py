import os
import logging
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

TD_KEY = os.environ.get("TWELVEDATA_API_KEY") or os.environ.get("TWELVE_DATA_API_KEY", "")

TD_SYMBOLS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDJPY": "USD/JPY",
}

async def fetch_m5(pair: str, limit: int = 1000):
    symbol = TD_SYMBOLS.get(pair)
    if not symbol or not TD_KEY:
        logger.error(f"No Twelve Data symbol/key for {pair}")
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "5min", "outputsize": limit,
              "timezone": "UTC", "apikey": TD_KEY}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        values = data.get("values")
        if not values:
            logger.warning(f"Empty response for {pair}: {data.get('message')}")
            return None
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)
        df = df[["open", "high", "low", "close"]]
    df["volume"] = 1000
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index + pd.Timedelta(minutes=5) <= now]
        return df if not df.empty else None
    except Exception as e:
        logger.error(f"Fetch failed {pair}: {e}")
        return None
