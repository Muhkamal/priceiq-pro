"""
PriceIQ Pro — Twelve Data Client v3.2
Updated with rate limiting, retries, connection pooling, and bug fixes.

Twelve Data free tier: 800 requests/day, 8 requests/minute.
Supports real-time forex, stocks, crypto with volume.

Get a free API key at: https://twelvedata.com
"""

import httpx
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from functools import wraps
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

_TD_BASE = "https://api.twelvedata.com"

_TD_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min", 
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1day"}

_TD_SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD",
    "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "XAUUSD": "XAU/USD": "XAG/USD"}


class TDRateLimitError(Exception):
    pass


def _retry_on_error(max_retries: int = 3, backoff: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = backoff * (2 ** attempt)
                        logger.warning(f"{func.__name__} failed, retrying in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


class TwelveDataClient:
    def __init__(self, api_key: str, max_requests_per_minute: int = 8):
        self.api_key = api_key
        self.base_url = _TD_BASE
        self._semaphore = asyncio.Semaphore(1)
        self._min_interval = 60.0 / max_requests_per_minute
        self._last_request_time = None
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limited_request(self, url: str, params: dict) -> dict:
        async with self._semaphore:
            if self._last_request_time:
                elapsed = (datetime.now(timezone.utc) - self._last_request_time).total_seconds()
                if elapsed < self._min_interval:
                    wait = self._min_interval - elapsed
                    await asyncio.sleep(wait)
            client = await self._get_client()
            response = await client.get(url, params=params)
            self._last_request_time = datetime.now(timezone.utc)
            response.raise_for_status()
            return response.json()

    @_retry_on_error(max_retries=3, backoff=1.0)
    async def get_candles(self, pair: str, timeframe: str, limit: int = 300) -> List[Dict]:
        pair = pair.upper().replace("/", "")
        timeframe = timeframe.lower()
        
        symbol = _TD_SYMBOL_MAP.get(pair)
        if not symbol:
            if len(pair) == 6:
                symbol = f"{pair[:3]}/{pair[3:]}"
            else:
                symbol = pair
        
        interval = _TD_INTERVAL_MAP.get(timeframe, "1h")
        limit = min(limit, 5000)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": limit,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC"}
        
        url = f"{self.base_url}/time_series"
        
        try:
            data = await self._rate_limited_request(url, params)
            if data.get("status") == "error":
                code = data.get("code", 0)
                message = data.get("message", "Unknown error")
                if code in (429, 400) or "rate limit" in message.lower():
                    raise TDRateLimitError(message)
                raise ValueError(f"API error {code}: {message}")
            
            values = data.get("values", [])
            if not values:
                raise ValueError(f"No data for {pair}")
            
            candles = []
            for bar in values:
                try:
                    ts_str = bar.get("datetime", "")
                    timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    o = float(bar.get("open", 0))
                    h = float(bar.get("high", 0))
                    l = float(bar.get("low", 0))
                    c = float(bar.get("close", 0))
                    v = bar.get("volume")
                    volume = int(float(v)) if v and float(v) > 0 else None
                    if not all([o, h, l, c]) or h < l or c <= 0:
                        continue
                    candles.append({
                        "timestamp": timestamp,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": volume})
                except (ValueError, KeyError, TypeError):
                    continue
            
            logger.info(f"[TD] {pair} {timeframe} -> {len(candles)} candles")
            return candles
        except TDRateLimitError:
            raise
        except Exception as e:
            raise RuntimeError(f"Twelve Data fetch failed: {e}")

    @_retry_on_error(max_retries=2, backoff=0.5)
    async def get_live_price(self, pair: str) -> Optional[float]:
        pair = pair.upper().replace("/", "")
        symbol = _TD_SYMBOL_MAP.get(pair, pair)
        url = f"{self.base_url}/price"
        params = {"symbol": symbol, "apikey": self.api_key}
        try:
            data = await self._rate_limited_request(url, params)
            if "price" in data:
                return float(data["price"])
            return None
        except Exception as e:
            logger.warning(f"Live price failed for {pair}: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        try:
            price = await self.get_live_price("EURUSD")
            if price is not None:
                return {"status": "healthy", "last_price_eurusd": price}
            return {"status": "degraded", "message": "No price data"}
        except TDRateLimitError as e:
            return {"status": "rate_limited", "message": str(e)}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}


twelve_data = None


def init_twelve_data(api_key: Optional[str] = None) -> Optional[TwelveDataClient]:
    global twelve_data
    key = api_key or getattr(settings, 'TWELVE_DATA_API_KEY', None)
    if key:
        twelve_data = TwelveDataClient(api_key=key)
        logger.info("Twelve Data client initialized")
    else:
        logger.info("Twelve Data API key not found")
        twelve_data = None
    return twelve_data


async def close_twelve_data():
    global twelve_data
    if twelve_data:
        await twelve_data.close()
        twelve_data = None
        logger.info("Twelve Data client closed")
