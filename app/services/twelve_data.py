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
from typing import List, Dict, Optional, Set, Any
from functools import wraps
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

_TD_BASE = "https://api.twelvedata.com"

_TD_INTERVAL_MAP = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "1d":  "1day",
    "1wk": "1week",
    "1mo": "1month"}

# Extended symbol map — add more as needed
_TD_SYMBOL_MAP = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY", "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD", "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD", "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY",
    "EURCHF": "EUR/CHF", "AUDJPY": "AUD/JPY",
    "CADJPY": "CAD/JPY", "NZDJPY": "NZD/JPY",
    "USDCNH": "USD/CNH", "AUDCAD": "AUD/CAD",
    "GBPAUD": "GBP/AUD", "EURAUD": "EUR/AUD",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD",
    "SOLUSD": "SOL/USD", "ADAUSD": "ADA/USD"}


class TDRateLimitError(Exception):
    """Raised when Twelve Data rate limit is hit."""
    pass


def _retry_on_error(max_retries: int = 3, backoff: float = 1.0):
    """Decorator: retry async function with exponential backoff."""
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
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


class TwelveDataClient:
    """
    Twelve Data REST API client with rate limiting and connection pooling.

    Free tier: 800 requests/day, 8 requests/minute.
    """

    def __init__(self, api_key: str, max_requests_per_minute: int = 8):
        self.api_key = api_key
        self.base_url = _TD_BASE

        # Rate limiting: semaphore controls concurrent requests
        # For 8 req/min, allow 1 concurrent with 7.5s delay between releases
        self._semaphore = asyncio.Semaphore(1)
        self._min_interval = 60.0 / max_requests_per_minute  # 7.5s for 8/min
        self._last_request_time: Optional[datetime] = None

        # Reusable HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

        # Simple in-memory cache for live prices
        self._price_cache: Dict[str, tuple] = {}  # pair -> (price, timestamp)
        self._cache_ttl = timedelta(seconds=30)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def close(self):
        """Close the HTTP client. Call on shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limited_request(self, url: str, params: dict) -> dict:
        """Make a rate-limited request to Twelve Data."""
        async with self._semaphore:
            # Enforce minimum interval between requests
            if self._last_request_time:
                elapsed = (datetime.now(timezone.utc) - self._last_request_time).total_seconds()
                if elapsed < self._min_interval:
                    wait = self._min_interval - elapsed
                    logger.debug(f"Rate limiting: waiting {wait:.2f}s")
                    await asyncio.sleep(wait)

            client = await self._get_client()
            response = await client.get(url, params=params)
            self._last_request_time = datetime.now(timezone.utc)

            response.raise_for_status()
            return response.json()

    @_retry_on_error(max_retries=3, backoff=1.0)
    async def get_candles(
        self,
        pair: str,
        timeframe: str,
        limit: int = 300,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles from Twelve Data.
        Returns list of dicts compatible with Candle schema.
        """
        pair = pair.upper().replace("/", "").replace("-", "")
        timeframe = timeframe.lower()

        symbol = _TD_SYMBOL_MAP.get(pair)
        if not symbol:
            # Dynamic formatting for unknown pairs
            if len(pair) == 6:
                symbol = f"{pair[:3]}/{pair[3:]}"
            else:
                symbol = pair
            logger.warning(f"Unknown pair {pair}, using dynamic symbol: {symbol}")

        interval = _TD_INTERVAL_MAP.get(timeframe, "1h")

        if limit > 5000:
            logger.warning(f"Limit {limit} exceeds API max (5000), clamping to 5000")
            limit = 5000

        params = {
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": limit,
            "apikey":     self.api_key,
            "format":     "JSON",
            "order":      "ASC"}

        url = f"{self.base_url}/time_series"

        try:
            data = await self._rate_limited_request(url, params)

            # Check for API errors
            if data.get("status") == "error":
                code = data.get("code", 0)
                message = data.get("message", "Unknown error")

                if code in (429, 400) or "rate limit" in message.lower():
                    raise TDRateLimitError(f"Twelve Data rate limit: {message}")

                raise ValueError(f"Twelve Data API error {code}: {message}")

            values = data.get("values", [])
            if not values:
                raise ValueError(f"No data returned for {pair} {timeframe}")

            candles, skipped = self._parse_td_series(values)
            if skipped > 0:
                logger.warning(f"[TD] {pair} {timeframe}: skipped {skipped} invalid candles")

            logger.info(
                f"[TD] {pair} {timeframe} → {len(candles)} candles, "
                f"latest: {candles[-1]['close']:.5f} @ {candles[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')} UTC"
            )
            return candles

        except TDRateLimitError:
            raise
        except httpx.TimeoutException:
            raise RuntimeError(f"Twelve Data timed out for {pair} {timeframe}")
        except Exception as e:
            raise RuntimeError(f"Twelve Data fetch failed for {pair} {timeframe}: {e}")

    def _parse_td_series(self, values: List[Dict]) -> tuple:
        """
        Parse Twelve Data time series into standard candle dicts.
        Returns: (candles_list, skipped_count)
        """
        candles = []
        skipped = 0

        for bar in values:
            try:
                ts_str = bar.get("datetime", "")
                timestamp = self._parse_timestamp(ts_str)
                if timestamp is None:
                    skipped += 1
                    continue

                o = float(bar.get("open", 0))
                h = float(bar.get("high", 0))
                l = float(bar.get("low", 0))
                c = float(bar.get("close", 0))
                v = bar.get("volume")
                volume = int(float(v)) if v and float(v) > 0 else None

                # Validation
                if not all([o, h, l, c]) or h < l or c <= 0:
                    skipped += 1
                    continue

                candles.append({
                    "timestamp": timestamp,
                    "open":      o,
                    "high":      h,
                    "low":       l,
                    "close":     c,
                    "volume":    volume})
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue

        return candles, skipped

    @staticmethod
    def _parse_timestamp(ts_str: str) -> Optional[datetime]:
        """Parse Twelve Data timestamp string."""
        if not ts_str:
            return None

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z"]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        logger.warning(f"Could not parse timestamp: {ts_str}")
        return None

    @_retry_on_error(max_retries=2, backoff=0.5)
    async def get_live_price(self, pair: str) -> Optional[float]:
        """
        Get current live price with caching.
        Cache TTL: 30 seconds.
        """
        pair = pair.upper().replace("/", "").replace("-", "")

        # Check cache
        cached = self._price_cache.get(pair)
        if cached:
            price, cached_time = cached
            if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                logger.debug(f"[TD] Cache hit for {pair}: {price}")
                return price

        symbol = _TD_SYMBOL_MAP.get(pair, pair)
        url = f"{self.base_url}/price"
        params = {"symbol": symbol, "apikey": self.api_key}

        try:
            data = await self._rate_limited_request(url, params)

            if "price" in data:
                price = float(data["price"])
                self._price_cache[pair] = (price, datetime.now(timezone.utc))
                return price
            return None
        except Exception as e:
            logger.warning(f"Twelve Data live price failed for {pair}: {e}")
            return None

    @_retry_on_error(max_retries=2, backoff=0.5)
    async def get_quote(self, pair: str) -> Optional[Dict]:
        """
        Get full quote (bid, ask, close, volume) for a pair.
        """
        pair = pair.upper().replace("/", "").replace("-", "")
        symbol = _TD_SYMBOL_MAP.get(pair, pair)
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}

        try:
            data = await self._rate_limited_request(url, params)

            if data.get("status") == "error":
                return None

            return {
                "pair":      pair,
                "bid":       float(data.get("bid", 0) or data.get("close", 0)),
                "ask":       float(data.get("ask", data.get("close", 0))),
                "close":     float(data.get("close", 0)),
                "volume":    data.get("volume"),
                "timestamp": datetime.now(timezone.utc)}
        except Exception as e:
            logger.warning(f"Twelve Data quote failed for {pair}: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Check if Twelve Data API is responding."""
        try:
            price = await self.get_live_price("EURUSD")
            if price is not None:
                return {
                    "status": "healthy",
                    "message": "API responding",
                    "last_price_eurusd": price,
                    "api_key_configured": bool(self.api_key)
                }
            else:
                return {
                    "status": "degraded",
                    "message": "API responded but no price data",
                    "api_key_configured": bool(self.api_key)
                }
        except TDRateLimitError as e:
            return {
                "status": "rate_limited",
                "message": str(e),
                "api_key_configured": bool(self.api_key)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "api_key_configured": bool(self.api_key)
            }

    def get_remaining_requests_estimate(self) -> dict:
        """
        Estimate remaining API quota.
        Note: Twelve Data doesn't provide this in responses, so we track locally.
        """
        return {
            "daily_limit": 800,
            "per_minute_limit": 8,
            "note": "Track usage locally or check Twelve Data dashboard"}


# Module-level singleton for dependency injection
twelve_data: Optional[TwelveDataClient] = None


def init_twelve_data(api_key: Optional[str] = None) -> Optional[TwelveDataClient]:
    """Initialize the global Twelve Data client."""
    global twelve_data
    key = api_key or getattr(settings, 'TWELVE_DATA_API_KEY', None)
    if key:
        twelve_data = TwelveDataClient(api_key=key)
        logger.info("✅ Twelve Data client initialized")
    else:
        logger.info("ℹ️ Twelve Data API key not found — client disabled")
        twelve_data = None
    return twelve_data


async def close_twelve_data():
    """Close the global Twelve Data client. Call on shutdown."""
    global twelve_data
    if twelve_data:
        await twelve_data.close()
        twelve_data = None
        logger.info("Twelve Data client closed")
