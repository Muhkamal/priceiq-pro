"""
PriceIQ Pro — Market Data Fetcher v2.0 (Quota Shield)

Fixes the 429 Rate Limit Wall:
  1. UNIFIED CACHE: limit=300 and limit=1 share the same cache. Slices on return.
  2. CIRCUIT BREAKERS: If a provider 429s, it is banned for 1 hour globally.
  3. ASSET ROUTING: Crypto routes to Binance (free, no key, unlimited).
  4. SAFE STALENESS: Never deletes cache on staleness; serves cached data to protect APIs.
"""

import httpx
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from app.models.schemas import Candle
    from app.core.config import settings
    from app.services.twelve_data import TwelveDataClient, TDRateLimitError
    _imports_ok = True
except ImportError as e:
    _imports_ok = False
    _import_error = str(e)
    logger.error(f"data_fetcher import error: {e}")

_AV_BASE = "https://www.alphavantage.co/query"
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_BINANCE_BASE = "https://api.binance.com/api/v3/klines"

_AV_INTRADAY_INTERVALS = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "60min",
}

_YF_INTERVAL_MAP = {
    "1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"),
    "30m": ("30m", "60d"), "1h": ("1h", "60d"),
    "4h": ("1h", "60d"), "1d": ("1d", "5y"),
}

_YF_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X", "USDCAD": "USDCAD=X", "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
}

# ═══ QUOTA SHIELD: Binance Routing (Free, No Key) ═══
_BINANCE_SYMBOL_MAP = {
    "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT"
}
_BINANCE_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"
}

# ═══ QUOTA SHIELD: Global Circuit Breakers ═══
_provider_bans = {"twelve_data": 0, "alpha_vantage": 0, "yahoo_finance": 0, "binance": 0}
_BAN_DURATION = 3600  # 1 Hour ban on 429

_candle_cache: Dict[str, Tuple[List, datetime]] = {}
_CACHE_TTL_INTRADAY = 55 * 60  # 55 minutes in seconds
_CACHE_TTL_DAILY    = 300 * 60

_MAX_AGE_HOURS = {
    "1m": 0.5, "5m": 1, "15m": 2, "30m": 3,
    "1h": 3, "4h": 6, "1d": 30,
}


class DataFetcher:

    def __init__(self):
        if not _imports_ok:
            raise RuntimeError(f"DataFetcher import failed: {_import_error}")

        self.av_key  = getattr(settings, "ALPHA_VANTAGE_API_KEY", "")
        self.td_key  = getattr(settings, "TWELVE_DATA_API_KEY", "")
        self.av_base = _AV_BASE
        self.yf_base = _YF_BASE

        self.td_client = TwelveDataClient(self.td_key) if self.td_key else None
        logger.info("✅ DataFetcher v2.0 (Quota Shield) initialized.")

    def _is_banned(self, provider: str) -> bool:
        return time.time() < _provider_bans.get(provider, 0)

    def _ban(self, provider: str):
        _provider_bans[provider] = time.time() + _BAN_DURATION
        logger.warning(f"🚫 CIRCUIT BREAKER: {provider} banned for 1 hour due to 429 Rate Limit.")

    async def get_candles(
        self, pair: str, timeframe: str, limit: int = 300, outputsize: str = "full",
    ) -> List[Candle]:
        pair      = pair.upper().replace("/", "")
        timeframe = timeframe.lower()
        now       = datetime.now(timezone.utc)
        
        # ═══ QUOTA SHIELD FIX 1: Unified Cache Key (ignores limit) ═══
        cache_key = f"{pair}_{timeframe}"
        ttl       = _CACHE_TTL_DAILY if timeframe == "1d" else _CACHE_TTL_INTRADAY

        # Check cache
        if cache_key in _candle_cache:
            cached, cached_at = _candle_cache[cache_key]
            age_seconds = (now - cached_at).total_seconds()
            if age_seconds < ttl and cached:
                # Serve from cache, slice to requested limit
                return cached[-limit:]

        candles = []
        source  = "none"

        # ── 1. Binance (Crypto Routing - Free/Unlimited) ────────
        if pair in _BINANCE_SYMBOL_MAP and not self._is_banned("binance"):
            try:
                candles = await self._fetch_binance(pair, timeframe)
                if candles: source = "binance"
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429: self._ban("binance")
            except Exception as e:
                logger.warning(f"[Binance] Failed for {pair}: {e}")

        # ── 2. Twelve Data (primary FX/Metals) ──────────────────
        if not candles and self.td_client and not self._is_banned("twelve_data"):
            try:
                raw = await self.td_client.get_candles(pair, timeframe, limit=limit)
                candles = self._td_to_candles(raw)
                if candles: source = "twelve_data"
            except TDRateLimitError:
                self._ban("twelve_data")
            except Exception as e:
                logger.warning(f"[TD] Failed for {pair}: {e}")
                if "429" in str(e): self._ban("twelve_data")

        # ── 3. Alpha Vantage (secondary) ──────────────────────
        if not candles and self.av_key and not self._is_banned("alpha_vantage"):
            try:
                candles = await self._fetch_av(pair, timeframe, outputsize)
                if candles: source = "alpha_vantage"
            except RateLimitError:
                self._ban("alpha_vantage")
            except Exception as e:
                logger.warning(f"[AV] Failed for {pair}: {e}")

        # ── 4. Yahoo Finance (tertiary) ───────────────────────
        if not candles and not self._is_banned("yahoo_finance"):
            try:
                candles = await self._fetch_yf(pair, timeframe)
                if candles: source = "yahoo_finance"
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429: self._ban("yahoo_finance")
                logger.error(f"[YF] HTTP Error for {pair}: {e}")
            except Exception as e:
                logger.error(f"[YF] Failed for {pair}: {e}")

        if not candles:
            # If all APIs are banned/failed, check if we have ANY older cached data to serve as fallback
            if cache_key in _candle_cache:
                logger.warning(f"⚠️ All APIs failed/banned for {pair}. Serving stale cache.")
                return _candle_cache[cache_key][0][-limit:]
            raise RuntimeError(f"All data sources failed/banned for {pair} {timeframe}.")

        # ═══ QUOTA SHIELD FIX 2: Never delete cache on staleness ═══
        # We log it, but we KEEP the data to prevent API hammering loops.
        if not self._is_data_fresh(candles, timeframe, now):
            logger.warning(f"⏳ Data for {pair} from {source} is stale, but using it to protect API quotas.")

        self._validate_price(candles[-1], pair, source)

        # Memory guard
        candles = candles[-max(limit * 4, 1200):]
        _candle_cache[cache_key] = (candles, now)
        return candles[-limit:]

    async def get_multi_timeframe_candles(self, pair: str, limit_h1: int = 300, limit_h4: int = 200, limit_d1: int = 200) -> Dict[str, List]:
        h1, h4, d1 = await asyncio.gather(
            self.get_candles(pair, "1h", limit=limit_h1),
            self.get_candles(pair, "4h", limit=limit_h4),
            self.get_candles(pair, "1d", limit=limit_d1),
        )
        return {"h1": h1, "h4": h4, "d1": d1}

    async def get_live_price(self, pair: str) -> Optional[float]:
        # Use 1m or 1h cache to avoid separate API calls
        try:
            candles = await self.get_candles(pair, "1h", limit=1)
            return candles[-1].close if candles else None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────
    # Binance (Crypto)
    # ──────────────────────────────────────────────────────────
    async def _fetch_binance(self, pair: str, timeframe: str, limit: int = 500) -> List:
        symbol = _BINANCE_SYMBOL_MAP.get(pair)
        interval = _BINANCE_INTERVAL_MAP.get(timeframe, "1h")
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = await self._get(_BINANCE_BASE, params)
        candles = []
        for bar in data:
            try:
                ts = datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc)
                o, h, l, c = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
                v = int(float(bar[5]))
                candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
            except Exception: continue
        return candles

    # ──────────────────────────────────────────────────────────
    # Validation & Helpers
    # ──────────────────────────────────────────────────────────
    def _td_to_candles(self, raw: List[Dict]) -> List[Candle]:
        candles = []
        for bar in raw:
            try:
                vol = bar.get("volume")
                if vol is None: vol = self._estimate_volume(bar["high"], bar["low"])
                candles.append(Candle(
                    timestamp=bar["timestamp"], open=bar["open"], high=bar["high"],
                    low=bar["low"], close=bar["close"], volume=vol,
                ))
            except (KeyError, TypeError): continue
        return candles

    def _is_data_fresh(self, candles: List, timeframe: str, now: datetime) -> bool:
        if not candles: return False
        latest = candles[-1].timestamp
        if latest.tzinfo is None: latest = latest.replace(tzinfo=timezone.utc)
        age_hours = (now - latest).total_seconds() / 3600
        max_age   = _MAX_AGE_HOURS.get(timeframe, 6)
        weekday   = now.weekday()
        if weekday in (5, 6) or (weekday == 4 and now.hour >= 21): max_age *= 72
        return age_hours <= max_age

    def _validate_price(self, candle, pair: str, source: str):
        price = candle.close
        if price <= 0: raise RuntimeError(f"Invalid price {price} for {pair}")
        logger.info(f"✅ {pair} latest close = {price:.5f} (source: {source})")

    # ──────────────────────────────────────────────────────────
    # Alpha Vantage
    # ──────────────────────────────────────────────────────────
    async def _fetch_av(self, pair: str, timeframe: str, outputsize: str) -> List:
        if timeframe == "4h":
            h1 = await self._fetch_av(pair, "1h", outputsize)
            return self._resample_to_4h(h1)
        elif timeframe == "1d":
            return await self._fetch_av_daily(pair, outputsize)
        return await self._fetch_av_intraday(pair, timeframe, outputsize)

    async def _fetch_av_intraday(self, pair: str, timeframe: str, outputsize: str) -> List:
        interval = _AV_INTRADAY_INTERVALS[timeframe]
        params   = {
            "function": "FX_INTRADAY", "from_symbol": pair[:3],
            "to_symbol": pair[3:], "interval": interval,
            "outputsize": outputsize, "apikey": self.av_key,
        }
        data   = await self._get(self.av_base, params)
        ts_key = f"Time Series FX ({interval})"
        if "Note" in data or "Information" in data: raise RateLimitError("AV rate limit")
        if ts_key not in data: raise ValueError(f"AV unexpected response")
        return self._parse_av_series(data[ts_key])

    async def _fetch_av_daily(self, pair: str, outputsize: str) -> List:
        params = {
            "function": "FX_DAILY", "from_symbol": pair[:3],
            "to_symbol": pair[3:], "outputsize": outputsize, "apikey": self.av_key,
        }
        data   = await self._get(self.av_base, params)
        ts_key = "Time Series FX (Daily)"
        if "Note" in data or "Information" in data: raise RateLimitError("AV rate limit")
        if ts_key not in data: raise ValueError(f"AV unexpected response")
        return self._parse_av_series(data[ts_key])

    def _parse_av_series(self, time_series: Dict) -> List:
        candles = []
        for ts_str, values in time_series.items():
            try:
                try: ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except ValueError: ts = datetime.strptime(ts_str, "%Y-%m-%d")
                ts = ts.replace(tzinfo=timezone.utc)
                o, h, l, c = float(values.get("1. open", 0)), float(values.get("2. high", 0)), float(values.get("3. low", 0)), float(values.get("4. close", 0))
                v = values.get("5. volume")
                vol = int(float(v)) if v else self._estimate_volume(h, l)
                if not all([o, h, l, c]) or h < l: continue
                candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=vol))
            except (ValueError, KeyError): continue
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ──────────────────────────────────────────────────────────
    # Yahoo Finance
    # ──────────────────────────────────────────────────────────
    async def _fetch_yf(self, pair: str, timeframe: str) -> List:
        symbol = _YF_SYMBOL_MAP.get(pair, f"{pair[:3]}{pair[3:]}=X")
        if timeframe == "4h":
            h1 = await self._fetch_yf_raw(symbol, "1h", "60d")
            return self._resample_to_4h(h1)
        interval, period = _YF_INTERVAL_MAP.get(timeframe, ("1d", "5y"))
        return await self._fetch_yf_raw(symbol, interval, period)

    async def _fetch_yf_raw(self, symbol: str, interval: str, period: str) -> List:
        url    = f"{self.yf_base}/{symbol}"
        params = {"interval": interval, "range": period, "includePrePost": "false"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "application/json"}
        data = await self._get(url, params, headers=headers)
        try:
            result     = data["chart"]["result"][0]
            quotes     = result["indicators"]["quote"][0]
            timestamps = result["timestamp"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"YF unexpected response: {e}")

        opens, highs, lows, closes, volumes = quotes.get("open", []), quotes.get("high", []), quotes.get("low", []), quotes.get("close", []), quotes.get("volume", [])
        candles = []
        for i, ts in enumerate(timestamps):
            try:
                o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                v = volumes[i] if i < len(volumes) else None
                if None in [o, h, l, c] or h < l or c <= 0: continue
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                volume    = int(v) if v and v > 0 else self._estimate_volume(h, l)
                candles.append(Candle(timestamp=timestamp, open=float(o), high=float(h), low=float(l), close=float(c), volume=volume))
            except (TypeError, ValueError, IndexError): continue
        candles.sort(key=lambda c: c.timestamp)
        return candles

    def _resample_to_4h(self, h1_candles: List) -> List:
        if not h1_candles: return []
        groups: Dict[datetime, List] = {}
        for c in h1_candles:
            hour_4 = (c.timestamp.hour // 4) * 4
            key    = c.timestamp.replace(hour=hour_4, minute=0, second=0, microsecond=0)
            groups.setdefault(key, []).append(c)
        result = []
        for ts in sorted(groups.keys()):
            bars = groups[ts]
            if len(bars) < 2: continue
            result.append(Candle(
                timestamp=ts, open=bars[0].open, high=max(b.high for b in bars),
                low=min(b.low for b in bars), close=bars[-1].close,
                volume=sum(b.volume or 0 for b in bars),
            ))
        return result

    async def _get(self, url: str, params: Dict, headers: Dict = None) -> Dict:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params, headers=headers or {})
            response.raise_for_status()
            return response.json()

    def _estimate_volume(self, high: float, low: float) -> int:
        pip = 0.01 if high > 50 else 0.0001
        return max(1, int((high - low) / pip))

    def get_cache_status(self) -> Dict:
        now    = datetime.now(timezone.utc)
        status = {}
        for key, (candles, cached_at) in _candle_cache.items():
            latest = candles[-1] if candles else None
            status[key] = {
                "cached_seconds_ago": round((now - cached_at).total_seconds()),
                "candle_count": len(candles),
                "latest_price": latest.close if latest else None,
            }
        status["banned_providers"] = {k: max(0, round(v - time.time())) for k, v in _provider_bans.items() if v > time.time()}
        return status

class RateLimitError(Exception):
    pass

data_fetcher = DataFetcher()
