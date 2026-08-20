"""
PriceIQ Pro — Market Data Fetcher v1.4

Source priority (first available wins):
  1. Twelve Data    — 800 req/day free, real volume, reliable
  2. Alpha Vantage  — 25 req/day free, tick volume
  3. Yahoo Finance  — unlimited, no volume for forex

Set TWELVE_DATA_API_KEY in .env to enable primary source.
AV and Yahoo are automatic fallbacks — no config needed.

Stale data detection, price sanity checks, and clear logging retained.
"""

import httpx
import asyncio
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

_AV_INTRADAY_INTERVALS = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "60min",
}

_YF_INTERVAL_MAP = {
    "1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"),
    "30m": ("30m", "60d"), "1h": ("1h", "730d"),
    "4h": ("1h", "730d"), "1d": ("1d", "5y"),
}

_YF_SYMBOL_MAP = {
    # Forex pairs
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X", "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X", "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    # Commodities
    "XAUUSD": "GC=F",      # Gold Futures
    "XAGUSD": "SI=F",      # Silver Futures (FIXED)
    # Crypto
    "BTCUSD": "BTC-USD",   # Bitcoin
    "ETHUSD": "ETH-USD",   # Ethereum (FIXED)
    "SOLUSD": "SOL-USD",   # Solana (FIXED)
}

_candle_cache: Dict[str, Tuple[List, datetime]] = {}
_CACHE_TTL_INTRADAY = 55
_CACHE_TTL_DAILY    = 300

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

        # Initialize Twelve Data client if key available
        self.td_client = TwelveDataClient(self.td_key) if self.td_key else None

        if self.td_client:
            logger.info("✅ Data source: Twelve Data (primary) + AV + Yahoo (fallbacks)")
        elif self.av_key:
            logger.info("⚠️  Data source: Alpha Vantage (primary) + Yahoo (fallback) — add TWELVE_DATA_API_KEY for 800 req/day")
        else:
            logger.info("⚠️  Data source: Yahoo Finance only — add API keys for better data")

    async def get_candles(
        self,
        pair:       str,
        timeframe:  str,
        limit:      int = 300,
        outputsize: str = "full",
    ) -> List[Candle]:
        pair      = pair.upper().replace("/", "")
        timeframe = timeframe.lower()
        now       = datetime.now(timezone.utc)
        cache_key = f"{pair}_{timeframe}_{limit}"
        ttl       = _CACHE_TTL_DAILY if timeframe == "1d" else _CACHE_TTL_INTRADAY

        # Check cache
        if cache_key in _candle_cache:
            cached, cached_at = _candle_cache[cache_key]
            if (now - cached_at).total_seconds() < ttl and cached:
                if self._is_data_fresh(cached, timeframe, now):
                    return cached[-limit:]
                else:
                    del _candle_cache[cache_key]

        candles = []
        source  = "none"

        # ── 1. Twelve Data (primary) ──────────────────────────
        if self.td_client:
            try:
                raw = await self.td_client.get_candles(pair, timeframe, limit=limit)
                candles = self._td_to_candles(raw)
                if candles:
                    source = "twelve_data"
            except TDRateLimitError:
                logger.warning(f"[TD] Rate limit for {pair} {timeframe} — trying Alpha Vantage")
            except Exception as e:
                logger.warning(f"[TD] Failed for {pair} {timeframe}: {e} — trying Alpha Vantage")

        # ── 2. Alpha Vantage (secondary) ──────────────────────
        if not candles and self.av_key:
            try:
                candles = await self._fetch_av(pair, timeframe, outputsize)
                if candles:
                    source = "alpha_vantage"
                    logger.info(f"[AV] {pair} {timeframe} → {len(candles)} candles, latest: {candles[-1].close:.5f}")
            except RateLimitError:
                logger.warning(f"[AV] Rate limit for {pair} {timeframe} — trying Yahoo Finance")
            except Exception as e:
                logger.warning(f"[AV] Failed for {pair} {timeframe}: {e} — trying Yahoo Finance")

        # ── 3. Yahoo Finance (tertiary) ───────────────────────
        if not candles:
            try:
                candles = await self._fetch_yf(pair, timeframe)
                if candles:
                    source = "yahoo_finance"
                    logger.info(f"[YF] {pair} {timeframe} → {len(candles)} candles, latest: {candles[-1].close:.5f}")
            except Exception as e:
                logger.error(f"[YF] Also failed for {pair} {timeframe}: {e}")

        if not candles:
            raise RuntimeError(
                f"All data sources failed for {pair} {timeframe}. "
                f"Check internet connection and API keys."
            )

        # Stale data check
        if not self._is_data_fresh(candles, timeframe, now):
            latest  = candles[-1].timestamp
            age_h   = (now - latest).total_seconds() / 3600
            raise RuntimeError(
                f"Data for {pair} {timeframe} from {source} is stale — "
                f"latest candle is {age_h:.1f}h old. "
                f"Check API connection."
            )

        # Price sanity check
        self._validate_price(candles[-1], pair, source)

        _candle_cache[cache_key] = (candles, now)
        return candles[-limit:]

    async def get_multi_timeframe_candles(
        self,
        pair: str, limit_h1: int = 300, limit_h4: int = 200, limit_d1: int = 200,
    ) -> Dict[str, List]:
        h1, h4, d1 = await asyncio.gather(
            self.get_candles(pair, "1h", limit=limit_h1),
            self.get_candles(pair, "4h", limit=limit_h4),
            self.get_candles(pair, "1d", limit=limit_d1),
        )
        return {"h1": h1, "h4": h4, "d1": d1}

    async def get_live_price(self, pair: str) -> Optional[float]:
        """Get current live price — uses Twelve Data /price endpoint if available."""
        if self.td_client:
            return await self.td_client.get_live_price(pair)
        return None

    def _td_to_candles(self, raw: List[Dict]) -> List[Candle]:
        """Convert Twelve Data raw dicts to Candle objects."""
        candles = []
        for bar in raw:
            try:
                vol = bar.get("volume")
                if vol is None:
                    vol = self._estimate_volume(bar["high"], bar["low"])
                candles.append(Candle(
                    timestamp=bar["timestamp"],
                    open=bar["open"], high=bar["high"],
                    low=bar["low"],   close=bar["close"],
                    volume=vol,
                ))
            except (KeyError, TypeError):
                continue
        return candles

    # ──────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────

    def _is_data_fresh(self, candles: List, timeframe: str, now: datetime) -> bool:
        if not candles:
            return False
        latest = candles[-1].timestamp
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age_hours = (now - latest).total_seconds() / 3600
        max_age   = _MAX_AGE_HOURS.get(timeframe, 6)
        weekday   = now.weekday()
        if weekday in (5, 6) or (weekday == 4 and now.hour >= 21):
            max_age *= 72
        return age_hours <= max_age

    def _validate_price(self, candle, pair: str, source: str):
        price = candle.close
        if price <= 0:
            raise RuntimeError(f"Invalid price {price} for {pair} from {source}")
        price_str = f"{price:.5f}"
        trailing  = len(price_str) - len(price_str.rstrip('0'))
        if trailing >= 4 and price < 10:
            logger.warning(
                f"⚠️  Suspicious round price {price:.5f} for {pair} from {source}. "
                f"May be a default value — verify API is returning live data."
            )
        logger.info(f"✅ {pair} latest close = {price:.5f} (source: {source}, ts: {candle.timestamp.strftime('%Y-%m-%d %H:%M')} UTC)")

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
        if "Note" in data or "Information" in data:
            raise RateLimitError("AV rate limit")
        if ts_key not in data:
            raise ValueError(f"AV unexpected response: {list(data.keys())}")
        return self._parse_av_series(data[ts_key])

    async def _fetch_av_daily(self, pair: str, outputsize: str) -> List:
        params = {
            "function": "FX_DAILY", "from_symbol": pair[:3],
            "to_symbol": pair[3:], "outputsize": outputsize, "apikey": self.av_key,
        }
        data   = await self._get(self.av_base, params)
        ts_key = "Time Series FX (Daily)"
        if "Note" in data or "Information" in data:
            raise RateLimitError("AV rate limit")
        if ts_key not in data:
            raise ValueError(f"AV unexpected response: {list(data.keys())}")
        return self._parse_av_series(data[ts_key])

    def _parse_av_series(self, time_series: Dict) -> List:
        candles = []
        for ts_str, values in time_series.items():
            try:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d")
                ts = ts.replace(tzinfo=timezone.utc)
                o = float(values.get("1. open",  0))
                h = float(values.get("2. high",  0))
                l = float(values.get("3. low",   0))
                c = float(values.get("4. close", 0))
                v = values.get("5. volume")
                vol = int(float(v)) if v else self._estimate_volume(h, l)
                if not all([o, h, l, c]) or h < l:
                    continue
                candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=vol))
            except (ValueError, KeyError):
                continue
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ──────────────────────────────────────────────────────────
    # Yahoo Finance
    # ──────────────────────────────────────────────────────────

    async def _fetch_yf(self, pair: str, timeframe: str) -> List:
        # Use symbol map for crypto/silver, fallback to forex format for others
        symbol = _YF_SYMBOL_MAP.get(pair, f"{pair[:3]}{pair[3:]}=X")
        if timeframe == "4h":
            h1 = await self._fetch_yf_raw(symbol, "1h", "730d")
            return self._resample_to_4h(h1)
        interval, period = _YF_INTERVAL_MAP.get(timeframe, ("1d", "5y"))
        return await self._fetch_yf_raw(symbol, interval, period)

    async def _fetch_yf_raw(self, symbol: str, interval: str, period: str) -> List:
        url    = f"{self.yf_base}/{symbol}"
        params = {"interval": interval, "range": period, "includePrePost": "false"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        data = await self._get(url, params, headers=headers)
        try:
            result     = data["chart"]["result"][0]
            quotes     = result["indicators"]["quote"][0]
            timestamps = result["timestamp"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"YF unexpected response: {e}")

        opens   = quotes.get("open",   [])
        highs   = quotes.get("high",   [])
        lows    = quotes.get("low",    [])
        closes  = quotes.get("close",  [])
        volumes = quotes.get("volume", [])

        candles = []
        for i, ts in enumerate(timestamps):
            try:
                o = opens[i]  if i < len(opens)  else None
                h = highs[i]  if i < len(highs)  else None
                l = lows[i]   if i < len(lows)   else None
                c = closes[i] if i < len(closes) else None
                v = volumes[i] if i < len(volumes) else None
                if None in [o, h, l, c] or not all([o, h, l, c]) or h < l or c <= 0:
                    continue
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                volume    = int(v) if v and v > 0 else self._estimate_volume(h, l)
                candles.append(Candle(
                    timestamp=timestamp, open=float(o), high=float(h),
                    low=float(l), close=float(c), volume=volume,
                ))
            except (TypeError, ValueError, IndexError):
                continue

        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ──────────────────────────────────────────────────────────
    # 4H resampling
    # ──────────────────────────────────────────────────────────

    def _resample_to_4h(self, h1_candles: List) -> List:
        if not h1_candles:
            return []
        groups: Dict[datetime, List] = {}
        for c in h1_candles:
            hour_4 = (c.timestamp.hour // 4) * 4
            key    = c.timestamp.replace(hour=hour_4, minute=0, second=0, microsecond=0)
            groups.setdefault(key, []).append(c)
        result = []
        for ts in sorted(groups.keys()):
            bars = groups[ts]
            if len(bars) < 2:
                continue
            result.append(Candle(
                timestamp=ts, open=bars[0].open,
                high=max(b.high for b in bars), low=min(b.low for b in bars),
                close=bars[-1].close, volume=sum(b.volume or 0 for b in bars),
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
                "candle_count":       len(candles),
                "latest_price":       latest.close if latest else None,
                "latest_timestamp":   latest.timestamp.isoformat() if latest else None,
            }
        return status


class RateLimitError(Exception):
    pass


data_fetcher = DataFetcher()
