"""
PriceIQ Pro — Data Quality Validator v1.0

Gate upstream of everything else. Catches corrupted data before
any analysis runs. Trading on bad data is worse than not trading.

Checks per candle batch:
    1. Staleness     — last bar timestamp > N hours old (feed dead)
    2. Duplicates    — same timestamp appears twice
    3. Sequence gaps — missing bars in otherwise continuous feed
    4. Price outliers — close > Nσ from rolling mean (data error vs real move)
    5. OHLC sanity   — high < low, open < 0, etc.
    6. Zero volume   — all-zero volume when feed should have data
    7. Flat candles   — 20+ consecutive identical closes (frozen feed)

Usage:
    validator = DataQualityValidator()
    report = validator.validate(candles, pair="XAUUSD", timeframe="1h")
    if report.block:
        return None   # do not trade on corrupted data
    if report.warn:
        logger.warning(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────
STALE_HOURS: dict = {"1m": 0.05, "5m": 0.2, "15m": 0.5,
                     "30m": 1.0, "1h": 2.0, "4h": 6.0, "1d": 26.0}
OUTLIER_SIGMA    = 6.0    # closes beyond 6σ → likely data error
FLAT_CANDLES_MAX = 20     # 20 identical closes → frozen feed
GAP_TOLERANCE    = 2      # allow up to 2 missing bars before warning


@dataclass
class DataQualityReport:
    pair:      str
    timeframe: str
    timestamp: str
    issues:    List[str]
    block:     bool      # True = do not trade
    warn:      bool      # True = log warning, trade with caution
    n_candles: int
    oldest_bar: Optional[str]
    newest_bar: Optional[str]

    def summary(self) -> str:
        if not self.issues:
            return f"{self.pair}/{self.timeframe}: data quality OK ({self.n_candles} bars)"
        return (
            f"{self.pair}/{self.timeframe}: "
            f"{'BLOCK' if self.block else 'WARN'} — "
            f"{'; '.join(self.issues)}"
        )


class DataQualityValidator:
    """
    Validates candle data quality before any trading logic runs.
    Fast, synchronous, no side effects.
    """

    def validate(
        self,
        candles: List,
        pair:       str = "",
        timeframe:  str = "1h",
    ) -> DataQualityReport:
        now    = datetime.now(timezone.utc).isoformat()
        issues = []
        block  = False
        warn   = False

        if not candles:
            return DataQualityReport(
                pair=pair, timeframe=timeframe, timestamp=now,
                issues=["Empty candle list"],
                block=True, warn=True, n_candles=0,
                oldest_bar=None, newest_bar=None,
            )

        closes = [getattr(c, "close", None) for c in candles]
        n      = len(candles)

        # Oldest / newest timestamps
        oldest_bar = newest_bar = None
        try:
            ts_attr = "timestamp" if hasattr(candles[0], "timestamp") else "time"
            oldest_bar = str(getattr(candles[0],  ts_attr, ""))
            newest_bar = str(getattr(candles[-1], ts_attr, ""))
        except Exception:
            pass

        # ── 1. OHLC sanity ─────────────────────────────────
        bad_ohlc = 0
        for c in candles[-50:]:
            try:
                if (c.high < c.low or c.open <= 0 or c.close <= 0
                        or c.high < c.open or c.high < c.close
                        or c.low  > c.open or c.low  > c.close):
                    bad_ohlc += 1
            except AttributeError:
                bad_ohlc += 1
        if bad_ohlc > 0:
            issues.append(f"OHLC_INVALID: {bad_ohlc} bars with high<low or negative prices")
            block = True

        # ── 2. Staleness check ──────────────────────────────
        stale_h = STALE_HOURS.get(timeframe.lower(), 2.0)
        try:
            ts_attr = "timestamp" if hasattr(candles[-1], "timestamp") else "time"
            last_ts = getattr(candles[-1], ts_attr, None)
            if last_ts:
                if isinstance(last_ts, str):
                    last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                if isinstance(last_ts, datetime):
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                    if age_h > stale_h:
                        issues.append(
                            f"STALE_FEED: last bar is {age_h:.1f}h old "
                            f"(max {stale_h}h for {timeframe})"
                        )
                        block = True
        except Exception as e:
            logger.debug(f"Staleness check error: {e}")

        # ── 3. Duplicate timestamps ──────────────────────────
        try:
            ts_attr  = "timestamp" if hasattr(candles[0], "timestamp") else "time"
            all_ts   = [str(getattr(c, ts_attr, i)) for i, c in enumerate(candles)]
            seen: Set[str] = set()
            dups = 0
            for ts in all_ts:
                if ts in seen:
                    dups += 1
                seen.add(ts)
            if dups > 0:
                issues.append(f"DUPLICATE_BARS: {dups} duplicate timestamps")
                warn = True
        except Exception:
            pass

        # ── 4. Price outliers ────────────────────────────────
        try:
            valid_closes = [c for c in closes if c and np.isfinite(c) and c > 0]
            if len(valid_closes) >= 20:
                mu  = np.mean(valid_closes[-100:])
                std = np.std(valid_closes[-100:])
                if std > 0:
                    last_close = valid_closes[-1]
                    z = abs(last_close - mu) / std
                    if z > OUTLIER_SIGMA:
                        issues.append(
                            f"PRICE_OUTLIER: close={last_close:.5f} is "
                            f"{z:.1f}σ from mean={mu:.5f}"
                        )
                        warn = True
        except Exception:
            pass

        # ── 5. Flat / frozen feed ────────────────────────────
        try:
            recent_closes = [c for c in closes[-FLAT_CANDLES_MAX:] if c]
            if len(recent_closes) >= FLAT_CANDLES_MAX:
                if len(set(recent_closes)) == 1:
                    issues.append(
                        f"FROZEN_FEED: {FLAT_CANDLES_MAX} consecutive identical closes ({recent_closes[0]})"
                    )
                    block = True
        except Exception:
            pass

        # ── 6. Zero / missing volume ─────────────────────────
        try:
            recent_vols = [getattr(c, "volume", None) for c in candles[-20:]]
            has_vol = [v for v in recent_vols if v and v > 0]
            if len(recent_vols) > 0 and len(has_vol) == 0:
                issues.append("ZERO_VOLUME: no volume data in last 20 bars")
                warn = True
        except Exception:
            pass

        # ── 7. Sequence gap detection ────────────────────────
        try:
            tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                          "1h": 60, "4h": 240, "1d": 1440}.get(timeframe.lower(), 60)
            expected_gap = timedelta(minutes=tf_minutes)
            ts_attr      = "timestamp" if hasattr(candles[0], "timestamp") else "time"
            timestamps   = []
            for c in candles[-50:]:
                ts = getattr(c, ts_attr, None)
                if ts:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if isinstance(ts, datetime):
                        timestamps.append(ts)
            gaps = 0
            for i in range(1, len(timestamps)):
                diff_min = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                expected  = tf_minutes
                if diff_min > expected * (1 + GAP_TOLERANCE):
                    gaps += 1
            if gaps > 2:
                issues.append(f"SEQUENCE_GAPS: {gaps} gaps detected in last 50 bars")
                warn = True
        except Exception:
            pass

        # ── Final verdict ─────────────────────────────────────
        if not block and issues:
            warn = True

        if issues:
            logger.log(
                logging.ERROR if block else logging.WARNING,
                f"DataQuality {pair}/{timeframe}: {'; '.join(issues)}"
            )

        return DataQualityReport(
            pair=pair, timeframe=timeframe, timestamp=now,
            issues=issues, block=block, warn=warn,
            n_candles=n, oldest_bar=oldest_bar, newest_bar=newest_bar,
        )
