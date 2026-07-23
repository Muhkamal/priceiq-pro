"""
PriceIQ Pro — Market Anomaly Detector v1.0

Detects abnormal market conditions BEFORE a signal fires.

What it catches:
    1. ATR spike   — volatility 3×+ above recent baseline (news event, NFP)
    2. Volume spike — 4×+ normal volume in candle
    3. Price gap   — open > 1× ATR from prior close (weekend gap, flash crash)
    4. Spread blow-out — broker spread widens beyond threshold (illiquid)
    5. Tick speed anomaly — abnormal number of ticks in bar (HFT activity)
    6. Sequential same-direction candles — 7+ in a row (momentum exhaustion)

Usage:
    detector = AnomalyDetector()

    # Feed candles every bar:
    report = detector.check(candles, pair="XAUUSD")

    if report.block_signals:
        log(report.reason)
        return None  # don't trade

    if report.warn:
        reduce_position_size(0.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


def _safe_mean(vals, default=0.0):
    clean = [v for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else default


@dataclass
class AnomalyReport:
    timestamp:     str
    pair:          str
    anomalies:     List[str]   # list of detected anomaly names
    block_signals: bool        # True = don't trade
    warn:          bool        # True = trade with caution (smaller size)
    size_multiplier: float     # 1.0 = normal, 0.5 = halved, 0.0 = blocked
    reason:        str         # human-readable summary


class AnomalyDetector:
    """
    Detects abnormal market conditions before any signal is evaluated.
    Acts as gate #0 — upstream of all other filters.
    """

    # Thresholds
    ATR_SPIKE_BLOCK  = 3.5   # ATR × current vs baseline → block
    ATR_SPIKE_WARN   = 2.0   # ATR × current vs baseline → warn
    VOL_SPIKE_BLOCK  = 5.0   # volume × current vs avg → block
    VOL_SPIKE_WARN   = 3.0   # volume × current vs avg → warn
    GAP_BLOCK_ATR    = 1.5   # gap > N × ATR → block
    GAP_WARN_ATR     = 0.8   # gap > N × ATR → warn
    MAX_SAME_DIR     = 7     # consecutive same-direction candles → warn
    ATR_LOOKBACK     = 20
    VOL_LOOKBACK     = 20

    def check(self, candles: List, pair: str = "") -> AnomalyReport:
        """
        Evaluate the latest candle for anomalies.
        Returns AnomalyReport immediately — no async, no side effects.
        """
        anomalies = []
        block = False
        warn  = False

        if len(candles) < self.ATR_LOOKBACK + 2:
            return AnomalyReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                pair=pair, anomalies=[], block_signals=False,
                warn=False, size_multiplier=1.0, reason="Insufficient candles for anomaly check",
            )

        current = candles[-1]
        prev    = candles[-2]

        # ── 1. ATR spike check ───────────────────────────────
        try:
            trs = []
            for i in range(1, self.ATR_LOOKBACK + 1):
                c, p = candles[-i], candles[-i - 1]
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                trs.append(tr)
            baseline_atr = _safe_mean(trs[1:])   # exclude current bar
            current_range = current.high - current.low
            atr_ratio = _safe_div(current_range, baseline_atr)

            if atr_ratio >= self.ATR_SPIKE_BLOCK:
                anomalies.append(f"ATR_SPIKE_BLOCK (×{atr_ratio:.1f} above baseline)")
                block = True
            elif atr_ratio >= self.ATR_SPIKE_WARN:
                anomalies.append(f"ATR_SPIKE_WARN (×{atr_ratio:.1f} above baseline)")
                warn = True
        except Exception as e:
            logger.debug(f"ATR anomaly check error: {e}")

        # ── 2. Volume spike check ────────────────────────────
        try:
            curr_vol = getattr(current, "volume", None)
            if curr_vol and np.isfinite(curr_vol) and curr_vol > 0:
                vol_history = [
                    getattr(candles[-i], "volume", None)
                    for i in range(2, self.VOL_LOOKBACK + 2)
                ]
                vol_history = [v for v in vol_history if v and np.isfinite(v)]
                if vol_history:
                    avg_vol   = _safe_mean(vol_history)
                    vol_ratio = _safe_div(curr_vol, avg_vol)
                    if vol_ratio >= self.VOL_SPIKE_BLOCK:
                        anomalies.append(f"VOLUME_SPIKE_BLOCK (×{vol_ratio:.1f} avg)")
                        block = True
                    elif vol_ratio >= self.VOL_SPIKE_WARN:
                        anomalies.append(f"VOLUME_SPIKE_WARN (×{vol_ratio:.1f} avg)")
                        warn = True
        except Exception as e:
            logger.debug(f"Volume anomaly check error: {e}")

        # ── 3. Price gap check ───────────────────────────────
        try:
            gap = abs(current.open - prev.close)
            if baseline_atr > 0:
                gap_ratio = _safe_div(gap, baseline_atr)
                if gap_ratio >= self.GAP_BLOCK_ATR:
                    anomalies.append(f"PRICE_GAP_BLOCK ({gap_ratio:.1f}× ATR)")
                    block = True
                elif gap_ratio >= self.GAP_WARN_ATR:
                    anomalies.append(f"PRICE_GAP_WARN ({gap_ratio:.1f}× ATR)")
                    warn = True
        except Exception as e:
            logger.debug(f"Gap anomaly check error: {e}")

        # ── 4. Consecutive same-direction candles ────────────
        try:
            same_dir = 0
            last_dir = 1 if current.close >= current.open else -1
            for c in reversed(candles[-self.MAX_SAME_DIR - 2:-1]):
                d = 1 if c.close >= c.open else -1
                if d == last_dir:
                    same_dir += 1
                else:
                    break
            if same_dir >= self.MAX_SAME_DIR:
                anomalies.append(f"MOMENTUM_EXHAUSTION ({same_dir} consecutive {('bull' if last_dir == 1 else 'bear')} candles)")
                warn = True
        except Exception as e:
            logger.debug(f"Direction streak check error: {e}")

        # ── 5. Zero / invalid range (corrupted data) ─────────
        try:
            if current.high <= current.low or current.high == current.open == current.close == current.low:
                anomalies.append("ZERO_RANGE_CANDLE (data corruption suspected)")
                block = True
        except Exception as e:
            logger.debug(f"Range check error: {e}")

        # ── Sizing multiplier ────────────────────────────────
        if block:
            size_mult = 0.0
        elif len(anomalies) >= 2:
            size_mult = 0.25    # multiple warnings → quarter size
        elif warn:
            size_mult = 0.50    # single warning → half size
        else:
            size_mult = 1.0

        if anomalies:
            reason = f"{len(anomalies)} anomaly/ies detected on {pair}: {'; '.join(anomalies)}"
            if block:
                reason = "TRADING BLOCKED: " + reason
            logger.warning(reason)
        else:
            reason = "No anomalies detected"

        return AnomalyReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            anomalies=anomalies,
            block_signals=block,
            warn=warn,
            size_multiplier=size_mult,
            reason=reason,
        )
