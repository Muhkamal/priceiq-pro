"""
PriceIQ Pro — Multi-Timeframe Confluence Engine v1.0

Scores directional alignment across H1 + H4 + D1 before any signal fires.
A bearish D1 trend should override a bullish H1 signal — always.

Weighting (higher TF = more authority):
    D1  → 50% weight
    H4  → 35% weight
    H1  → 15% weight

Confluence score:
    1.0 = all three timeframes agree on direction
    0.5 = mixed / uncertain
    0.0 = higher TF directly contradicts signal

Outputs:
    confluence_score  (0–1)
    size_multiplier   (0.0 = block, 0.5 = half, 1.0 = full)
    dominant_bias     ("buy" | "sell" | "neutral")
    should_block      (True if D1 directly opposes H1 signal)

Usage:
    mtf = MultiTimeframeConfluence()
    await mtf.update("XAUUSD", h1_candles, h4_candles, d1_candles)
    result = mtf.score("XAUUSD", proposed_direction="buy")
    if result.should_block:
        return None
    lots *= result.size_multiplier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Timeframe weights: higher TF = more authority
TF_WEIGHTS = {"d1": 0.50, "h4": 0.35, "h1": 0.15}

# Minimum confluence to trade full size
FULL_SIZE_THRESHOLD  = 0.70
HALF_SIZE_THRESHOLD  = 0.50
BLOCK_THRESHOLD      = 0.25   # below this = block entirely

# D1 override: if D1 directly opposes signal → always block
D1_OVERRIDE_ENABLED  = True


def _safe_mean(vals, default=0.0):
    clean = [v for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else default


def _ema(values: List[float], period: int) -> float:
    if not values or period < 1:
        return values[-1] if values else 0.0
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _timeframe_bias(candles: List) -> Tuple[str, float]:
    """
    Determine directional bias for a set of candles.
    Returns (bias, strength) where bias is "buy" | "sell" | "neutral"
    and strength is 0–1.
    """
    if len(candles) < 20:
        return "neutral", 0.0

    closes = [getattr(c, "close", None) for c in candles
               if getattr(c, "close", None) and np.isfinite(getattr(c, "close", 0))]
    if len(closes) < 20:
        return "neutral", 0.0

    sma20 = _safe_mean(closes[-20:])
    sma50 = _safe_mean(closes[-min(50, len(closes)):])
    ema20 = _ema(closes[-50:], 20)
    ema50 = _ema(closes[-50:], 50)

    # Trend direction: both SMA and EMA crossover must agree
    sma_bull = sma20 > sma50 * 1.001
    sma_bear = sma20 < sma50 * 0.999
    ema_bull = ema20 > ema50 * 1.001
    ema_bear = ema20 < ema50 * 0.999

    # Higher high / higher low structure
    highs = [getattr(c, "high", 0) for c in candles[-20:] if getattr(c, "high", None)]
    lows  = [getattr(c, "low",  0) for c in candles[-20:] if getattr(c, "low",  None)]
    hh_hl = (len(highs) >= 4 and highs[-1] > highs[-3]
              and lows[-1]  > lows[-3])
    lh_ll = (len(lows)  >= 4 and highs[-1] < highs[-3]
              and lows[-1]  < lows[-3])

    # Price position vs SMAs
    curr      = closes[-1]
    above_sma = curr > sma20 > sma50
    below_sma = curr < sma20 < sma50

    # Momentum: 10-bar return
    mom = (closes[-1] - closes[-10]) / max(closes[-10], 1e-9) if len(closes) >= 10 else 0.0

    # Score bullish signals
    bull_signals = sum([sma_bull, ema_bull, hh_hl, above_sma, mom > 0.002])
    bear_signals = sum([sma_bear, ema_bear, lh_ll, below_sma, mom < -0.002])
    total        = 5

    if bull_signals >= 3:
        strength = bull_signals / total
        return "buy", round(strength, 3)
    elif bear_signals >= 3:
        strength = bear_signals / total
        return "sell", round(strength, 3)
    else:
        return "neutral", round(max(bull_signals, bear_signals) / total, 3)


@dataclass
class MTFResult:
    pair:             str
    proposed_dir:     str
    confluence_score: float
    dominant_bias:    str
    d1_bias:          str
    h4_bias:          str
    h1_bias:          str
    d1_strength:      float
    h4_strength:      float
    h1_strength:      float
    size_multiplier:  float
    should_block:     bool
    reason:           str


class MultiTimeframeConfluence:
    """
    Maintains bias state per pair across three timeframes.
    Call update() each time new candles are available.
    Call score() before executing any signal.
    """

    def __init__(self):
        # pair → {tf → (bias, strength)}
        self._bias_cache: Dict[str, Dict[str, Tuple[str, float]]] = {}

    def update(
        self,
        pair:      str,
        h1_candles: List,
        h4_candles: List,
        d1_candles: List,
    ):
        """Update bias cache for all three timeframes for a pair."""
        pair = pair.upper()
        self._bias_cache[pair] = {
            "h1": _timeframe_bias(h1_candles),
            "h4": _timeframe_bias(h4_candles),
            "d1": _timeframe_bias(d1_candles),
        }
        h1b, h1s = self._bias_cache[pair]["h1"]
        h4b, h4s = self._bias_cache[pair]["h4"]
        d1b, d1s = self._bias_cache[pair]["d1"]
        logger.debug(
            f"MTF {pair}: H1={h1b}({h1s:.2f}) H4={h4b}({h4s:.2f}) D1={d1b}({d1s:.2f})"
        )

    def update_single(
        self,
        pair:      str,
        timeframe: str,
        candles:   List,
    ):
        """Update one timeframe only (when you fetch TFs on different schedules)."""
        pair = pair.upper()
        tf   = timeframe.lower().replace("h", "h").replace("d", "d")
        # Normalise: "4h" → "h4", "1h" → "h1", "1d" → "d1"
        tf_map = {"1h": "h1", "4h": "h4", "1d": "d1", "h1": "h1", "h4": "h4", "d1": "d1"}
        tf_key = tf_map.get(tf, tf)
        if pair not in self._bias_cache:
            self._bias_cache[pair] = {}
        self._bias_cache[pair][tf_key] = _timeframe_bias(candles)

    def score(self, pair: str, proposed_direction: str) -> MTFResult:
        """
        Score confluence for a proposed signal direction.
        Returns MTFResult with confluence_score, size_multiplier, and block flag.
        """
        pair  = pair.upper()
        cache = self._bias_cache.get(pair, {})
        prop  = proposed_direction.lower()

        h1_bias, h1_str = cache.get("h1", ("neutral", 0.5))
        h4_bias, h4_str = cache.get("h4", ("neutral", 0.5))
        d1_bias, d1_str = cache.get("d1", ("neutral", 0.5))

        # ── D1 override check ─────────────────────────────────
        d1_opposes = (
            D1_OVERRIDE_ENABLED
            and d1_bias not in ("neutral", prop)
            and d1_str > 0.55    # only block if D1 is confident
        )
        if d1_opposes:
            return MTFResult(
                pair=pair, proposed_dir=prop,
                confluence_score=0.0,
                dominant_bias=d1_bias,
                d1_bias=d1_bias, h4_bias=h4_bias, h1_bias=h1_bias,
                d1_strength=d1_str, h4_strength=h4_str, h1_strength=h1_str,
                size_multiplier=0.0, should_block=True,
                reason=(
                    f"D1 OVERRIDE: D1={d1_bias.upper()}({d1_str:.0%}) "
                    f"directly opposes {prop.upper()} signal"
                ),
            )

        # ── Weighted confluence score ─────────────────────────
        def alignment(bias: str, strength: float, prop: str) -> float:
            """How much does this TF agree with proposed direction?"""
            if bias == prop:
                return strength            # positive alignment
            elif bias == "neutral":
                return 0.5 * strength      # neutral = half weight
            else:
                return 0.0                 # opposes = zero contribution

        score = (
            TF_WEIGHTS["d1"] * alignment(d1_bias, d1_str, prop) +
            TF_WEIGHTS["h4"] * alignment(h4_bias, h4_str, prop) +
            TF_WEIGHTS["h1"] * alignment(h1_bias, h1_str, prop)
        )
        score = round(min(1.0, max(0.0, score)), 4)

        # ── Dominant bias (weighted majority) ─────────────────
        buy_score  = (
            TF_WEIGHTS["d1"] * (d1_str if d1_bias == "buy"  else 0) +
            TF_WEIGHTS["h4"] * (h4_str if h4_bias == "buy"  else 0) +
            TF_WEIGHTS["h1"] * (h1_str if h1_bias == "buy"  else 0)
        )
        sell_score = (
            TF_WEIGHTS["d1"] * (d1_str if d1_bias == "sell" else 0) +
            TF_WEIGHTS["h4"] * (h4_str if h4_bias == "sell" else 0) +
            TF_WEIGHTS["h1"] * (h1_str if h1_bias == "sell" else 0)
        )
        if buy_score > sell_score + 0.05:
            dominant = "buy"
        elif sell_score > buy_score + 0.05:
            dominant = "sell"
        else:
            dominant = "neutral"

        # ── Size multiplier ───────────────────────────────────
        if score >= FULL_SIZE_THRESHOLD:
            size_mult = 1.0
            reason    = f"FULL SIZE: MTF confluence={score:.2f} ≥ {FULL_SIZE_THRESHOLD}"
        elif score >= HALF_SIZE_THRESHOLD:
            size_mult = 0.60
            reason    = f"REDUCED SIZE: MTF confluence={score:.2f} (mixed TF alignment)"
        elif score >= BLOCK_THRESHOLD:
            size_mult = 0.30
            reason    = f"MINIMAL SIZE: MTF confluence={score:.2f} (weak alignment)"
        else:
            size_mult = 0.0
            reason    = f"BLOCKED: MTF confluence={score:.2f} < {BLOCK_THRESHOLD}"

        return MTFResult(
            pair=pair, proposed_dir=prop,
            confluence_score=score,
            dominant_bias=dominant,
            d1_bias=d1_bias, h4_bias=h4_bias, h1_bias=h1_bias,
            d1_strength=d1_str, h4_strength=h4_str, h1_strength=h1_str,
            size_multiplier=size_mult,
            should_block=size_mult == 0.0,
            reason=reason,
        )

    def get_all_biases(self) -> Dict[str, Dict]:
        """Full bias state for all pairs — for dashboard."""
        return {
            pair: {
                tf: {"bias": b, "strength": s}
                for tf, (b, s) in tfs.items()
            }
            for pair, tfs in self._bias_cache.items()
        }
