"""
PriceIQ Pro — Book + Structure Filters (A/B Test Mode)
1) RAYNER TEO (Ch.5): close > 200 SMA | RSI(10) < 30 | IBS <= 20% | long-only optional
2) YOUR CHARTS:       RBS flip zone (resistance→support) + engulfing confirmation
                      (mirrored for shorts: SBR zone + bearish engulfing)

Env flags (Render), default = LOG-ONLY (nothing is blocked until you flip them):
  MR_BOOK_FILTERS_ENABLED=1 -> enforce book rules
  MR_STRUCTURE_ENABLED=1    -> enforce zone + confirmation rules
  MR_BOOK_LONG_ONLY=1       -> block all MR shorts
"""
from __future__ import annotations
import logging, os
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

ENFORCED           = os.environ.get("MR_BOOK_FILTERS_ENABLED", "0") == "1"
STRUCTURE_ENFORCED = os.environ.get("MR_STRUCTURE_ENABLED", "0") == "1"
LONG_ONLY          = os.environ.get("MR_BOOK_LONG_ONLY", "0") == "1"

SMA_PERIOD, RSI_PERIOD = 200, 10
RSI_LONG, RSI_SHORT    = 30, 70
IBS_LONG, IBS_SHORT    = 0.20, 0.80
ZONE_ATR, ZONE_HOLD_ATR = 0.6, 1.0
LOOKBACK, SWING_K      = 160, 5

def _sma(v, p): return float(np.mean(v[-p:])) if len(v) >= p else None

def _rsi(closes, period=RSI_PERIOD):
    if len(closes) < period + 1: return 50.0
    g, l = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]; g.append(max(d, 0.0)); l.append(max(-d, 0.0))
    ag, al = np.mean(g[:period]), np.mean(l[:period])
    for i in range(period, len(g)):
        ag = (ag * (period-1) + g[i]) / period
        al = (al * (period-1) + l[i]) / period
    return 100.0 if al == 0 else float(100 - 100 / (1 + ag / al))

def _atr(candles, period=14):
    if len(candles) < period + 1: return 0.0
    trs = [max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
           for c, p in zip(candles[-period:], candles[-period-1:-1])]
    return float(np.mean(trs)) if trs else 0.0

def internal_bar_strength(c):
    rng = c.high - c.low
    return 0.5 if rng <= 0 else float((c.close - c.low) / rng)

def _engulfing(candles, direction):
    if len(candles) < 3: return False
    prev, curr = candles[-2], candles[-1]
    pb, cb = abs(prev.close - prev.open), abs(curr.close - curr.open)
    if direction == "buy":
        return (prev.close < prev.open and curr.close > curr.open
                and curr.open <= prev.close and curr.close >= prev.open and cb > pb)
    return (prev.close > prev.open and curr.close < curr.open
            and curr.open >= prev.close and curr.close <= prev.open and cb > pb)

def _flip_zone(candles, atr, direction) -> Tuple[bool, Optional[float]]:
    """RBS (buy): swing-high broken to upside, pullback into zone, zone holds."""
    n = len(candles)
    if n < LOOKBACK + SWING_K or atr <= 0: return False, None
    win, k = candles[-LOOKBACK:], SWING_K
    for i in range(len(win) - k - 1, k - 1, -1):
        if direction == "buy":
            lvl = win[i].high
            if not all(win[j].high <= lvl for j in range(i - k, i + k + 1)): continue
            b = next((m for m in range(i + 1, len(win)) if win[m].close > lvl), None)
            if b is None: continue
            if min(c.low for c in win[b:]) < lvl - ZONE_HOLD_ATR * atr: continue
            if abs(candles[-1].close - lvl) <= ZONE_ATR * atr: return True, round(lvl, 6)
        else:
            lvl = win[i].low
            if not all(win[j].low >= lvl for j in range(i - k, i + k + 1)): continue
            b = next((m for m in range(i + 1, len(win)) if win[m].close < lvl), None)
            if b is None: continue
            if max(c.high for c in win[b:]) > lvl + ZONE_HOLD_ATR * atr: continue
            if abs(candles[-1].close - lvl) <= ZONE_ATR * atr: return True, round(lvl, 6)
    return False, None

def check(candles: List, direction: str) -> Dict:
    out = {"direction": direction, "trend_ok": True, "rsi_ok": True, "ibs_ok": True,
           "long_only_ok": True, "zone_ok": False, "confirm_ok": False, "zone_level": None,
           "passed": True, "book_passed": True, "struct_passed": False, "data_ok": True,
           "sma200": None, "rsi10": None, "ibs": None, "summary": ""}

    if LONG_ONLY and direction == "sell": out["long_only_ok"] = False

    closes = [c.close for c in candles if hasattr(c, "close")]
    sma200, rsi10 = _sma(closes, SMA_PERIOD), _rsi(closes)
    ibs = internal_bar_strength(candles[-1]) if candles else 0.5
    atr = _atr(candles)
    out["sma200"] = round(sma200, 6) if sma200 else None
    out["rsi10"] = round(rsi10, 2)
    out["ibs"] = round(ibs, 3)

    if sma200 is None or atr <= 0:
        out["data_ok"] = False
        out["summary"] = "insufficient data — filters not applied"
        return out

    curr = closes[-1]
    out["trend_ok"] = curr > sma200 if direction == "buy" else curr < sma200
    out["rsi_ok"]   = rsi10 < RSI_LONG if direction == "buy" else rsi10 > RSI_SHORT
    out["ibs_ok"]   = ibs <= IBS_LONG if direction == "buy" else ibs >= IBS_SHORT
    out["book_passed"] = out["trend_ok"] and out["rsi_ok"] and out["ibs_ok"] and out["long_only_ok"]

    zone_ok, lvl = _flip_zone(candles, atr, direction)
    out["zone_ok"], out["zone_level"] = zone_ok, lvl
    out["confirm_ok"] = _engulfing(candles, direction)
    out["struct_passed"] = zone_ok and out["confirm_ok"]

    if ENFORCED and not out["book_passed"]: out["passed"] = False
    if STRUCTURE_ENFORCED and not out["struct_passed"]: out["passed"] = False

    out["summary"] = (
        f"trend={'OK' if out['trend_ok'] else 'X'} rsi={rsi10:.0f}{'OK' if out['rsi_ok'] else 'X'} "
        f"ibs={ibs:.2f}{'OK' if out['ibs_ok'] else 'X'} "
        f"zone={'OK@' + str(lvl) if zone_ok else 'X'} engulf={'OK' if out['confirm_ok'] else 'X'}"
        + ("" if out["long_only_ok"] else " LONG_ONLY_X"))
    return out
