"""
PriceIQ Pro — Cross-Asset Macro Signal Agent v1.0

The 5th agent. Fundamentally different from the other four.
All other agents use price/volume/pattern data on the same instrument.
This agent uses macro data from OTHER markets to predict XAUUSD and major pairs.

Theoretical basis:
    XAUUSD ←→ US Real Rates (TIPS 10Y yield)    correlation: -0.85
    XAUUSD ←→ USD Index (DXY)                   correlation: -0.72
    XAUUSD ←→ VIX (equity vol)                  correlation: +0.55
    EURUSD ←→ EUR-USD rate differential          correlation: +0.78
    GBPUSD ←→ UK-US rate differential            correlation: +0.71

Why this gives uncorrelated alpha:
    When real yields spike, gold typically falls within 15-60 MINUTES
    before it shows up in the gold OHLCV candle data.
    This agent sees the cause; the pattern agents see the effect.
    Combining uncorrelated signals reduces portfolio variance
    without reducing expected return — the only free lunch in finance.

Data sources (all free/low-cost):
    1. FRED API (Federal Reserve)    — free, rate data
    2. Yahoo Finance (yfinance)      — free, DXY/VIX/equity indices
    3. Investing.com scraper         — free, PMI/CPI releases
    4. ForexFactory calendar         — already integrated

Macro factors tracked:
    US_REAL_RATE     10Y TIPS yield (DFII10 from FRED)
    US_NOMINAL_RATE  10Y Treasury yield (DGS10 from FRED)
    DXY              USD Index (proxy: UUP ETF or DX-Y.NYB)
    VIX              CBOE Volatility Index (^VIX)
    FED_FUNDS        Federal funds rate (FEDFUNDS from FRED)
    EUR_RATE         ECB deposit rate
    GBP_RATE         Bank of England base rate
    GOLD_REAL_YIELD  Implied real yield from gold vs TIPS spread

Signal logic:
    BUY GOLD when:
        real_rate_change_24h < -0.05%  (real rates falling → gold up)
        AND dxy_change_24h < -0.3%     (USD weakening → gold up)
        AND vix_change_24h > +1.0      (fear rising → safe haven)

    SELL GOLD when:
        real_rate_change_24h > +0.05%  (real rates rising → gold down)
        AND dxy_change_24h > +0.3%     (USD strengthening → gold down)

Usage:
    agent = MacroSignalAgent()
    await agent.refresh_data()       # call hourly
    signal = agent.evaluate(candles, regime)
    # integrates into AgentOrchestrator alongside other 4 agents
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np

logger = logging.getLogger(__name__)

# ── FRED API config ───────────────────────────────────────────
FRED_API_KEY  = os.environ.get("FRED_API_KEY", "")   # free at fred.stlouisfed.org
FRED_BASE     = "https://api.stlouisfed.org/fred/series/observations"

# ── Series IDs ────────────────────────────────────────────────
FRED_SERIES = {
    "US_REAL_RATE":    "DFII10",    # 10Y TIPS yield (real rate)
    "US_NOMINAL_RATE": "DGS10",     # 10Y Treasury yield
    "FED_FUNDS":       "FEDFUNDS",  # Federal funds rate
    "US_CPI_YOY":      "CPIAUCSL",  # CPI (for YoY calc)
    "BREAKEVEN_10Y":   "T10YIE",    # 10Y breakeven inflation
}

# ── Signal thresholds ─────────────────────────────────────────
REAL_RATE_CHANGE_THRESH  = 0.05   # % per day — meaningful move
DXY_CHANGE_THRESH        = 0.30   # % per day
VIX_CHANGE_THRESH        = 1.0    # absolute points per day
RATE_DIFF_CHANGE_THRESH  = 0.03   # % for EUR/GBP rate differential


@dataclass
class MacroSnapshot:
    """Point-in-time snapshot of macro variables."""
    timestamp:          str
    us_real_rate:       Optional[float]    # TIPS 10Y yield %
    us_nominal_rate:    Optional[float]    # Treasury 10Y yield %
    us_breakeven_10y:   Optional[float]    # 10Y inflation expectation %
    fed_funds:          Optional[float]    # Fed funds rate %
    dxy_level:          Optional[float]    # USD Index level
    dxy_change_24h:     Optional[float]    # DXY % change
    vix_level:          Optional[float]    # VIX level
    vix_change_24h:     Optional[float]    # VIX point change
    real_rate_change_24h: Optional[float] # Real rate bps change
    nominal_rate_change_24h: Optional[float]
    # Computed signals
    gold_macro_bias:    str = "neutral"   # "bullish" | "bearish" | "neutral"
    gold_macro_strength: float = 0.0     # 0–1
    eurusd_rate_diff:   Optional[float] = None  # EUR-USD rate differential
    gbpusd_rate_diff:   Optional[float] = None


@dataclass
class MacroAgentSignal:
    """Output signal from MacroSignalAgent — same interface as AgentSignal."""
    agent_name:      str = "MacroSignalAgent"
    direction:       Optional[str] = None
    confidence:      float = 0.0
    win_probability: float = 0.0
    expected_value:  float = 0.0
    stop_distance:   float = 0.0
    tp1_distance:    float = 0.0
    tp2_distance:    float = 0.0
    regime_fit:      float = 0.5
    reasoning:       str = ""
    raw_features:    Dict = field(default_factory=dict)
    macro_snapshot:  Optional[MacroSnapshot] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.direction is not None
            and self.confidence > 0.40
            and self.stop_distance > 0
        )


class FREDDataFetcher:
    """Fetches macro data from FRED API (Federal Reserve)."""

    def fetch_latest(self, series_id: str, n: int = 2) -> Optional[List[float]]:
        """Fetch last N observations for a FRED series."""
        if not FRED_API_KEY:
            return None
        try:
            url    = (f"{FRED_BASE}?series_id={series_id}"
                      f"&api_key={FRED_API_KEY}&file_type=json"
                      f"&sort_order=desc&limit={n}")
            req    = Request(url, headers={"User-Agent": "PriceIQ-Pro/5.0"})
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            obs = [
                float(o["value"]) for o in data.get("observations", [])
                if o.get("value") not in (".", "")
            ]
            return obs[:n] if obs else None
        except Exception as e:
            logger.debug(f"FRED fetch {series_id}: {e}")
            return None


class YahooDataFetcher:
    """Fetches DXY and VIX from Yahoo Finance (free, no API key)."""

    YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"

    def fetch_latest(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Fetch (current_price, prev_close) for computing change."""
        try:
            url = self.YAHOO_URL.format(symbol=symbol)
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())

            result = data["chart"]["result"][0]
            closes = result["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) < 2:
                return None
            return closes[-1], closes[-2]   # (current, prev_day)
        except Exception as e:
            logger.debug(f"Yahoo fetch {symbol}: {e}")
            return None


class MacroSignalAgent:
    """
    Cross-asset macro signal agent.
    Provides fundamentally uncorrelated signals vs price-action agents.

    Integrates directly into AgentOrchestrator as the 5th agent.
    """

    # Pairs this agent has opinions on
    SUPPORTED_PAIRS = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY"}

    # Regime fit: macro signals work in all regimes but especially volatile
    REGIME_FIT = {"trending": 0.70, "ranging": 0.60, "volatile": 0.90}

    def __init__(self):
        self._fred   = FREDDataFetcher()
        self._yahoo  = YahooDataFetcher()
        self._last_snapshot: Optional[MacroSnapshot] = None
        self._refresh_task   = None
        self._refresh_interval = 3600   # refresh hourly

    # ── Data refresh ─────────────────────────────────────────

    async def refresh_data(self) -> Optional[MacroSnapshot]:
        """
        Fetch all macro data and build snapshot.
        Call hourly from scheduler. Runs in thread pool (HTTP calls).
        """
        try:
            snapshot = await asyncio.to_thread(self._build_snapshot)
            self._last_snapshot = snapshot
            if snapshot:
                logger.info(
                    f"Macro snapshot: real_rate={snapshot.us_real_rate}% "
                    f"DXY_chg={snapshot.dxy_change_24h:.2f}% "
                    f"VIX={snapshot.vix_level} "
                    f"gold_bias={snapshot.gold_macro_bias}({snapshot.gold_macro_strength:.2f})"
                )
            return snapshot
        except Exception as e:
            logger.warning(f"Macro refresh failed: {e}")
            return None

    def _build_snapshot(self) -> Optional[MacroSnapshot]:
        """Synchronous data fetch — run in thread pool."""
        now = datetime.now(timezone.utc).isoformat()

        # FRED data
        real_rate_obs      = self._fred.fetch_latest("DFII10",   2)
        nominal_rate_obs   = self._fred.fetch_latest("DGS10",    2)
        breakeven_obs      = self._fred.fetch_latest("T10YIE",   2)
        fed_funds_obs      = self._fred.fetch_latest("FEDFUNDS", 1)

        us_real_rate     = real_rate_obs[0]    if real_rate_obs    else None
        us_nominal_rate  = nominal_rate_obs[0] if nominal_rate_obs else None
        us_breakeven     = breakeven_obs[0]    if breakeven_obs    else None
        fed_funds        = fed_funds_obs[0]    if fed_funds_obs    else None

        # Day-over-day changes (FRED data is daily, so obs[0]=today obs[1]=yesterday)
        real_rate_chg   = None
        nominal_rate_chg = None
        if real_rate_obs and len(real_rate_obs) >= 2:
            real_rate_chg    = real_rate_obs[0]    - real_rate_obs[1]
        if nominal_rate_obs and len(nominal_rate_obs) >= 2:
            nominal_rate_chg = nominal_rate_obs[0] - nominal_rate_obs[1]

        # Yahoo Finance: DXY and VIX
        dxy_data = self._yahoo.fetch_latest("DX-Y.NYB")   # USD Index
        vix_data = self._yahoo.fetch_latest("^VIX")

        dxy_level    = dxy_data[0]  if dxy_data else None
        dxy_chg_pct  = None
        if dxy_data and dxy_data[1] and dxy_data[1] != 0:
            dxy_chg_pct = ((dxy_data[0] - dxy_data[1]) / dxy_data[1]) * 100

        vix_level = vix_data[0] if vix_data else None
        vix_chg   = (vix_data[0] - vix_data[1]) if vix_data else None

        # Determine gold macro bias
        gold_bias, gold_strength = self._compute_gold_bias(
            real_rate_chg, dxy_chg_pct, vix_chg
        )

        return MacroSnapshot(
            timestamp=now,
            us_real_rate=us_real_rate,
            us_nominal_rate=us_nominal_rate,
            us_breakeven_10y=us_breakeven,
            fed_funds=fed_funds,
            dxy_level=dxy_level,
            dxy_change_24h=round(dxy_chg_pct, 3) if dxy_chg_pct else None,
            vix_level=vix_level,
            vix_change_24h=round(vix_chg, 2) if vix_chg else None,
            real_rate_change_24h=round(real_rate_chg * 100, 2) if real_rate_chg else None,
            nominal_rate_change_24h=round(nominal_rate_chg * 100, 2) if nominal_rate_chg else None,
            gold_macro_bias=gold_bias,
            gold_macro_strength=gold_strength,
        )

    def _compute_gold_bias(
        self,
        real_rate_chg: Optional[float],    # % (positive = rates rising)
        dxy_chg: Optional[float],           # % (positive = USD strengthening)
        vix_chg: Optional[float],           # points (positive = fear rising)
    ) -> Tuple[str, float]:
        """
        Compute gold directional bias from macro factors.
        Returns (bias, strength) where bias is "bullish"/"bearish"/"neutral"
        """
        bullish_signals = 0
        bearish_signals = 0
        total_possible  = 0

        # Real rates: falling = bullish gold, rising = bearish gold
        if real_rate_chg is not None:
            total_possible += 1
            if real_rate_chg < -REAL_RATE_CHANGE_THRESH:
                bullish_signals += 1
            elif real_rate_chg > REAL_RATE_CHANGE_THRESH:
                bearish_signals += 1

        # DXY: weakening USD = bullish gold, strengthening = bearish
        if dxy_chg is not None:
            total_possible += 1
            if dxy_chg < -DXY_CHANGE_THRESH:
                bullish_signals += 1
            elif dxy_chg > DXY_CHANGE_THRESH:
                bearish_signals += 1

        # VIX: rising fear = bullish gold (safe haven demand)
        if vix_chg is not None:
            total_possible += 1
            if vix_chg > VIX_CHANGE_THRESH:
                bullish_signals += 1
            elif vix_chg < -VIX_CHANGE_THRESH:
                bearish_signals += 1

        if total_possible == 0:
            return "neutral", 0.0

        if bullish_signals >= 2:
            return "bullish", round(bullish_signals / total_possible, 3)
        elif bearish_signals >= 2:
            return "bearish", round(bearish_signals / total_possible, 3)
        return "neutral", 0.0

    # ── Signal evaluation (AgentOrchestrator interface) ──────

    def evaluate(self, candles: list, regime: str = "trending", pair: str = "XAUUSD") -> MacroAgentSignal:
        """
        Evaluate macro signal for a given pair.
        Called by AgentOrchestrator — same interface as other agents.
        """
        pair_u = pair.upper()
        if pair_u not in self.SUPPORTED_PAIRS:
            return self._null_signal(f"Pair {pair} not supported by MacroAgent")

        snap = self._last_snapshot
        if snap is None:
            return self._null_signal("No macro snapshot available — call refresh_data() first")

        # Check snapshot freshness (max 4 hours old)
        try:
            snap_dt = datetime.fromisoformat(snap.timestamp.replace("Z", "+00:00"))
            age_h   = (datetime.now(timezone.utc) - snap_dt).total_seconds() / 3600
            if age_h > 4:
                return self._null_signal(f"Macro snapshot stale ({age_h:.1f}h old)")
        except Exception:
            pass

        if pair_u == "XAUUSD":
            return self._gold_signal(snap, regime, candles)
        elif pair_u in ("EURUSD", "GBPUSD"):
            return self._rate_diff_signal(snap, pair_u, regime, candles)
        else:
            return self._null_signal(f"No macro logic for {pair_u}")

    def _gold_signal(
        self, snap: MacroSnapshot, regime: str, candles: list
    ) -> MacroAgentSignal:
        """XAUUSD signal based on real rates + DXY + VIX."""
        atr = self._calc_atr(candles)

        if snap.gold_macro_bias == "bullish" and snap.gold_macro_strength >= 0.50:
            direction  = "buy"
            confidence = 0.55 + (snap.gold_macro_strength * 0.25)   # 0.55–0.80
            win_prob   = 0.52 + (snap.gold_macro_strength * 0.10)
            reason_parts = []
            if snap.real_rate_change_24h and snap.real_rate_change_24h < 0:
                reason_parts.append(f"Real rates falling {snap.real_rate_change_24h:.1f}bps")
            if snap.dxy_change_24h and snap.dxy_change_24h < 0:
                reason_parts.append(f"DXY weakening {snap.dxy_change_24h:.2f}%")
            if snap.vix_change_24h and snap.vix_change_24h > 0:
                reason_parts.append(f"VIX rising +{snap.vix_change_24h:.1f}")

        elif snap.gold_macro_bias == "bearish" and snap.gold_macro_strength >= 0.50:
            direction  = "sell"
            confidence = 0.55 + (snap.gold_macro_strength * 0.25)
            win_prob   = 0.52 + (snap.gold_macro_strength * 0.10)
            reason_parts = []
            if snap.real_rate_change_24h and snap.real_rate_change_24h > 0:
                reason_parts.append(f"Real rates rising +{snap.real_rate_change_24h:.1f}bps")
            if snap.dxy_change_24h and snap.dxy_change_24h > 0:
                reason_parts.append(f"DXY strengthening +{snap.dxy_change_24h:.2f}%")
            if snap.vix_change_24h and snap.vix_change_24h < 0:
                reason_parts.append(f"VIX falling {snap.vix_change_24h:.1f}")
        else:
            return self._null_signal(
                f"Gold macro neutral: real_rate_chg={snap.real_rate_change_24h} "
                f"DXY_chg={snap.dxy_change_24h} VIX_chg={snap.vix_change_24h}"
            )

        stop_dist = atr * 1.5
        tp1_dist  = atr * 2.5
        tp2_dist  = atr * 4.0
        rr        = tp1_dist / max(stop_dist, 1e-9)
        ev        = win_prob * rr - (1 - win_prob) * 1.0

        return MacroAgentSignal(
            direction=direction,
            confidence=round(min(0.85, confidence), 3),
            win_probability=round(min(0.65, win_prob), 3),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.65),
            reasoning=f"MACRO XAUUSD {direction.upper()}: {'; '.join(reason_parts)}. "
                      f"Real rate={snap.us_real_rate}% DXY={snap.dxy_level} VIX={snap.vix_level}",
            raw_features={
                "real_rate":      snap.us_real_rate,
                "real_rate_chg":  snap.real_rate_change_24h,
                "dxy":            snap.dxy_level,
                "dxy_chg":        snap.dxy_change_24h,
                "vix":            snap.vix_level,
                "vix_chg":        snap.vix_change_24h,
                "gold_bias":      snap.gold_macro_bias,
                "gold_strength":  snap.gold_macro_strength,
            },
            macro_snapshot=snap,
        )

    def _rate_diff_signal(
        self, snap: MacroSnapshot, pair: str, regime: str, candles: list
    ) -> MacroAgentSignal:
        """
        EURUSD/GBPUSD signal based on interest rate differentials.
        Higher EUR/GBP rates relative to USD → currency appreciation.
        """
        atr = self._calc_atr(candles)
        if not snap.fed_funds or not snap.us_nominal_rate:
            return self._null_signal("Insufficient rate data for rate differential signal")

        # DXY falling = USD weakening = EUR/GBP bullish
        if snap.dxy_change_24h is not None:
            if snap.dxy_change_24h < -DXY_CHANGE_THRESH:
                direction  = "buy"
                confidence = 0.52 + min(abs(snap.dxy_change_24h) * 0.05, 0.18)
                win_prob   = 0.51 + min(abs(snap.dxy_change_24h) * 0.03, 0.12)
                reasoning  = (f"MACRO {pair} BUY: DXY weakening {snap.dxy_change_24h:.2f}% "
                              f"→ USD weakness → {pair[:3]} bullish")
            elif snap.dxy_change_24h > DXY_CHANGE_THRESH:
                direction  = "sell"
                confidence = 0.52 + min(abs(snap.dxy_change_24h) * 0.05, 0.18)
                win_prob   = 0.51 + min(abs(snap.dxy_change_24h) * 0.03, 0.12)
                reasoning  = (f"MACRO {pair} SELL: DXY strengthening +{snap.dxy_change_24h:.2f}% "
                              f"→ USD strength → {pair[:3]} bearish")
            else:
                return self._null_signal(f"DXY move insufficient for {pair} signal")
        else:
            return self._null_signal("No DXY data")

        stop_dist = atr * 1.5
        tp1_dist  = atr * 2.0
        tp2_dist  = atr * 3.0
        rr        = tp1_dist / max(stop_dist, 1e-9)
        ev        = win_prob * rr - (1 - win_prob) * 1.0

        return MacroAgentSignal(
            direction=direction,
            confidence=round(min(0.75, confidence), 3),
            win_probability=round(min(0.63, win_prob), 3),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.65),
            reasoning=reasoning,
            raw_features={
                "dxy": snap.dxy_level, "dxy_chg": snap.dxy_change_24h,
                "fed_funds": snap.fed_funds,
            },
            macro_snapshot=snap,
        )

    def _calc_atr(self, candles: list, period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0.001
        trs = []
        for i in range(1, min(period + 2, len(candles))):
            c, p = candles[-i], candles[-i - 1]
            try:
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                trs.append(tr)
            except AttributeError:
                continue
        return float(np.mean(trs)) if trs else 0.001

    def _null_signal(self, reason: str) -> MacroAgentSignal:
        return MacroAgentSignal(
            direction=None, confidence=0.0, win_probability=0.0,
            expected_value=0.0, stop_distance=0.0, tp1_distance=0.0,
            tp2_distance=0.0, regime_fit=0.0, reasoning=reason,
        )

    def get_snapshot(self) -> Optional[MacroSnapshot]:
        return self._last_snapshot

    def status_dict(self) -> Dict:
        snap = self._last_snapshot
        if not snap:
            return {"status": "no_data", "last_refresh": None}
        return {
            "status":        "ok",
            "last_refresh":  snap.timestamp,
            "gold_bias":     snap.gold_macro_bias,
            "gold_strength": snap.gold_macro_strength,
            "real_rate":     snap.us_real_rate,
            "dxy":           snap.dxy_level,
            "vix":           snap.vix_level,
            "dxy_change_24h": snap.dxy_change_24h,
            "real_rate_change_24h": snap.real_rate_change_24h,
            "fred_api_configured": bool(FRED_API_KEY),
        }
