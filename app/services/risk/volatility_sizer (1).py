"""
PriceIQ Pro — Volatility-Targeted Position Sizer v1.0

Replaces fixed-percentage risk sizing with volatility-adjusted sizing.

Problem with fixed 2% risk:
    - 2% risk when XAUUSD ATR = 5.0 pips → 1 lot
    - 2% risk when XAUUSD ATR = 25.0 pips (NFP day) → 0.2 lots
    But both use the same stop = 1.5× ATR → the dollar risk is the same,
    but the market is 5× more violent. Fixed % ignores this.

Solution — Volatility Targeting:
    Target a FIXED daily dollar volatility on the account (e.g. $200 per day).
    When market vol is high → trade smaller.
    When market vol is low  → trade larger (up to a cap).

    Size = (target_vol_usd) / (pair_daily_vol_usd_per_lot)

Kelly Criterion (optional):
    Full Kelly = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
    Half Kelly  = Full Kelly × 0.5   (much safer in practice)
    Use Kelly as a SIZE CAP, not the primary sizer.

Usage:
    sizer = VolatilityTargetedSizer(
        account_balance=10_000,
        target_vol_pct=0.02,   # target 2% daily vol on account
    )

    lots = sizer.compute_lots(
        pair="XAUUSD",
        atr=15.0,                # current ATR in price units
        win_rate=0.55,
        avg_win_r=1.8,
        avg_loss_r=1.0,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# USD pip values per standard lot (100,000 units)
PIP_VALUE_USD: Dict[str, float] = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY": 9.09, "USDCHF": 10.0, "USDCAD": 7.50,
    "EURCAD": 7.50, "GBPCAD": 7.50, "EURGBP": 12.50,
    "EURJPY": 9.09, "GBPJPY": 9.09, "AUDJPY": 9.09,
    "CADJPY": 9.09, "NZDJPY": 9.09, "CHFJPY": 9.09,
    "XAUUSD": 10.0,   # $1 per 0.1 pip × 100 oz = $10 per pip per lot
    "XAGUSD": 50.0,
}

PIP_SIZE: Dict[str, float] = {
    "XAUUSD": 0.1, "XAGUSD": 0.001,
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "AUDJPY": 0.01, "CADJPY": 0.01, "NZDJPY": 0.01, "CHFJPY": 0.01,
}

MAX_LOTS = 10.0
MIN_LOTS = 0.01


def _pip_size(pair: str) -> float:
    return PIP_SIZE.get(pair.upper(), 0.0001)


def _pip_value(pair: str, current_price: Optional[float] = None) -> float:
    pair = pair.upper()
    usd_base = {"USDJPY", "USDCHF", "USDCAD", "USDMXN"}
    if pair in usd_base and current_price and current_price > 0:
        return _pip_size(pair) * 100_000 / current_price
    return PIP_VALUE_USD.get(pair, 10.0)


@dataclass
class SizingResult:
    lots:            float
    method:          str     # "vol_target" | "kelly" | "fixed_risk" | "fallback"
    risk_pct:        float   # actual % of account being risked
    risk_usd:        float
    kelly_fraction:  Optional[float]
    vol_target_lots: Optional[float]
    stop_pips:       float
    reasoning:       str


class VolatilityTargetedSizer:
    """
    Volatility-targeted position sizer with Kelly criterion cap.

    Sizing priority:
        1. Compute vol-targeted size  (primary)
        2. Compute Kelly cap          (safety ceiling)
        3. Apply hard risk cap        (never > max_risk_pct of balance)
        4. Clamp to [MIN_LOTS, MAX_LOTS]
    """

    def __init__(
        self,
        account_balance:  float = 10_000.0,
        target_vol_pct:   float = 0.02,    # 2% daily vol target
        max_risk_pct:     float = 0.03,    # never risk more than 3% per trade
        use_kelly:        bool  = True,
        kelly_fraction:   float = 0.5,     # half-Kelly
        atr_period_days:  float = 1.0,     # ATR expressed in N-day window
    ):
        self.account_balance = account_balance
        self.target_vol_pct  = target_vol_pct
        self.max_risk_pct    = max_risk_pct
        self.use_kelly       = use_kelly
        self.kelly_fraction  = kelly_fraction
        self.atr_period_days = atr_period_days

    def update_balance(self, new_balance: float):
        self.account_balance = new_balance

    def compute_lots(
        self,
        pair:            str,
        atr:             float,             # ATR in price units
        stop_distance:   Optional[float] = None,   # if None, use 1.5 × ATR
        win_rate:        float = 0.50,
        avg_win_r:       float = 1.5,
        avg_loss_r:      float = 1.0,
        current_price:   Optional[float] = None,
        historical_atr:  Optional[float] = None,   # long-run ATR for vol scaling
    ) -> SizingResult:
        """
        Compute optimal lot size using volatility targeting + Kelly cap.

        Args:
            pair:           instrument symbol
            atr:            current ATR in price units
            stop_distance:  stop loss in price units (defaults to 1.5 × ATR)
            win_rate:       empirical win rate from learning loop
            avg_win_r:      average win in R-multiples
            avg_loss_r:     average loss in R-multiples (usually 1.0)
            current_price:  for accurate pip value on USD-base pairs
            historical_atr: baseline ATR to scale vol (if None, no scaling)
        """
        pip   = _pip_size(pair)
        pv    = _pip_value(pair, current_price)
        stop  = stop_distance if stop_distance else atr * 1.5
        stop_pips = max(stop / pip, 1.0)

        # ── 1. Volatility-targeted size ──────────────────────
        # daily_vol_usd_per_lot = ATR_in_pips × pip_value_per_lot
        atr_pips             = atr / pip
        daily_vol_per_lot    = atr_pips * pv   # USD vol per lot per ATR period
        target_vol_usd       = self.account_balance * self.target_vol_pct

        # Vol scaling: if market is 2× more volatile than normal → halve size
        vol_scale = 1.0
        if historical_atr and historical_atr > 0:
            vol_ratio = atr / historical_atr
            vol_scale = 1.0 / max(vol_ratio, 0.2)   # floor at 0.2 to prevent blowup

        vol_target_lots = (target_vol_usd / max(daily_vol_per_lot, 0.01)) * vol_scale
        vol_target_lots = round(vol_target_lots, 2)

        # ── 2. Kelly fraction ────────────────────────────────
        kelly_lots  = None
        kelly_frac  = None
        if self.use_kelly and win_rate > 0 and avg_win_r > 0:
            # Kelly formula: f* = (p × b - q) / b
            # where p=win_rate, q=1-win_rate, b=avg_win_r/avg_loss_r
            b = avg_win_r / max(avg_loss_r, 0.01)
            q = 1.0 - win_rate
            full_kelly = (win_rate * b - q) / b
            half_kelly = full_kelly * self.kelly_fraction
            kelly_frac = max(0.0, half_kelly)   # never negative

            # Kelly lot size based on risk (stop-loss based)
            kelly_risk_usd  = self.account_balance * kelly_frac
            kelly_lots = kelly_risk_usd / max(stop_pips * pv, 0.01)
            kelly_lots = round(max(MIN_LOTS, kelly_lots), 2)

        # ── 3. Hard risk cap ─────────────────────────────────
        max_risk_usd  = self.account_balance * self.max_risk_pct
        max_risk_lots = max_risk_usd / max(stop_pips * pv, 0.01)

        # ── 4. Select and clamp ──────────────────────────────
        candidates = [vol_target_lots]
        if kelly_lots is not None:
            candidates.append(kelly_lots)

        raw_lots = min(candidates)                  # take the more conservative estimate
        raw_lots = min(raw_lots, max_risk_lots)     # hard risk cap
        raw_lots = min(raw_lots, MAX_LOTS)          # hard lot cap
        raw_lots = max(raw_lots, MIN_LOTS)          # minimum size
        final_lots = round(raw_lots, 2)

        # Compute actual risk metrics
        actual_risk_usd = final_lots * stop_pips * pv
        actual_risk_pct = actual_risk_usd / max(self.account_balance, 1.0)

        method = "vol_target"
        if kelly_lots is not None and kelly_lots < vol_target_lots:
            method = "kelly_capped"
        if max_risk_lots < vol_target_lots:
            method = "risk_capped"

        reasoning = (
            f"Vol-target: {vol_target_lots:.2f}L "
            f"(ATR={atr_pips:.1f}pip, scale={vol_scale:.2f})"
        )
        if kelly_lots:
            reasoning += f" | Kelly({self.kelly_fraction}×): {kelly_lots:.2f}L"
        reasoning += f" | RiskCap: {max_risk_lots:.2f}L → Final: {final_lots:.2f}L ({actual_risk_pct:.1%} risk)"

        return SizingResult(
            lots=final_lots,
            method=method,
            risk_pct=round(actual_risk_pct, 4),
            risk_usd=round(actual_risk_usd, 2),
            kelly_fraction=round(kelly_frac, 4) if kelly_frac is not None else None,
            vol_target_lots=vol_target_lots,
            stop_pips=round(stop_pips, 1),
            reasoning=reasoning,
        )

    def describe(self) -> Dict:
        return {
            "account_balance": self.account_balance,
            "target_vol_pct":  self.target_vol_pct,
            "max_risk_pct":    self.max_risk_pct,
            "use_kelly":       self.use_kelly,
            "kelly_fraction":  self.kelly_fraction,
        }
