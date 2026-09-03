"""
PriceIQ Pro — Volatility-Targeted Position Sizer v1.2 (CRYPTO FIXED)

Fix: Adds BTCUSD/ETHUSD to pip dictionaries so position sizing 
     doesn't break on high-ATR crypto assets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

PIP_VALUE_USD: Dict[str, float] = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY": 9.09, "USDCHF": 10.0, "USDCAD": 7.50,
    "EURCAD": 7.50, "GBPCAD": 7.50, "EURGBP": 12.50,
    "EURJPY": 9.09, "GBPJPY": 9.09, "AUDJPY": 9.09,
    "CADJPY": 9.09, "NZDJPY": 9.09, "CHFJPY": 9.09,
    "XAUUSD": 10.0,
    # ═══ NEW: Crypto pip values (1 lot = 1 coin, $1 move = $1 per lot) ═══
    "BTCUSD": 1.0,
    "ETHUSD": 1.0}
    "ETHUSD": 1.0}

PIP_SIZE: Dict[str, float] = {
    "XAUUSD": 0.1,
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "AUDJPY": 0.01, "CADJPY": 0.01, "NZDJPY": 0.01, "CHFJPY": 0.01,
    # ═══ NEW: Crypto pip sizes ($1.00 move = 1 pip) ═══
    "BTCUSD": 1.0,
    "ETHUSD": 1.0}

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
    method:          str
    risk_pct:        float
    risk_usd:        float
    kelly_fraction:  Optional[float]
    vol_target_lots: Optional[float]
    stop_pips:       float
    reasoning:       str


class VolatilityTargetedSizer:
    """
    Volatility-targeted position sizer with Kelly criterion cap.

    CRITICAL: Call update_balance() with your REAL account balance
    before compute_lots(), or sizing will be wrong by 4×+.
    """

    def __init__(
        self,
        account_balance:  float = 10_000.0,
        target_vol_pct:   float = 0.02,
        max_risk_pct:     float = 0.03,
        use_kelly:        bool  = True,
        kelly_fraction:   float = 0.5,
        atr_period_days:  float = 1.0,
    ):
        self.account_balance = account_balance
        self.target_vol_pct  = target_vol_pct
        self.max_risk_pct    = max_risk_pct
        self.use_kelly       = use_kelly
        self.kelly_fraction  = kelly_fraction
        self.atr_period_days = atr_period_days
        self._default_balance = account_balance  # remember default for warning

    def update_balance(self, new_balance: float):
        if new_balance <= 0:
            logger.error(f"Sizer received invalid balance {new_balance} — keeping {self.account_balance}")
            return
        if self.account_balance == self._default_balance:
            logger.info(f"Sizer balance initialized: ${new_balance:,.2f}")
        self.account_balance = new_balance

    def compute_lots(
        self,
        pair:            str,
        atr:             float,
        stop_distance:   Optional[float] = None,
        win_rate:        float = 0.50,
        avg_win_r:       float = 1.5,
        avg_loss_r:      float = 1.0,
        current_price:   Optional[float] = None,
        historical_atr:  Optional[float] = None,
    ) -> SizingResult:

        # ── DEFENSIVE: warn if balance was never updated from default ──
        if self.account_balance == self._default_balance and self._default_balance == 10_000.0:
            logger.critical(
                f"SIZER WARNING: compute_lots() called with DEFAULT balance $10,000. "
                f"Call update_balance({{YOUR_REAL_BALANCE}}) first or lots will be oversized!"
            )

        pip   = _pip_size(pair)
        pv    = _pip_value(pair, current_price)
        stop  = stop_distance if stop_distance else atr * 1.5
        stop_pips = max(stop / pip, 1.0)

        atr_pips          = atr / pip
        daily_vol_per_lot = atr_pips * pv
        target_vol_usd    = self.account_balance * self.target_vol_pct

        vol_scale = 1.0
        if historical_atr and historical_atr > 0:
            vol_ratio = atr / historical_atr
            vol_scale = 1.0 / max(vol_ratio, 0.2)

        vol_target_lots = (target_vol_usd / max(daily_vol_per_lot, 0.01)) * vol_scale
        vol_target_lots = round(vol_target_lots, 2)

        kelly_lots  = None
        kelly_frac  = None
        if self.use_kelly and win_rate > 0 and avg_win_r > 0:
            b = avg_win_r / max(avg_loss_r, 0.01)
            q = 1.0 - win_rate
            full_kelly = (win_rate * b - q) / b
            half_kelly = full_kelly * self.kelly_fraction
            kelly_frac = max(0.0, half_kelly)

            kelly_risk_usd = self.account_balance * kelly_frac
            kelly_lots = kelly_risk_usd / max(stop_pips * pv, 0.01)
            kelly_lots = round(max(MIN_LOTS, kelly_lots), 2)

        max_risk_usd  = self.account_balance * self.max_risk_pct
        max_risk_lots = max_risk_usd / max(stop_pips * pv, 0.01)

        candidates = [vol_target_lots]
        if kelly_lots is not None:
            candidates.append(kelly_lots)

        raw_lots = min(candidates)
        raw_lots = min(raw_lots, max_risk_lots)
        raw_lots = min(raw_lots, MAX_LOTS)
        raw_lots = max(raw_lots, MIN_LOTS)
        final_lots = round(raw_lots, 2)

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
            reasoning += f" | Kelly({self.kelly_fraction}x): {kelly_lots:.2f}L"
        reasoning += f" | RiskCap: {max_risk_lots:.2f}L -> Final: {final_lots:.2f}L ({actual_risk_pct:.1%} risk)"

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
            "kelly_fraction":  self.kelly_fraction}
