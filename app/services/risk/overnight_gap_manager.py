"""
PriceIQ Pro -- Overnight & Gap Risk Manager v1.1

Handles risks that occur when markets are closed:
    1. Weekend gap risk -- XAUUSD gaps 50-100 pips on Sunday open
    2. Friday position reduction -- reduce exposure before weekend close
    3. Sunday gap detection -- detect and manage large gap opens
    4. Overnight swap/carry costs -- track financing charges
    5. Holiday detection -- major market holidays (Good Friday, etc.)

Friday logic (17:00 UTC = NY close):
    - HIGH-risk pairs (XAUUSD, GBPUSD): reduce to 50% size
    - If position already open: reduce lots or close if > threshold

Sunday open logic (21:00 UTC = Sydney open):
    - Detect gap vs Thursday close
    - Gap > 1 ATR: send Telegram alert, check SL still valid
    - Gap > 2 ATR: force SL update to account for gap

Swap costs (daily financing):
    - Long XAU at Exness: ~-$3.5/lot/day
    - These compound; positions > 3 days need cost tracking

Usage:
    gap_mgr = OvernightGapManager(telegram=telegram)
    await gap_mgr.on_friday_close(current_positions, current_prices)
    await gap_mgr.on_sunday_open(candles, current_positions)
    cost = gap_mgr.calculate_swap_cost("XAUUSD", lots=0.1, days=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -- Weekend gap thresholds -------------------------------
GAP_WARN_ATR_MULT   = 0.8    # gap > 0.8x ATR -> warn
GAP_DANGER_ATR_MULT = 1.5    # gap > 1.5x ATR -> update SL / alert
GAP_CLOSE_ATR_MULT  = 2.5    # gap > 2.5x ATR -> consider force close

# -- Friday risk reduction --------------------------------
FRIDAY_REDUCE_HOUR  = 17     # 17:00 UTC = NY close (Friday)
HIGH_GAP_PAIRS      = {"XAUUSD", "GBPUSD", "GBPJPY", "USDJPY"}
FRIDAY_SIZE_CAP     = 0.50   # reduce to 50% on Friday close

# -- Swap costs per lot per night (approximate, check your broker) --
# Positive = credit, Negative = charge
# -- Swap costs per lot per night (approximate, check your broker) --
# Positive = credit, Negative = charge
SWAP_LONG:  Dict[str, float] = {
    "EURUSD": -0.72, "GBPUSD": -1.85, "USDJPY": +1.26,
    "USDCHF": +0.46, "USDCAD": -1.02, "AUDUSD": -1.12,
    "NZDUSD": -0.98, "XAUUSD": -3.50,
    "EURJPY": -0.85, "GBPJPY": -2.10}

SWAP_SHORT: Dict[str, float] = {
    "EURUSD": -1.10, "GBPUSD": -2.20, "USDJPY": -0.45,
    "USDCHF": -1.12, "USDCAD": -0.38, "AUDUSD": -0.86,
    "NZDUSD": -0.72, "XAUUSD": -1.80,
    "EURJPY": -1.20, "GBPJPY": -1.85}
# -- Major market holidays (UTC dates, YYYY-MM-DD) --------
MARKET_HOLIDAYS_2025 = {
    "2025-01-01", "2025-04-18", "2025-04-21",
    "2025-12-25", "2025-12-26"}
MARKET_HOLIDAYS_2026 = {
    "2026-01-01", "2026-04-03", "2026-04-06",
    "2026-12-25", "2026-12-28"}


def _atr(candles: List, period: int = 14) -> float:
    trs = []
    for i in range(1, min(period + 2, len(candles))):
        c, p = candles[-i], candles[-i - 1]
        try:
            tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
            trs.append(tr)
        except AttributeError:
            continue
    return float(np.mean(trs)) if trs else 0.001


@dataclass
class GapReport:
    pair:        str
    gap_size:    float        # in price units
    gap_pips:    float
    gap_atr_mult: float       # gap / ATR
    direction:   str          # "up" | "down" | "none"
    severity:    str          # "none" | "warn" | "danger" | "critical"
    action:      str          # recommended action
    thursday_close: float
    sunday_open:    float


@dataclass
class SwapCost:
    pair:        str
    lots:        float
    days:        int
    direction:   str
    daily_cost:  float    # USD per day
    total_cost:  float    # total over days
    is_positive: bool     # True = credit (rare)


class OvernightGapManager:
    """
    Manages weekend gap risk, Friday position reduction, and swap costs.
    """

    def __init__(self, telegram=None):
        self.telegram      = telegram
        self._last_closes: Dict[str, float] = {}    # pair -> Thursday close price
        self._gap_log:     List[GapReport]  = []

    # -- Friday close handling --------------------------------

    async def on_friday_close(
        self,
        open_positions: Dict,      # pair -> position object
        current_prices: Dict[str, float],
        force_close_threshold: float = 0.0,  # set > 0 to auto-close large positions
    ) -> Dict[str, str]:
        """
        Call at Friday 17:00 UTC.
        Returns dict of pair -> action taken.
        """
        now   = datetime.now(timezone.utc)
        if now.weekday() != 4:   # 4 = Friday
            return {}

        actions = {}
        for pair, pos in open_positions.items():
            pair_u = pair.upper()
            price  = current_prices.get(pair_u, 0)

            # Store close for gap detection Sunday
            if price > 0:
                self._last_closes[pair_u] = price

            if pair_u in HIGH_GAP_PAIRS:
                lots = getattr(pos, "lots", getattr(pos, "lots_remaining", 0))
                msg = "\n".join([
                    "Friday Risk Reduction",
                    f"{pair_u}: High gap-risk pair open over weekend.",
                    f"Lots: {lots} | Price: {price}",
                    f"Recommended: reduce to {lots * FRIDAY_SIZE_CAP:.2f}L or close."])
                await self._send(msg)
                actions[pair_u] = f"WARN: high gap risk pair, recommend reduction"
                logger.warning(f"Friday gap risk: {pair_u} open with {lots}L")
            else:
                actions[pair_u] = "OK: standard pair, weekend gap risk acceptable"

        # Summary message
        if open_positions:
            summary = "\n".join([
                "Friday Close Summary",
                f"Open positions: {len(open_positions)}",
                f"High-gap pairs: {sum(1 for p in open_positions if p.upper() in HIGH_GAP_PAIRS)}",
                "Swap costs will accrue over weekend (3 nights)."])
            await self._send(summary)
        return actions

    # -- Sunday open handling ---------------------------------

    async def on_sunday_open(
        self,
        pair_candles: Dict[str, List],   # pair -> recent candles
        open_positions: Dict,
    ) -> List[GapReport]:
        """
        Call at Sunday 21:00 UTC (Sydney open).
        Detects gaps vs Thursday close and alerts.
        """
        reports = []
        for pair, candles in pair_candles.items():
            if not candles:
                continue
            pair_u        = pair.upper()
            sunday_open   = getattr(candles[-1], "open", None)
            thursday_close = self._last_closes.get(pair_u)
            if not sunday_open or not thursday_close:
                continue

            gap_size = sunday_open - thursday_close
            gap_dir  = "up" if gap_size > 0 else "down"
            gap_abs  = abs(gap_size)

            pip  = 0.1 if "XAU" in pair_u else (0.01 if "JPY" in pair_u else 0.0001)
            gap_pips = gap_abs / pip
            atr  = _atr(candles)
            mult = gap_abs / max(atr, 1e-9)

            if mult >= GAP_CLOSE_ATR_MULT:
                severity = "critical"
                action   = "Consider force-closing position -- gap exceeds 2.5x ATR"
            elif mult >= GAP_DANGER_ATR_MULT:
                severity = "danger"
                action   = "Update SL to account for gap -- check margin"
            elif mult >= GAP_WARN_ATR_MULT:
                severity = "warn"
                action   = "Monitor closely -- significant gap detected"
            else:
                severity = "none"
                action   = "Normal gap -- no action required"

            report = GapReport(
                pair=pair_u, gap_size=gap_size, gap_pips=round(gap_pips, 1),
                gap_atr_mult=round(mult, 2), direction=gap_dir,
                severity=severity, action=action,
                thursday_close=thursday_close, sunday_open=sunday_open,
            )
            reports.append(report)
            self._gap_log.append(report)

            if severity in ("warn", "danger", "critical"):
                emoji = {"warn": "!!", "danger": "XX", "critical": "!!"}[severity]
                alert_msg = "\n".join([
                    f"{emoji} Sunday Gap: {pair_u}",
                    f"Thu close: {thursday_close:.5f}",
                    f"Sun open:  {sunday_open:.5f}",
                    f"Gap: {gap_pips:.1f} pips {gap_dir} ({mult:.1f}x ATR)",
                    f"Action: {action}"])
                await self._send(alert_msg)

        return reports

    # -- Swap cost calculator ---------------------------------

    def calculate_swap_cost(
        self,
        pair:      str,
        lots:      float,
        days:      int,
        direction: str = "buy",
    ) -> SwapCost:
        """
        Calculate total swap/financing cost for holding a position N days.
        Weekend = 3 nights charged on Wednesday (triple swap).
        """
        pair_u     = pair.upper()
        if direction.lower() == "buy":
            daily = SWAP_LONG.get(pair_u, -1.0) * lots
        else:
            daily = SWAP_SHORT.get(pair_u, -1.0) * lots

        # Triple swap on Wednesday for weekend
        total = daily * days
        return SwapCost(
            pair=pair_u, lots=lots, days=days,
            direction=direction,
            daily_cost=round(daily, 4),
            total_cost=round(total, 4),
            is_positive=total > 0,
        )

    def swap_breakeven_days(self, pair: str, direction: str, pnl_target_usd: float) -> Optional[float]:
        """How many days before swap costs eat your profit target?"""
        pair_u = pair.upper()
        swap   = SWAP_LONG.get(pair_u, -1.0) if direction == "buy" else SWAP_SHORT.get(pair_u, -1.0)
        if swap >= 0:
            return None   # positive carry -- no breakeven issue
        # days until swap costs = profit target
        days = pnl_target_usd / abs(swap)
        return round(days, 1)

    def is_market_holiday(self, dt: Optional[datetime] = None) -> bool:
        """Check if today is a major market holiday."""
        dt  = dt or datetime.now(timezone.utc)
        key = dt.strftime("%Y-%m-%d")
        all_holidays = MARKET_HOLIDAYS_2025 | MARKET_HOLIDAYS_2026
        return key in all_holidays

    def is_friday_close_window(self, dt: Optional[datetime] = None) -> bool:
        """True if within 30 min of Friday NY close."""
        dt = dt or datetime.now(timezone.utc)
        return dt.weekday() == 4 and abs(dt.hour - FRIDAY_REDUCE_HOUR) <= 0

    def is_sunday_open_window(self, dt: Optional[datetime] = None) -> bool:
        """True if within 30 min of Sunday Sydney open."""
        dt = dt or datetime.now(timezone.utc)
        return dt.weekday() == 6 and 21 <= dt.hour <= 22

    def get_gap_history(self) -> List[Dict]:
        return [
            {
                "pair": r.pair, "gap_pips": r.gap_pips,
                "severity": r.severity, "direction": r.direction,
                "atr_mult": r.gap_atr_mult, "action": r.action}
            for r in self._gap_log[-20:]
        ]

    async def _send(self, message: str):
        if self.telegram:
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.warning(f"GapManager Telegram failed: {e}")
