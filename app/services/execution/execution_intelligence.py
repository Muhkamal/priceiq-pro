"""
PriceIQ Pro — Execution Intelligence
Session-aware slippage and fill price estimator.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class ExecutionResult:
    fill_price: float
    slippage_pips: float
    session: str
    spread_pips: float
    adjusted_entry: float


class ExecutionIntelligence:
    """Session-aware slippage and fill price estimator."""

    SPREAD_PIPS = {
        "XAUUSD": 3.0, "EURUSD": 1.2, "GBPUSD": 1.5,
        "USDJPY": 1.3, "USDCHF": 1.8, "AUDUSD": 1.4,
    }

    SESSION_SLIPPAGE = {
        "london": 0.5, "london_newyork": 0.3, "newyork": 0.6,
        "tokyo": 0.8, "sydney": 1.0, "other": 0.7,
    }

    def __init__(self, live_mode: bool = False, **kwargs):
        self.live_mode = live_mode

    def estimate_fill(
        self,
        pair: str,
        direction: str,
        signal_price: float,
        session: str = "london",
        atr: float = 0.001,
    ) -> ExecutionResult:

        pip = 0.1 if "XAU" in pair else (0.01 if "JPY" in pair else 0.0001)

        spread_pips = self.SPREAD_PIPS.get(pair.upper(), 2.0)
        slip_pips   = self.SESSION_SLIPPAGE.get(session, 0.7)

        total_pips  = spread_pips + slip_pips
        slippage    = total_pips * pip

        fill = signal_price + slippage if direction == "buy" else signal_price - slippage

        return ExecutionResult(
            fill_price=round(fill, 6),
            slippage_pips=round(total_pips, 2),
            session=session,
            spread_pips=spread_pips,
            adjusted_entry=round(fill, 6),
        )

    def simulate_fill(
        self,
        pair: str,
        direction: str,
        signal_price: float = 0.0,
        session: str = "london",
        atr: float = 0.001,
        requested_price: float = None,
        order_type: str = "market",
        lots: float = 0.01,
        **kwargs,
    ) -> "ExecutionResult":
        """Alias for estimate_fill — accepts all orchestrator kwargs."""
        price = requested_price if requested_price is not None else signal_price
        return self.estimate_fill(pair, direction, price, session, atr)

    def get_session(self) -> str:
        h = datetime.now(timezone.utc).hour

        if 7 <= h < 12:
            return "london"
        if 12 <= h < 16:
            return "london_newyork"
        if 0 <= h < 7:
            return "tokyo"

        return "other"
