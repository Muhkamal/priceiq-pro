"""
PriceIQ Pro — Correlation & Position Filter v1.1

Prevents over-exposure by detecting correlated open positions.

Problem: EURUSD long + GBPUSD long = you're not in 2 trades,
you're in 1 massive USD short with double the risk.

This filter:
  1. Tracks open positions
  2. Calculates net currency exposure
  3. Blocks new signals that would push any currency beyond its limit
  4. Warns on correlated pairs even if under hard limit
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import threading


class CorrelationGroup(str, Enum):
    USD_MAJORS  = "usd_majors"
    EUR_CROSSES = "eur_crosses"
    GBP_CROSSES = "gbp_crosses"
    JPY_CROSSES = "jpy_crosses"
    COMMODITY   = "commodity"


class OpenPosition(BaseModel):
    pair:        str
    direction:   str          # "buy" | "sell"
    lots:        float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss:   float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_amount: float = Field(ge=0)
    opened_at:   datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_long(self) -> bool:
        return self.direction == "buy"

    @property
    def is_short(self) -> bool:
        return self.direction == "sell"

    @property
    def pair_upper(self) -> str:
        return self.pair.upper()

    class Config:
        frozen = True


class CorrelationCheckResult(BaseModel):
    is_allowed:      bool
    reason:          Optional[str]      = None
    warnings:        List[str]          = Field(default_factory=list)
    net_exposure:    Dict[str, float]   = Field(default_factory=dict)
    correlated_open: List[str]          = Field(default_factory=list)
    checked_at:      datetime


# ── Correlation matrix (simplified) ───────────────────────────
# Positive = moves same direction, Negative = opposite
# Values: 1.0 = perfect correlation, -1.0 = perfect inverse
CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"):  0.85,
    ("EURUSD", "AUDUSD"):  0.75,
    ("EURUSD", "NZDUSD"):  0.70,
    ("EURUSD", "USDCHF"): -0.90,
    ("EURUSD", "USDJPY"): -0.65,
    ("GBPUSD", "AUDUSD"):  0.70,
    ("GBPUSD", "USDCHF"): -0.85,
    ("AUDUSD", "NZDUSD"):  0.95,
    ("USDJPY", "USDCHF"):  0.70,
    ("EURUSD", "EURGBP"):  0.60,
    ("GBPUSD", "GBPJPY"):  0.80,
    ("USDJPY", "EURJPY"):  0.70,
    ("XAUUSD", "EURUSD"):  0.65,
    ("XAUUSD", "AUDUSD"):  0.70,
    ("BTCUSD", "ETHUSD"):  0.85,
    ("BTCUSD", "SOLUSD"):  0.75,
    ("ETHUSD", "SOLUSD"):  0.80,
    # Add to CORRELATIONS dict
	("USDJPY", "EURJPY"):  0.70,
	("USDJPY", "GBPJPY"):  0.75,
	("GBPJPY", "EURJPY"):  0.80,
	("AUDUSD", "USDCAD"):  0.65,
	("NZDUSD", "USDCAD"):  0.60,
}

# Pip value per lot per pair (approximate, for exposure calculation)
PAIR_BASE_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "XAUUSD": ("XAU", "USD"), "BTCUSD": ("BTC", "USD"),
    "ETHUSD": ("ETH", "USD"), "SOLUSD": ("SOL", "USD"),
}

# Max net exposure per currency in standard lots
MAX_NET_LOTS_PER_CURRENCY: float = 3.0
# Correlation warning threshold
CORRELATION_WARNING_THRESHOLD: float = 0.70
# Correlation block threshold (very high correlation = practically same trade)
CORRELATION_BLOCK_THRESHOLD: float = 0.90


class CorrelationFilter:
    """
    Tracks open positions and prevents over-correlated new entries.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_lots_per_currency: float = MAX_NET_LOTS_PER_CURRENCY):
        self._positions: List[OpenPosition] = []
        self.max_lots_per_currency = max_lots_per_currency
        self._lock = threading.RLock()

    # ──────────────────────────────────────────────────────────
    # PUBLIC: Check before opening a new position
    # ──────────────────────────────────────────────────────────

    def check(
        self,
        new_pair:      str,
        new_direction: str,
        new_lots:      float,
    ) -> CorrelationCheckResult:
        """
        Check whether opening a new position would create dangerous correlation.

        Args:
            new_pair      : e.g. "EURUSD"
            new_direction : "buy" | "sell"
            new_lots      : position size in standard lots (must be > 0)
        """
        if new_lots <= 0:
            raise ValueError("new_lots must be positive")

        new_pair = new_pair.upper()
        new_direction = new_direction.lower()
        if new_direction not in ("buy", "sell"):
            raise ValueError("new_direction must be 'buy' or 'sell'")

        with self._lock:
            return self._check_locked(new_pair, new_direction, new_lots)

    def add_position(self, position: OpenPosition):
        """Register a new open position."""
        with self._lock:
            self._positions.append(position)

    def close_position(self, pair: str):
        """Remove a position when it's closed."""
        pair_upper = pair.upper()
        with self._lock:
            self._positions = [p for p in self._positions if p.pair_upper != pair_upper]

    def close_position_by_id(self, pair: str, entry_price: float, direction: str) -> bool:
        """
        Remove a specific position by pair, entry_price, and direction.
        Returns True if a position was removed.
        """
        pair_upper = pair.upper()
        direction_lower = direction.lower()
        with self._lock:
            original_len = len(self._positions)
            self._positions = [
                p for p in self._positions
                if not (p.pair_upper == pair_upper and p.entry_price == entry_price and p.direction == direction_lower)
            ]
            return len(self._positions) < original_len

    def get_open_positions(self) -> List[OpenPosition]:
        with self._lock:
            return list(self._positions)

    def get_net_exposure(self) -> Dict[str, float]:
        with self._lock:
            return self._calculate_net_exposure()

    def get_total_risk(self) -> float:
        """Total risk amount across all open positions."""
        with self._lock:
            return sum(p.risk_amount for p in self._positions)

    def get_position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def clear_all(self):
        """Clear all positions (e.g., end of trading session)."""
        with self._lock:
            self._positions.clear()

    # ──────────────────────────────────────────────────────────
    # Internal helpers (must be called with self._lock held)
    # ──────────────────────────────────────────────────────────

    def _check_locked(
        self,
        new_pair: str,
        new_direction: str,
        new_lots: float,
    ) -> CorrelationCheckResult:
        now = datetime.now(timezone.utc)
        warnings: List[str] = []
        correlated_open: List[str] = []

        # 1. Hard block: near-perfectly correlated pair already open in same direction
        for pos in self._positions:
            corr = self._get_correlation(new_pair, pos.pair_upper)
            if corr is None:
                continue

            same_direction = self._is_same_effective_direction(
                new_pair, new_direction, pos.pair_upper, pos.direction
            )

            effective_corr = corr if same_direction else -corr

            if effective_corr >= CORRELATION_BLOCK_THRESHOLD:
                return CorrelationCheckResult(
                    is_allowed=False,
                    reason=(
                        f"Already have a {pos.direction.upper()} on {pos.pair} "
                        f"({corr:.0%} correlation with {new_pair}). "
                        f"This would be the same trade with double risk."
                    ),
                    warnings=warnings,
                    net_exposure=self._calculate_net_exposure(),
                    correlated_open=[pos.pair],
                    checked_at=now,
                )

            if abs(effective_corr) >= CORRELATION_WARNING_THRESHOLD:
                correlated_open.append(pos.pair)
                direction_note = "same direction" if same_direction else "opposite direction"
                warnings.append(
                    f"{pos.pair} is {corr:.0%} correlated with {new_pair} ({direction_note})"
                )

        # 2. Currency exposure check
        exposure = self._calculate_net_exposure()
        new_exposure = self._add_exposure(exposure, new_pair, new_direction, new_lots)

        blocked_currency = None
        for currency, net_lots in new_exposure.items():
            if abs(net_lots) > self.max_lots_per_currency:
                blocked_currency = currency
                break

        if blocked_currency:
            current = exposure.get(blocked_currency, 0)
            return CorrelationCheckResult(
                is_allowed=False,
                reason=(
                    f"Adding {new_pair} would push {blocked_currency} exposure to "
                    f"{new_exposure[blocked_currency]:+.2f} lots "
                    f"(limit: ±{self.max_lots_per_currency:.1f} lots). "
                    f"Close an existing {blocked_currency} position first."
                ),
                warnings=warnings,
                net_exposure=new_exposure,
                correlated_open=correlated_open,
                checked_at=now,
            )

        return CorrelationCheckResult(
            is_allowed=True,
            warnings=warnings,
            net_exposure=new_exposure,
            correlated_open=correlated_open,
            checked_at=now,
        )

    def _get_correlation(self, pair1: str, pair2: str) -> Optional[float]:
        if pair1 == pair2:
            return 1.0
        key1 = (pair1, pair2)
        key2 = (pair2, pair1)
        return CORRELATIONS.get(key1) or CORRELATIONS.get(key2)

    def _is_same_effective_direction(
        self, pair1: str, dir1: str, pair2: str, dir2: str
    ) -> bool:
        """
        Two positions are 'same direction' if they both express the same
        currency view (e.g. EURUSD buy + GBPUSD buy = both long vs USD).
        """
        base1, quote1 = PAIR_BASE_CURRENCIES.get(pair1, (pair1[:3], pair1[3:]))
        base2, quote2 = PAIR_BASE_CURRENCIES.get(pair2, (pair2[:3], pair2[3:]))

        shared = set([base1, quote1]) & set([base2, quote2])
        if not shared:
            return dir1 == dir2

        shared_ccy = shared.pop()
        side1 = "long" if (shared_ccy == quote1 and dir1 == "buy") or (shared_ccy == base1 and dir1 == "sell") else "short"
        side2 = "long" if (shared_ccy == quote2 and dir2 == "buy") or (shared_ccy == base2 and dir2 == "sell") else "short"
        return side1 == side2

    def _calculate_net_exposure(self) -> Dict[str, float]:
        """Calculate net lot exposure per currency across all open positions."""
        exposure: Dict[str, float] = {}

        for pos in self._positions:
            pair = pos.pair_upper
            base, quote = PAIR_BASE_CURRENCIES.get(pair, (pair[:3], pair[3:]))
            sign = 1.0 if pos.direction == "buy" else -1.0

            exposure[base]  = exposure.get(base, 0.0) + sign * pos.lots
            exposure[quote] = exposure.get(quote, 0.0) - sign * pos.lots

        return {k: round(v, 4) for k, v in exposure.items()}

    def _add_exposure(
        self, current: Dict[str, float], pair: str, direction: str, lots: float
    ) -> Dict[str, float]:
        """Return a copy of exposure dict with a new position added."""
        result = dict(current)
        base, quote = PAIR_BASE_CURRENCIES.get(pair, (pair[:3], pair[3:]))
        sign = 1.0 if direction == "buy" else -1.0
        result[base]  = result.get(base, 0.0) + sign * lots
        result[quote] = result.get(quote, 0.0) - sign * lots
        return {k: round(v, 4) for k, v in result.items()}
