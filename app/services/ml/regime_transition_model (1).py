"""
PriceIQ Pro — Regime Transition Model v1.0

Predicts regime shifts BEFORE they happen using a Markov chain.

Problem with current regime detection:
    RegimeClassifier tells you: "Right now = trending (72%)"
    It does NOT tell you: "In 4 bars, there's a 35% chance we shift to volatile"

Solution — Markov Transition Matrix:
    Track historical regime sequences.
    Build empirical transition probabilities:
        P(trending → volatile)   = 0.12
        P(trending → ranging)    = 0.18
        P(trending → trending)   = 0.70
    Use this to pre-emptively reduce position size when a shift is likely.

Usage:
    model = RegimeTransitionModel()

    # Feed regime labels as they're classified:
    model.update("trending")
    model.update("trending")
    model.update("ranging")

    # Before a signal:
    probs = model.next_state_probs("trending")
    # → {"trending": 0.70, "ranging": 0.18, "volatile": 0.12}

    shift_risk = model.shift_risk("trending")
    # → 0.30  (probability we leave trending next bar)

    if shift_risk > 0.35:
        reduce_size(0.5)
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ["trending", "ranging", "volatile"]
SMOOTH_ALPHA = 0.5   # Laplace smoothing to avoid zero probabilities


class RegimeTransitionModel:
    """
    Empirical Markov chain for regime transitions.
    Learns from live classified regimes over time.
    """

    def __init__(self, persistence_path: str = "regime_transitions.json"):
        self._path = persistence_path
        # transition_counts[from_regime][to_regime] = count
        self._counts: Dict[str, Dict[str, float]] = {
            r: {r2: SMOOTH_ALPHA for r2 in REGIMES} for r in REGIMES
        }
        self._sequence: List[str] = []   # rolling history
        self._last_regime: Optional[str] = None
        self._load()

    def update(self, regime: str):
        """Feed the latest classified regime. Call every bar from orchestrator."""
        if regime not in REGIMES:
            return
        if self._last_regime and self._last_regime in REGIMES:
            self._counts[self._last_regime][regime] += 1.0
        self._sequence.append(regime)
        if len(self._sequence) > 500:
            self._sequence = self._sequence[-500:]
        self._last_regime = regime
        self._save()

    def next_state_probs(self, current_regime: str) -> Dict[str, float]:
        """
        Probability of each next regime given current regime.
        Returns dict: {regime → probability}
        """
        if current_regime not in REGIMES:
            return {r: 1/3 for r in REGIMES}

        row    = self._counts[current_regime]
        total  = sum(row.values())
        return {r: round(row[r] / total, 4) for r in REGIMES}

    def shift_risk(self, current_regime: str) -> float:
        """
        Probability of transitioning AWAY from current regime next bar.
        High shift risk → consider reducing size.
        """
        probs = self.next_state_probs(current_regime)
        return round(1.0 - probs.get(current_regime, 0.33), 4)

    def volatility_surge_prob(self, current_regime: str) -> float:
        """Probability of transitioning to volatile next bar."""
        probs = self.next_state_probs(current_regime)
        return probs.get("volatile", 0.0)

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Full transition probability matrix."""
        return {r: self.next_state_probs(r) for r in REGIMES}

    def size_multiplier(self, current_regime: str) -> float:
        """
        Position size multiplier based on shift risk.
        High regime uncertainty → smaller position.
        1.0 = full size, 0.5 = half, 0.25 = quarter
        """
        risk = self.shift_risk(current_regime)
        if risk >= 0.50:
            return 0.50    # very unstable regime
        if risk >= 0.35:
            return 0.75    # elevated uncertainty
        return 1.0

    def regime_streak(self) -> Tuple[str, int]:
        """How many consecutive bars in the current regime?"""
        if not self._sequence:
            return ("unknown", 0)
        current = self._sequence[-1]
        streak  = 0
        for r in reversed(self._sequence):
            if r == current:
                streak += 1
            else:
                break
        return current, streak

    def summary(self) -> str:
        matrix  = self.get_transition_matrix()
        streak_r, streak_n = self.regime_streak()
        lines = [
            "Regime Transition Matrix:",
            f"{'':20} {'→trending':>12} {'→ranging':>12} {'→volatile':>12}",
            "-" * 58,
        ]
        for from_r in REGIMES:
            row = matrix[from_r]
            lines.append(
                f"{from_r + ' →':<20} "
                f"{row.get('trending', 0):>12.1%} "
                f"{row.get('ranging', 0):>12.1%} "
                f"{row.get('volatile', 0):>12.1%}"
            )
        lines.append(f"\nCurrent streak: {streak_r} × {streak_n} bars")
        lines.append(f"Shift risk (next bar): {self.shift_risk(streak_r):.1%}")
        return "\n".join(lines)

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump({"counts": self._counts, "sequence": self._sequence[-100:]}, f)
        except Exception as e:
            logger.debug(f"RegimeTransition save failed: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._counts   = data.get("counts",   self._counts)
            self._sequence = data.get("sequence", [])
            if self._sequence:
                self._last_regime = self._sequence[-1]
            logger.info(f"RegimeTransition loaded ({len(self._sequence)} history points)")
        except Exception as e:
            logger.debug(f"RegimeTransition load failed: {e}")
