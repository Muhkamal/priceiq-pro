"""
PriceIQ Pro — Win Probability Calibrator v1.0

Replaces hardcoded win probabilities in each agent (0.58, 0.62, etc.)
with empirical estimates derived from actual trade outcomes.

Problem:
    TrendAgent says win_probability = 0.58 — that's a guess.
    Real win rate depends on: agent, regime, session, confidence bin.
    After 50 trades we have enough data to compute real estimates.

Solution:
    Stratified probability table: agent × regime × confidence_bin → win_rate
    Falls back to hardcoded prior if insufficient data in a cell.

Usage:
    calibrator = WinProbabilityCalibrator()

    # After every trade:
    calibrator.record(
        agent="TrendAgent",
        regime="trending",
        confidence=0.72,
        session="london_newyork",
        outcome="win",
    )

    # At signal time (replaces hardcoded value):
    win_prob = calibrator.get_win_probability(
        agent="TrendAgent",
        regime="trending",
        confidence=0.72,
        session="london_newyork",
    )
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Prior win rates (hardcoded fallback when N < MIN_SAMPLES)
AGENT_PRIORS: Dict[str, float] = {
    "TrendAgent":          0.54,
    "MeanReversionAgent":  0.58,
    "BreakoutAgent":       0.52,
    "LiquidityTrapAgent":  0.58,
    "MAEFormula":          0.55,
}
DEFAULT_PRIOR = 0.52

# Confidence bins: [0.40, 0.55, 0.65, 0.75, 0.85, 1.00]
CONF_BINS = [0.40, 0.55, 0.65, 0.75, 0.85, 1.00]

MIN_SAMPLES_CELL   = 10    # minimum trades in a cell to use empirical estimate
MIN_SAMPLES_AGENT  = 5     # minimum for agent-level estimate


def _conf_bin(confidence: float) -> str:
    for i in range(len(CONF_BINS) - 1):
        if CONF_BINS[i] <= confidence < CONF_BINS[i + 1]:
            return f"{CONF_BINS[i]:.2f}-{CONF_BINS[i+1]:.2f}"
    return f"{CONF_BINS[-2]:.2f}-{CONF_BINS[-1]:.2f}"


def _win_rate(outcomes: List[str]) -> float:
    if not outcomes:
        return DEFAULT_PRIOR
    wins = sum(1 for o in outcomes if o == "win")
    return wins / len(outcomes)


class WinProbabilityCalibrator:
    """
    Stratified empirical win rate table:
        agent × regime × confidence_bin → [outcomes]

    Falls back through:
        full cell → agent+regime → agent only → prior
    """

    def __init__(self, persistence_path: str = "win_prob_calibrator.json"):
        self._path  = persistence_path
        # agent → regime → conf_bin → list of "win"/"loss"
        self._data: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self._load()

    def record(
        self,
        agent:      str,
        regime:     str,
        confidence: float,
        outcome:    str,         # "win" | "loss" | "timeout"
        session:    str = "",    # stored but not currently stratified
    ):
        """Record a completed trade outcome."""
        if outcome not in ("win", "loss", "timeout"):
            return
        bin_key = _conf_bin(confidence)
        self._data[agent][regime][bin_key].append(outcome)
        self._save()

    def get_win_probability(
        self,
        agent:      str,
        regime:     str,
        confidence: float,
        session:    str = "",
    ) -> float:
        """
        Return calibrated win probability for a signal.
        Falls back gracefully through stratification levels.
        """
        bin_key = _conf_bin(confidence)

        # Level 1: full cell
        cell = self._data.get(agent, {}).get(regime, {}).get(bin_key, [])
        if len(cell) >= MIN_SAMPLES_CELL:
            wr = _win_rate(cell)
            logger.debug(f"WinProb {agent}/{regime}/{bin_key}: empirical {wr:.3f} (n={len(cell)})")
            return round(wr, 4)

        # Level 2: agent + regime (collapse confidence bins)
        agent_regime = []
        for outcomes in self._data.get(agent, {}).get(regime, {}).values():
            agent_regime.extend(outcomes)
        if len(agent_regime) >= MIN_SAMPLES_AGENT:
            wr = _win_rate(agent_regime)
            logger.debug(f"WinProb {agent}/{regime} (collapsed): {wr:.3f} (n={len(agent_regime)})")
            return round(wr, 4)

        # Level 3: agent only
        agent_all = []
        for regime_data in self._data.get(agent, {}).values():
            for outcomes in regime_data.values():
                agent_all.extend(outcomes)
        if len(agent_all) >= MIN_SAMPLES_AGENT:
            wr = _win_rate(agent_all)
            logger.debug(f"WinProb {agent} (all regimes): {wr:.3f} (n={len(agent_all)})")
            return round(wr, 4)

        # Level 4: prior
        prior = AGENT_PRIORS.get(agent, DEFAULT_PRIOR)
        logger.debug(f"WinProb {agent}: using prior {prior:.3f}")
        return prior

    def calibration_table(self) -> Dict:
        """Full table of empirical win rates — for monitoring dashboard."""
        table = {}
        for agent in self._data:
            table[agent] = {}
            for regime in self._data[agent]:
                table[agent][regime] = {}
                for bin_key, outcomes in self._data[agent][regime].items():
                    n  = len(outcomes)
                    wr = _win_rate(outcomes) if n > 0 else None
                    table[agent][regime][bin_key] = {
                        "n": n, "win_rate": round(wr, 3) if wr else None
                    }
        return table

    def sample_sizes(self) -> Dict[str, int]:
        """Total samples per agent."""
        result = {}
        for agent in self._data:
            total = sum(
                len(outcomes)
                for regime_data in self._data[agent].values()
                for outcomes in regime_data.values()
            )
            result[agent] = total
        return result

    def _save(self):
        try:
            with open(self._path, "w") as f:
                # Convert defaultdict to regular dict for JSON
                data = {
                    agent: {
                        regime: dict(bins)
                        for regime, bins in regimes.items()
                    }
                    for agent, regimes in self._data.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"WinProbCalibrator save failed: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            for agent, regimes in raw.items():
                for regime, bins in regimes.items():
                    for bin_key, outcomes in bins.items():
                        self._data[agent][regime][bin_key] = outcomes
            total = sum(self.sample_sizes().values())
            logger.info(f"WinProbCalibrator loaded: {total} total samples")
        except Exception as e:
            logger.warning(f"WinProbCalibrator load failed: {e}")
