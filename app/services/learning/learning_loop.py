"""
PriceIQ Pro — Adaptive Learning System v1.0

Closes the feedback loop:
    trade outcome → weight update → better agent selection → better trades

Components:
    TradeOutcomeStore   — persists outcomes in-memory (+ optional JSON file)
    AdaptiveLearner     — updates agent weights and pair-level confidence thresholds
    LearningLoop        — orchestrates the whole cycle; call update() after each trade

Usage:
    loop = LearningLoop()

    # After a trade closes:
    loop.update(
        pair="XAUUSD",
        agent_name="TrendAgent",
        direction="buy",
        entry=1920.0,
        exit=1935.0,
        stop=1910.0,
        tp1=1935.0,
        outcome="win",          # "win" | "loss" | "timeout" | "breakeven"
        r_multiple=1.5,         # actual R gained/lost
    )

    # Get updated weights for orchestrator:
    weights = loop.get_agent_weights()
    orchestrator.update_weights(weights)

    # Get pair-level confidence threshold:
    threshold = loop.get_confidence_threshold("XAUUSD")
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PERSISTENCE_PATH = "learning_state.json"
_MIN_TRADES_TO_ADAPT = 10       # Minimum trades before weights change
_MAX_WEIGHT          = 3.0
_MIN_WEIGHT          = 0.1
_WEIGHT_DECAY        = 0.002    # Small decay per cycle keeps weights fresh
_WIN_BOOST           = 0.06     # Weight increase per win
_LOSS_PENALTY        = 0.04     # Weight decrease per loss
_CONFIDENCE_FLOOR    = 0.45     # Lowest threshold any pair can reach
_CONFIDENCE_CEILING  = 0.80     # Highest threshold any pair can reach


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


# ============================================================
# TRADE OUTCOME RECORD
# ============================================================

@dataclass
class TradeOutcome:
    timestamp:   str
    pair:        str
    agent_name:  str
    direction:   str
    entry:       float
    exit_price:  float
    stop:        float
    tp1:         float
    outcome:     str        # "win" | "loss" | "timeout" | "breakeven"
    r_multiple:  float      # +1.5 = won 1.5R; -1.0 = lost 1R
    regime:      str = "unknown"
    confidence:  float = 0.0
    session:     str = "unknown"


# ============================================================
# TRADE OUTCOME STORE
# ============================================================

class TradeOutcomeStore:
    """In-memory store with optional JSON persistence."""

    def __init__(self, path: str = _PERSISTENCE_PATH):
        self._path    = path
        self._records: List[TradeOutcome] = []
        self._load()

    def add(self, outcome: TradeOutcome):
        self._records.append(outcome)
        self._save()

    def all(self) -> List[TradeOutcome]:
        return list(self._records)

    def by_pair(self, pair: str) -> List[TradeOutcome]:
        return [r for r in self._records if r.pair.upper() == pair.upper()]

    def by_agent(self, agent_name: str) -> List[TradeOutcome]:
        return [r for r in self._records if r.agent_name == agent_name]

    def recent(self, n: int = 50) -> List[TradeOutcome]:
        return self._records[-n:]

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump([asdict(r) for r in self._records], f, indent=2)
        except Exception as e:
            logger.warning(f"LearningStore save failed: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._records = [TradeOutcome(**d) for d in data]
            logger.info(f"Loaded {len(self._records)} trade outcomes from {self._path}")
        except Exception as e:
            logger.warning(f"LearningStore load failed: {e}")


# ============================================================
# ADAPTIVE LEARNER
# ============================================================

class AdaptiveLearner:
    """
    Updates agent weights and per-pair confidence thresholds from trade outcomes.

    Weight logic:
        - WIN  → weight += WIN_BOOST
        - LOSS → weight -= LOSS_PENALTY
        - All weights decay slightly each cycle (keeps system responsive to regime changes)
        - Weights clamped to [MIN_WEIGHT, MAX_WEIGHT]

    Confidence threshold logic:
        - If pair win rate drops below 45% over last 20 trades → raise threshold (+0.03)
        - If pair win rate rises above 60% over last 20 trades → lower threshold (-0.02)
        - Threshold clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CEILING]
    """

    def __init__(self):
        self._agent_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._pair_thresholds: Dict[str, float] = defaultdict(lambda: 0.45)

    def update_from_outcome(self, outcome: TradeOutcome, all_records: List[TradeOutcome]):
        """Update weights and thresholds based on a new trade outcome."""
        name = outcome.agent_name

        # ── Agent weight update ──────────────────────────────
        if outcome.outcome == "win":
            self._agent_weights[name] = min(
                _MAX_WEIGHT, self._agent_weights[name] + _WIN_BOOST
            )
        elif outcome.outcome == "loss":
            self._agent_weights[name] = max(
                _MIN_WEIGHT, self._agent_weights[name] - _LOSS_PENALTY
            )
        # Decay all weights slightly each update
        for k in self._agent_weights:
            self._agent_weights[k] = max(
                _MIN_WEIGHT,
                self._agent_weights[k] * (1 - _WEIGHT_DECAY)
            )

        # ── Pair threshold update ────────────────────────────
        pair_records = [r for r in all_records if r.pair.upper() == outcome.pair.upper()]
        if len(pair_records) >= _MIN_TRADES_TO_ADAPT:
            recent_20 = pair_records[-20:]
            wins     = sum(1 for r in recent_20 if r.outcome == "win")
            win_rate = _safe_div(wins, len(recent_20))

            curr_threshold = self._pair_thresholds[outcome.pair.upper()]
            if win_rate < 0.45:
                new_threshold = min(_CONFIDENCE_CEILING, curr_threshold + 0.03)
                if new_threshold != curr_threshold:
                    logger.info(
                        f"{outcome.pair}: win rate {win_rate:.1%} < 45% → "
                        f"raising confidence threshold {curr_threshold:.2f} → {new_threshold:.2f}"
                    )
                    self._pair_thresholds[outcome.pair.upper()] = new_threshold

            elif win_rate > 0.60:
                new_threshold = max(_CONFIDENCE_FLOOR, curr_threshold - 0.02)
                if new_threshold != curr_threshold:
                    logger.info(
                        f"{outcome.pair}: win rate {win_rate:.1%} > 60% → "
                        f"lowering confidence threshold {curr_threshold:.2f} → {new_threshold:.2f}"
                    )
                    self._pair_thresholds[outcome.pair.upper()] = new_threshold

    def get_agent_weights(self) -> Dict[str, float]:
        return dict(self._agent_weights)

    def get_confidence_threshold(self, pair: str) -> float:
        return self._pair_thresholds.get(pair.upper(), 0.45)

    def get_pair_stats(self, pair: str, records: List[TradeOutcome]) -> Dict[str, Any]:
        pair_recs = [r for r in records if r.pair.upper() == pair.upper()]
        if not pair_recs:
            return {"trades": 0, "win_rate": None, "expectancy": None,
                    "threshold": self.get_confidence_threshold(pair)}
        wins    = [r for r in pair_recs if r.outcome == "win"]
        r_mults = [r.r_multiple for r in pair_recs]
        return {
            "trades":      len(pair_recs),
            "wins":        len(wins),
            "losses":      len(pair_recs) - len(wins),
            "win_rate":    round(_safe_div(len(wins), len(pair_recs)), 3),
            "expectancy":  round(float(np.mean(r_mults)), 3) if r_mults else None,
            "avg_r":       round(float(np.mean(r_mults)), 3) if r_mults else None,
            "threshold":   self.get_confidence_threshold(pair),
            "agent_weights": self.get_agent_weights(),
        }

    def get_agent_stats(self, records: List[TradeOutcome]) -> Dict[str, Any]:
        stats = {}
        for agent in set(r.agent_name for r in records):
            agent_recs = [r for r in records if r.agent_name == agent]
            wins       = [r for r in agent_recs if r.outcome == "win"]
            r_mults    = [r.r_multiple for r in agent_recs]
            stats[agent] = {
                "trades":   len(agent_recs),
                "win_rate": round(_safe_div(len(wins), len(agent_recs)), 3),
                "expectancy": round(float(np.mean(r_mults)), 3) if r_mults else 0,
                "weight":   round(self._agent_weights.get(agent, 1.0), 4),
            }
        return stats


# ============================================================
# FULL LEARNING LOOP
# ============================================================

class LearningLoop:
    """
    Top-level facade that orchestrates the full feedback cycle.

    Call update() from the scheduler after every trade closes.
    Then call get_agent_weights() and pass to orchestrator.update_weights().
    """

    def __init__(self, persistence_path: str = _PERSISTENCE_PATH):
        self.store   = TradeOutcomeStore(persistence_path)
        self.learner = AdaptiveLearner()

        # Bootstrap learner from historical records
        self._bootstrap()

    def _bootstrap(self):
        """Replay all stored outcomes to reconstruct weights on startup."""
        for record in self.store.all():
            self.learner.update_from_outcome(record, self.store.all())
        if self.store.all():
            logger.info(
                f"LearningLoop bootstrapped from {len(self.store.all())} historical outcomes. "
                f"Agent weights: {self.learner.get_agent_weights()}"
            )

    def update(
        self,
        pair:       str,
        agent_name: str,
        direction:  str,
        entry:      float,
        exit_price: float,
        stop:       float,
        tp1:        float,
        outcome:    str,
        r_multiple: float,
        regime:     str = "unknown",
        confidence: float = 0.0,
        session:    str = "unknown",
    ):
        """Record a completed trade and update all learning state."""
        record = TradeOutcome(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            agent_name=agent_name,
            direction=direction,
            entry=entry,
            exit_price=exit_price,
            stop=stop,
            tp1=tp1,
            outcome=outcome,
            r_multiple=r_multiple,
            regime=regime,
            confidence=confidence,
            session=session,
        )
        self.store.add(record)
        self.learner.update_from_outcome(record, self.store.all())

        logger.info(
            f"LearningLoop: {pair} {agent_name} {outcome} "
            f"{r_multiple:+.2f}R → weights updated"
        )

    def get_agent_weights(self) -> Dict[str, float]:
        return self.learner.get_agent_weights()

    def get_confidence_threshold(self, pair: str) -> float:
        return self.learner.get_confidence_threshold(pair)

    def get_full_stats(self) -> Dict[str, Any]:
        all_records = self.store.all()
        return {
            "total_trades":  len(all_records),
            "agent_stats":   self.learner.get_agent_stats(all_records),
            "agent_weights": self.learner.get_agent_weights(),
            "pairs": {
                pair: self.learner.get_pair_stats(pair, all_records)
                for pair in set(r.pair for r in all_records)
            },
        }
