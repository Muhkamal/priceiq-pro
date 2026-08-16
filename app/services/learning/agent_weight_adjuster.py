"""
Agent Weight Adjuster — Dynamically reduces confidence of losing agents.
Based on rolling 20-signal performance per agent.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DB_PATH = Path("agent_weights.json")
MIN_SAMPLES = 5
LOOKBACK = 20


class AgentWeightAdjuster:
    """
    Tracks per-agent win rate and applies confidence penalties/bonuses.
    """

    def __init__(self):
        self._history: Dict[str, List[Dict]] = {}  # agent -> [{outcome, pnl, time}]
        self._weights: Dict[str, float] = {}  # agent -> weight multiplier
        self._load()

    def _load(self):
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "r") as f:
                    data = json.load(f)
                    self._history = data.get("history", {})
                    self._weights = data.get("weights", {})
            except Exception:
                pass

    def _save(self):
        try:
            with open(DB_PATH, "w") as f:
                json.dump({"history": self._history, "weights": self._weights}, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Weight save failed: {e}")

    def record(self, agent: str, outcome: str, pnl_r: float):
        if agent not in self._history:
            self._history[agent] = []
        self._history[agent].append({
            "outcome": outcome,
            "pnl_r": pnl_r,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        # Trim old
        self._history[agent] = self._history[agent][-LOOKBACK:]
        self._recalculate(agent)

    def _recalculate(self, agent: str):
        hist = self._history.get(agent, [])
        if len(hist) < MIN_SAMPLES:
            self._weights[agent] = 1.0
            return

        wins = sum(1 for h in hist if h["outcome"] in ("tp1", "tp2", "win"))
        wr = wins / len(hist)
        avg_pnl = sum(h["pnl_r"] for h in hist) / len(hist)

        # Weight formula:
        # WR >= 60% and avg_pnl > 0.5R → boost to 1.10
        # WR 40-60% → neutral 1.00
        # WR < 40% or avg_pnl < -0.3R → penalty to 0.85
        # WR < 30% → heavy penalty to 0.70

        if wr >= 0.60 and avg_pnl > 0.5:
            weight = 1.10
        elif wr >= 0.40:
            weight = 1.00
        elif wr >= 0.30:
            weight = 0.85
        else:
            weight = 0.70

        old_weight = self._weights.get(agent, 1.0)
        self._weights[agent] = round(weight, 2)

        if old_weight != weight:
            logger.info(f"Agent weight adjusted: {agent} WR={wr:.0%} avgR={avg_pnl:+.2f} → weight={weight}")

        self._save()

    def get_weight(self, agent: str) -> float:
        return self._weights.get(agent, 1.0)

    def apply(self, agent: str, raw_confidence: float) -> float:
        weight = self.get_weight(agent)
        return min(1.0, max(0.0, raw_confidence * weight))

    def get_summary(self) -> Dict:
        return {
            agent: {
                "weight": self._weights.get(agent, 1.0),
                "samples": len(self._history.get(agent, [])),
                "recent_wr": self._calc_wr(self._history.get(agent, [])),
            }
            for agent in self._weights.keys()
        }

    def _calc_wr(self, hist: List[Dict]) -> float:
        if not hist:
            return 0.0
        wins = sum(1 for h in hist if h["outcome"] in ("tp1", "tp2", "win"))
        return round(wins / len(hist), 2)


agent_adjuster = AgentWeightAdjuster()
