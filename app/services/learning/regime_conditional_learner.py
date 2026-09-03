"""
PriceIQ Pro — Regime-Conditional Learning v1.1 (CRYPTO/GOLD BOOST)

Replaces the global weight update in learning_loop.py with per-regime weights.

Solution:
    Store a weight table:  agent × regime → weight
    Update only the cell matching the trade's regime.
    Orchestrator uses regime-specific weights at selection time.

Also implements Thompson Sampling for exploration:
    Instead of always picking the highest-weight agent (exploitation),
    sample from a Beta distribution to occasionally explore lower-weight
    agents — prevents premature convergence.

v1.1 Update:
    ✅ Added `pair` argument to thompson_sample() and get_weights_for_regime()
    ✅ Boosts Breakout/Trend agents for BTC/Gold in volatile/trending regimes
    ✅ Boosts MeanReversion for Gold in ranging regimes
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AGENTS  = ["TrendAgent", "MeanReversionAgent", "BreakoutAgent", "LiquidityTrapAgent"]
REGIMES = ["trending", "ranging", "volatile"]

# Beta distribution priors (uninformative: 1 win, 1 loss to start)
PRIOR_ALPHA = 1.0   # pseudo-win count
PRIOR_BETA  = 1.0   # pseudo-loss count

WEIGHT_FLOOR   = 0.10
WEIGHT_CEILING = 4.0
DECAY_PER_UPDATE = 0.001   # small decay keeps weights current


@dataclass
class RegimeCell:
    """Alpha/beta counts for one agent × regime cell."""
    agent:  str
    regime: str
    wins:   float = PRIOR_ALPHA
    losses: float = PRIOR_BETA

    @property
    def weight(self) -> float:
        """Point estimate: win rate scaled to [FLOOR, CEILING]."""
        wr = self.wins / (self.wins + self.losses)
        return round(WEIGHT_FLOOR + wr * (WEIGHT_CEILING - WEIGHT_FLOOR), 4)

    def thompson_sample(self) -> float:
        """Sample from Beta(alpha, beta) — exploration vs exploitation."""
        return float(np.random.beta(self.wins, self.losses))

    def update(self, outcome: str):
        """Update counts for a win or loss."""
        if outcome == "win":
            self.wins  += 1.0
        elif outcome in ("loss", "timeout"):
            self.losses += 1.0
        # breakeven: no update

    def decay(self):
        """Small decay per cycle — makes recent data more influential."""
        self.wins   = max(PRIOR_ALPHA, self.wins   * (1 - DECAY_PER_UPDATE))
        self.losses = max(PRIOR_BETA,  self.losses * (1 - DECAY_PER_UPDATE))

    def to_dict(self) -> Dict:
        return {"agent": self.agent, "regime": self.regime,
                "wins": round(self.wins, 4), "losses": round(self.losses, 4),
                "weight": self.weight}


class RegimeConditionalLearner:
    """
    Per-regime, per-agent weight table using Beta distribution counts.
    """

    def __init__(self, persistence_path: str = "regime_weights.json"):
        self._path  = persistence_path
        self._table: Dict[str, Dict[str, RegimeCell]] = {}   # agent → regime → cell

        # Initialise all cells with priors
        for agent in AGENTS:
            self._table[agent] = {}
            for regime in REGIMES:
                self._table[agent][regime] = RegimeCell(agent=agent, regime=regime)

        self._load()

    # ── Update ───────────────────────────────────────────────

    def update(self, agent: str, regime: str, outcome: str):
        """
        Update the specific agent × regime cell.
        outcome: "win" | "loss" | "timeout" | "breakeven"
        """
        if agent not in self._table:
            self._table[agent] = {r: RegimeCell(agent=agent, regime=r) for r in REGIMES}
        if regime not in self._table[agent]:
            self._table[agent][regime] = RegimeCell(agent=agent, regime=regime)

        self._table[agent][regime].update(outcome)

        # Decay all other cells slightly on each update
        for a in self._table:
            for r in self._table[a]:
                if not (a == agent and r == regime):
                    self._table[a][r].decay()

        self._save()
        logger.debug(
            f"RegimeConditional: {agent}/{regime} {outcome} → "
            f"w={self._table[agent][regime].weight:.3f} "
            f"({self._table[agent][regime].wins:.1f}W/{self._table[agent][regime].losses:.1f}L)"
        )

    # ── Query ────────────────────────────────────────────────

    def get_weights_for_regime(self, regime: str, pair: str = None) -> Dict[str, float]:
        """
        Point-estimate weights for the given regime.
        Use for deterministic selection (exploitation only).
        """
        weights = {
            agent: self._table[agent].get(
                regime, RegimeCell(agent=agent, regime=regime)
            ).weight
            for agent in self._table
        }
        
        # ═══ NEW: Pair-specific overrides for high-volatility assets ═══
        if pair and pair.upper() in ("BTCUSD", "XAUUSD", "ETHUSD"):
            if regime == "volatile":
                weights["BreakoutAgent"] = max(weights.get("BreakoutAgent", 0.10), 3.50)
                weights["TrendAgent"]    = max(weights.get("TrendAgent", 0.10), 3.00)
            elif regime == "trending":
                weights["TrendAgent"] = max(weights.get("TrendAgent", 0.10), 3.50)
            elif regime == "ranging" and pair.upper() in ("XAUUSD"):
                weights["MeanReversionAgent"] = max(weights.get("MeanReversionAgent", 0.10), 3.00)
                
        return weights

    def thompson_sample(self, regime: str, pair: str = None) -> Dict[str, float]:
        """
        Thompson-sampled weights for the given regime.
        Use for agent selection — explores uncertain agents.
        Returns scaled samples [FLOOR, CEILING].
        """
        samples = {}
        for agent in self._table:
            cell   = self._table[agent].get(regime, RegimeCell(agent=agent, regime=regime))
            raw    = cell.thompson_sample()
            scaled = WEIGHT_FLOOR + raw * (WEIGHT_CEILING - WEIGHT_FLOOR)
            samples[agent] = round(scaled, 4)
            
        # ═══ NEW: Pair-specific overrides for high-volatility assets ═══
        if pair and pair.upper() in ("BTCUSD", "XAUUSD", "ETHUSD"):
            if regime == "volatile":
                # Boost Breakout and Trend agents for crypto/gold in volatile markets
                samples["BreakoutAgent"] = max(samples.get("BreakoutAgent", 0.10), 3.50)
                samples["TrendAgent"]    = max(samples.get("TrendAgent", 0.10), 3.00)
            elif regime == "trending":
                # Boost Trend agent
                samples["TrendAgent"] = max(samples.get("TrendAgent", 0.10), 3.50)
            elif regime == "ranging" and pair.upper() in ("XAUUSD"):
                # Gold/Silver mean-reverts well in ranging markets
                samples["MeanReversionAgent"] = max(samples.get("MeanReversionAgent", 0.10), 3.00)
                
        return samples

    def get_full_table(self) -> Dict[str, Any]:
        """Full weight table for dashboard / debugging."""
        return {
            agent: {
                regime: self._table[agent][regime].to_dict()
                for regime in REGIMES
            }
            for agent in self._table
        }

    def get_best_agent_per_regime(self) -> Dict[str, Tuple[str, float]]:
        """For each regime, which agent is currently winning?"""
        result = {}
        for regime in REGIMES:
            weights = self.get_weights_for_regime(regime)
            best    = max(weights, key=weights.get)
            result[regime] = (best, weights[best])
        return result

    def regime_fit_summary(self) -> str:
        """Human-readable weight table."""
        lines = ["Agent Weights by Regime:"]
        lines.append(f"{'Agent':<25} {'trending':>10} {'ranging':>10} {'volatile':>10}")
        lines.append("-" * 57)
        for agent in AGENTS:
            row = f"{agent:<25}"
            for regime in REGIMES:
                w = self._table[agent].get(regime, RegimeCell(agent=agent, regime=regime)).weight
                row += f" {w:>10.3f}"
            lines.append(row)
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────

    def _save(self):
        try:
            data = {}
            for agent in self._table:
                data[agent] = {}
                for regime, cell in self._table[agent].items():
                    data[agent][regime] = {"wins": cell.wins, "losses": cell.losses}
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"RegimeConditional save failed: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for agent, regimes in data.items():
                if agent not in self._table:
                    self._table[agent] = {}
                for regime, counts in regimes.items():
                    self._table[agent][regime] = RegimeCell(
                        agent=agent, regime=regime,
                        wins=counts.get("wins", PRIOR_ALPHA),
                        losses=counts.get("losses", PRIOR_BETA),
                    )
            logger.info(f"RegimeConditional: loaded weights from {self._path}")
            logger.info(f"\n{self.regime_fit_summary()}")
        except Exception as e:
            logger.warning(f"RegimeConditional load failed: {e}")
