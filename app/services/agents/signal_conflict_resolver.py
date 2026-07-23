"""
PriceIQ Pro — Signal Conflict Resolver v1.0

Measures how much all agents AGREE on a signal before it fires.
High disagreement = reduce size or block. Unanimous = boost confidence.

Conflict types:
    DIRECTION conflict:  agents disagree on buy vs sell (worst case)
    MAGNITUDE conflict:  agents agree direction but have very different confidence
    REGIME conflict:     selected agent's regime_fit is low vs other agents

Usage:
    resolver = SignalConflictResolver()

    result = resolver.evaluate(orchestrator_result)
    # → ConflictReport(
    #       conflict_score=0.72,    # 0=unanimous, 1=full conflict
    #       direction_split={"buy": 2, "sell": 1, "none": 1},
    #       size_multiplier=0.60,   # reduce size due to conflict
    #       confidence_adj=-0.05,   # lower confidence
    #       should_block=False,
    #       reason="2 agents disagree on direction"
    #   )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Thresholds
CONFLICT_BLOCK     = 0.85   # conflict score above this → block
CONFLICT_REDUCE    = 0.50   # above this → reduce size
MIN_AGREE_FRACTION = 0.50   # at least 50% of valid agents must agree on direction


@dataclass
class ConflictReport:
    conflict_score:   float     # 0 = unanimous, 1 = maximum conflict
    direction_split:  Dict[str, int]   # "buy" → N, "sell" → M, "none" → K
    agree_fraction:   float     # fraction of valid agents agreeing with selected
    confidence_adj:   float     # adjustment to add to signal confidence
    size_multiplier:  float     # 1.0 = full, 0.5 = half, 0.0 = block
    should_block:     bool
    reason:           str


class SignalConflictResolver:
    """
    Evaluates agreement across all agents before a signal fires.

    Scoring:
        - Direction entropy: measure of buy/sell disagreement
        - Confidence spread: std dev of agent confidences
        - Regime fit gap: selected agent's fit vs best alternative

    The conflict_score combines all three into 0–1.
    """

    def evaluate(self, orchestrator_result) -> ConflictReport:
        """
        Args:
            orchestrator_result: OrchestratorResult from agent_layer.py
        Returns:
            ConflictReport with sizing/confidence adjustments
        """
        all_signals = orchestrator_result.all_signals
        selected    = orchestrator_result.selected_signal
        selected_dir = selected.direction

        # ── Direction split ───────────────────────────────────
        direction_count = {"buy": 0, "sell": 0, "none": 0}
        valid_confidences = []
        for sig in all_signals.values():
            if sig.direction == "buy":
                direction_count["buy"] += 1
            elif sig.direction == "sell":
                direction_count["sell"] += 1
            else:
                direction_count["none"] += 1
            if sig.direction is not None:
                valid_confidences.append(sig.confidence)

        n_valid = direction_count["buy"] + direction_count["sell"]
        n_agree = direction_count.get(selected_dir, 0)
        agree_frac = n_agree / max(n_valid, 1)

        # ── Entropy score (direction disagreement) ────────────
        if n_valid > 0:
            p_buy  = direction_count["buy"]  / n_valid
            p_sell = direction_count["sell"] / n_valid
            # Binary entropy (0=pure, 1=max disagreement)
            def _entropy(p):
                if p <= 0 or p >= 1:
                    return 0.0
                return -p * np.log2(p) - (1-p) * np.log2(1-p)
            direction_entropy = _entropy(p_buy)
        else:
            direction_entropy = 1.0

        # ── Confidence spread ─────────────────────────────────
        conf_spread = np.std(valid_confidences) if len(valid_confidences) > 1 else 0.0
        conf_spread_normalised = min(1.0, conf_spread / 0.3)   # 0.3 std = max conflict

        # ── Regime fit gap ────────────────────────────────────
        selected_fit  = selected.regime_fit
        other_fits    = [
            sig.regime_fit for name, sig in all_signals.items()
            if sig.direction is not None and sig.direction == selected_dir
        ]
        avg_other_fit = np.mean(other_fits) if other_fits else selected_fit
        fit_gap       = max(0.0, avg_other_fit - selected_fit)   # 0 = selected is best

        # ── Combined conflict score ───────────────────────────
        conflict = (
            0.50 * direction_entropy +
            0.30 * conf_spread_normalised +
            0.20 * fit_gap
        )
        conflict = round(float(np.clip(conflict, 0.0, 1.0)), 4)

        # ── Decisions ─────────────────────────────────────────
        should_block = conflict >= CONFLICT_BLOCK or agree_frac < MIN_AGREE_FRACTION

        if should_block:
            size_mult = 0.0
            conf_adj  = -0.10
            reason    = (
                f"BLOCKED: Conflict score {conflict:.2f} "
                f"(direction split {direction_count}, agree={agree_frac:.0%})"
            )
        elif conflict >= CONFLICT_REDUCE:
            size_mult = 0.60
            conf_adj  = -0.05
            reason    = (
                f"REDUCED: Conflict score {conflict:.2f} — "
                f"{n_agree}/{n_valid} agents agree on {selected_dir}"
            )
        elif agree_frac >= 0.75:
            # Strong consensus → boost
            size_mult = 1.0
            conf_adj  = +0.03
            reason    = f"CONSENSUS: {n_agree}/{n_valid} agents agree — confidence boosted"
        else:
            size_mult = 1.0
            conf_adj  = 0.0
            reason    = f"NORMAL: Conflict score {conflict:.2f} — within acceptable range"

        return ConflictReport(
            conflict_score=conflict,
            direction_split=direction_count,
            agree_fraction=round(agree_frac, 3),
            confidence_adj=round(conf_adj, 3),
            size_multiplier=size_mult,
            should_block=should_block,
            reason=reason,
        )

    def describe(self, report: ConflictReport) -> str:
        return (
            f"Conflict: {report.conflict_score:.2f} | "
            f"Split: {report.direction_split} | "
            f"Agreement: {report.agree_fraction:.0%} | "
            f"Size: {report.size_multiplier:.0%} | "
            f"{report.reason}"
        )
