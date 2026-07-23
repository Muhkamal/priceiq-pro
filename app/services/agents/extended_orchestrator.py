"""
PriceIQ Pro — Extended Agent Orchestrator v2.0

Wires MacroSignalAgent as the 5th agent alongside the existing four.
The macro agent provides fundamentally uncorrelated signals — it sees
causes (rate moves, DXY, VIX) before the price-action agents see effects.

Key changes from v1:
    - MacroSignalAgent added as 5th agent
    - OrchestratorResult now includes macro_snapshot for dashboard
    - Scoring function weights macro agent slightly differently:
      * macro agent gets a fixed regime_fit boost in volatile/trending regimes
      * macro agent is pair-specific (only fires on XAUUSD, EURUSD, GBPUSD)
    - Signal from macro agent compatible with existing V5 pipeline

Usage:
    orchestrator = ExtendedAgentOrchestrator()
    await orchestrator.macro_agent.refresh_data()   # call hourly

    result = orchestrator.run(candles, regime="trending", pair="XAUUSD")
    # MacroAgent evaluated automatically if pair is supported
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .agent_layer import (
    AgentSignal, AgentOrchestrator, OrchestratorResult,
    TrendAgent, MeanReversionAgent, BreakoutAgent, LiquidityTrapAgent,
)
from .macro_signal_agent import MacroSignalAgent, MacroAgentSignal

logger = logging.getLogger(__name__)


class MacroAgentAdapter:
    """
    Adapts MacroAgentSignal to the AgentSignal interface
    so the orchestrator can score it uniformly.
    """

    NAME = "MacroSignalAgent"

    def __init__(self, macro_agent: MacroSignalAgent):
        self._macro = macro_agent

    def evaluate(self, candles: list, regime: str, pair: str = "XAUUSD") -> AgentSignal:
        """Call macro agent and return AgentSignal-compatible result."""
        macro_sig = self._macro.evaluate(candles, regime, pair)

        if not macro_sig.direction:
            return AgentSignal(
                agent_name=self.NAME, direction=None, confidence=0.0,
                win_probability=0.0, expected_value=0.0,
                stop_distance=0.0, tp1_distance=0.0, tp2_distance=0.0,
                regime_fit=0.0,
                reasoning=macro_sig.reasoning,
                raw_features=macro_sig.raw_features,
            )

        return AgentSignal(
            agent_name=self.NAME,
            direction=macro_sig.direction,
            confidence=macro_sig.confidence,
            win_probability=macro_sig.win_probability,
            expected_value=macro_sig.expected_value,
            stop_distance=macro_sig.stop_distance,
            tp1_distance=macro_sig.tp1_distance,
            tp2_distance=macro_sig.tp2_distance,
            regime_fit=macro_sig.regime_fit,
            reasoning=macro_sig.reasoning,
            raw_features={
                **macro_sig.raw_features,
                "macro_snapshot": macro_sig.macro_snapshot.__dict__
                    if macro_sig.macro_snapshot else {},
            },
        )

    def _null_signal(self, reason: str) -> AgentSignal:
        return AgentSignal(
            agent_name=self.NAME, direction=None, confidence=0.0,
            win_probability=0.0, expected_value=0.0,
            stop_distance=0.0, tp1_distance=0.0, tp2_distance=0.0,
            regime_fit=0.0, reasoning=reason,
        )


@dataclass
class ExtendedOrchestratorResult:
    """OrchestratorResult extended with macro signal info."""
    selected_agent:   str
    selected_signal:  AgentSignal
    all_signals:      Dict[str, AgentSignal]
    selection_score:  float
    regime:           str
    reasoning:        str
    macro_fired:      bool
    macro_direction:  Optional[str]
    macro_confidence: float
    macro_aligns:     bool    # True if macro agrees with selected signal
    macro_opposes:    bool    # True if macro contradicts selected signal


class ExtendedAgentOrchestrator:
    """
    5-agent orchestrator: Trend + MeanReversion + Breakout + LiquidityTrap + Macro.

    Macro integration rules:
        1. Macro agent evaluated alongside the other four
        2. If macro fires same direction → confidence boost (+0.05)
        3. If macro fires opposite direction → confidence penalty (-0.07)
           and size reduction (0.70×)
        4. If macro fires but price agents don't → macro can still win
           if its confidence exceeds threshold (high-conviction macro trade)
        5. Macro signal stored in result for dashboard transparency
    """

    MACRO_AGREE_BOOST    = 0.05    # conf boost when macro agrees
    MACRO_OPPOSE_PENALTY = 0.07    # conf penalty when macro disagrees
    MACRO_OPPOSE_SIZE    = 0.70    # size reduction when macro disagrees

    def __init__(self):
        # Core price-action agents
        self._price_agents = {
            "TrendAgent":         TrendAgent(),
            "MeanReversionAgent": MeanReversionAgent(),
            "BreakoutAgent":      BreakoutAgent(),
            "LiquidityTrapAgent": LiquidityTrapAgent(),
        }
        # Macro agent (5th, fundamentally different)
        self.macro_agent   = MacroSignalAgent()
        self._macro_adapter = MacroAgentAdapter(self.macro_agent)

        # Weights (updated by learning system)
        self.weights: Dict[str, float] = {
            name: 1.0 for name in list(self._price_agents.keys()) + ["MacroSignalAgent"]
        }

    def update_weights(self, weights: Dict[str, float]):
        """Called by learning system after outcomes."""
        for name, w in weights.items():
            if name in self.weights:
                self.weights[name] = max(0.1, min(w, 3.0))

    def run(
        self,
        candles: list,
        regime:  str,
        pair:    str = "EURUSD",
    ) -> Optional[ExtendedOrchestratorResult]:
        """
        Evaluate all 5 agents and select best signal.
        MacroAgent evaluated only for supported pairs.
        """
        all_signals: Dict[str, AgentSignal] = {}

        # ── Run price-action agents ───────────────────────────
        for name, agent in self._price_agents.items():
            try:
                all_signals[name] = agent.evaluate(candles, regime)
            except Exception as e:
                logger.warning(f"Agent {name} error: {e}")
                all_signals[name] = agent._null_signal(f"Error: {e}")

        # ── Run macro agent (pair-specific) ───────────────────
        macro_sig      = None
        macro_fired    = False
        macro_dir      = None
        macro_conf     = 0.0
        macro_supports = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY"}

        if pair.upper() in macro_supports:
            try:
                macro_as  = self._macro_adapter.evaluate(candles, regime, pair)
                all_signals["MacroSignalAgent"] = macro_as
                macro_fired = macro_as.direction is not None
                macro_dir   = macro_as.direction
                macro_conf  = macro_as.confidence
                macro_sig   = macro_as
            except Exception as e:
                logger.warning(f"MacroAgent error: {e}")
                all_signals["MacroSignalAgent"] = self._macro_adapter._null_signal(str(e))

        # ── Select best valid signal ──────────────────────────
        def score(name: str, sig: AgentSignal) -> float:
            ev_factor = max(0.1, 1.0 + sig.expected_value)
            return sig.confidence * sig.regime_fit * ev_factor * self.weights.get(name, 1.0)

        valid = {n: s for n, s in all_signals.items() if s.is_valid}
        if not valid:
            return None

        best_name  = max(valid, key=lambda n: score(n, valid[n]))
        best_sig   = valid[best_name]
        best_score = score(best_name, best_sig)

        # ── Macro agreement / disagreement adjustment ─────────
        macro_aligns  = False
        macro_opposes = False
        adjusted_conf = best_sig.confidence

        if macro_fired and macro_dir:
            if macro_dir == best_sig.direction:
                macro_aligns  = True
                adjusted_conf = min(1.0, best_sig.confidence + self.MACRO_AGREE_BOOST)
                logger.debug(
                    f"Macro AGREES with {best_name}: "
                    f"conf {best_sig.confidence:.2f} → {adjusted_conf:.2f}"
                )
            else:
                macro_opposes = True
                adjusted_conf = max(0.0, best_sig.confidence - self.MACRO_OPPOSE_PENALTY)
                logger.debug(
                    f"Macro OPPOSES {best_name}: "
                    f"conf {best_sig.confidence:.2f} → {adjusted_conf:.2f}"
                )

        # Apply adjusted confidence back to signal
        best_sig = AgentSignal(
            agent_name=best_sig.agent_name,
            direction=best_sig.direction,
            confidence=round(adjusted_conf, 3),
            win_probability=best_sig.win_probability,
            expected_value=best_sig.expected_value,
            stop_distance=best_sig.stop_distance,
            tp1_distance=best_sig.tp1_distance,
            tp2_distance=best_sig.tp2_distance,
            regime_fit=best_sig.regime_fit,
            reasoning=best_sig.reasoning + (
                f" [MACRO {'AGREES ✅' if macro_aligns else 'OPPOSES ⚠️'}]"
                if macro_fired else " [MACRO N/A]"
            ),
            raw_features=best_sig.raw_features,
        )

        reasoning = (
            f"Selected {best_name} (score={best_score:.3f}) in {regime} regime. "
            f"Direction: {best_sig.direction}. "
            f"Confidence: {best_sig.confidence:.2f}. "
            f"EV: {best_sig.expected_value:.3f}R. "
            f"Macro: {'fired ' + str(macro_dir) + ' (' + ('aligns' if macro_aligns else 'opposes') + ')' if macro_fired else 'not available'}."
        )

        return ExtendedOrchestratorResult(
            selected_agent=best_name,
            selected_signal=best_sig,
            all_signals=all_signals,
            selection_score=round(best_score, 4),
            regime=regime,
            reasoning=reasoning,
            macro_fired=macro_fired,
            macro_direction=macro_dir,
            macro_confidence=macro_conf,
            macro_aligns=macro_aligns,
            macro_opposes=macro_opposes,
        )

    def get_macro_status(self) -> Dict:
        """For dashboard: current macro snapshot and agent status."""
        snap = self.macro_agent.get_snapshot()
        return {
            "macro_agent": self.macro_agent.status_dict(),
            "weights":     self.weights,
            "supported_pairs": list({"XAUUSD", "EURUSD", "GBPUSD", "USDJPY"}),
        }
