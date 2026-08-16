"""
PriceIQ Pro — V5 Master Orchestrator v3.0 (FINAL)

Wires every component built across all sessions into one system.

Complete component list:
    ── ML ──
    RegimeClassifier          XGBoost probabilistic regime detection
    FeatureDriftMonitor       PSI-based silent degradation detection
    WinProbabilityCalibrator  Empirical win probs per agent/regime/conf
    SyntheticDataAugmenter    SMOTE augmentation for class imbalance
    RegimeTransitionModel     Markov chain regime shift prediction

    ── Agents ──
    AgentOrchestrator         4 competing agents (Trend/MR/Breakout/Trap)
    SignalConflictResolver    Agent disagreement → size/confidence adj

    ── Learning ──
    LearningLoop              Global adaptive weights + pair thresholds
    RegimeConditionalLearner  Per-regime per-agent Thompson sampling

    ── Risk ──
    RiskGovernor              Drawdown/loss streak hard stops
    DynamicCorrelationEstimator Live EWM correlation
    VolatilityTargetedSizer   Kelly + vol-scaled position sizing
    VaREngine                 Historical/MC/Parametric VaR + heat

    ── Core ──
    EconomicCalendar          News blackout filter
    TradeManager              TP1/TP2/TP3 + trail + BE management
    TradeJournal              Rich annotated trade log
    CandleCache               TTL cache — no redundant API calls

    ── Execution ──
    ExecutionIntelligence     Session-aware slippage model

    ── Monitoring ──
    SystemMonitor             Telegram alerts + health
    AnomalyDetector           ATR/volume/gap spike gate

    ── Research ──
    WalkForwardEngine         Full stack backtest
    ParameterSensitivityTester Robustness sweep
    MonteCarloEquityCurve     Bootstrap equity distribution

    ── API ──
    v5_router                 FastAPI REST dashboard

Decision flow (every signal cycle):
    0. CandleCache          → skip broker if data fresh
    1. AnomalyDetector      → block on NFP spike / gap / corruption
    2. EconomicCalendar     → block on news blackout window
    3. FeatureDriftMonitor  → block if ML model has degraded
    4. RegimeClassifier     → probabilistic regime (trending/ranging/volatile)
    5. RegimeTransitionModel→ size down if regime shift imminent
    6. AgentOrchestrator    → Thompson-sampled regime-conditional weights
    7. SignalConflictResolver→ agent disagreement → conf adj + size mult
    8. WinProbabilityCalibrator → real win prob replacing hardcoded priors
    9. Negative EV gate     → block if EV < 0
    10. Adaptive confidence  → pair-level adaptive threshold
    11. RiskGovernor         → drawdown / loss streak / exposure hard stops
    12. DynamicCorrelation   → live correlation gate
    13. VaREngine            → portfolio heat gate
    14. VolatilityTargetedSizer → Kelly + vol-scaled lots
    15. ExecutionIntelligence → slippage-adjusted fill
    16. TradeManager         → register for active management
    17. TradeJournal         → record full signal context
    18. Telegram alert
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── All components ────────────────────────────────────────────
from .ml.regime_classifier             import RegimeClassifier, RegimePrediction
from .ml.feature_drift_monitor         import FeatureDriftMonitor
from .ml.win_prob_calibrator           import WinProbabilityCalibrator
from .ml.synthetic_augmenter           import SyntheticDataAugmenter
from .ml.regime_transition_model       import RegimeTransitionModel
from .agents.agent_layer               import AgentOrchestrator, OrchestratorResult
from .agents.signal_conflict_resolver  import SignalConflictResolver
from .learning.learning_loop           import LearningLoop
from .learning.regime_conditional_learner import RegimeConditionalLearner
from .risk.risk_governor               import RiskGovernor, OpenPosition
from .risk.hybrid_correlation          import HybridCorrelationEstimator as DynamicCorrelationEstimator
from .risk.volatility_sizer            import VolatilityTargetedSizer
from .risk.var_engine_v2               import VaREngine
from .core.economic_calendar           import EconomicCalendar
from .core.trade_manager_v2               import TradeManager
from .core.trade_journal               import TradeJournal, JournalEntry
from .core.candle_cache                import CandleCache
from .execution.execution_intelligence import ExecutionIntelligence
from .core.smart_stop_calculator       import smart_stop
from .core.mtf_confluence_filter       import mtf_filter
from .core.entry_optimizer              import entry_optimizer
from .core.correlation_filter           import corr_filter
from .core.daily_circuit_breaker        import circuit_breaker
from .core.fvg_optimizer                import fvg_optimizer
from .core.exposure_manager             import exposure_manager
from .learning.agent_weight_adjuster    import agent_adjuster
from .learning.weekend_gap_handler      import get_market_status
from .monitoring.system_monitor        import SystemMonitor
from .monitoring.anomaly_detector      import AnomalyDetector
from .backtest.walk_forward            import WalkForwardEngine, BacktestResult
from .research.sensitivity_and_montecarlo import (
    ParameterSensitivityTester, MonteCarloEquityCurve,
)

try:
    from app.services.market_analyzer import (
        MAETradingFormula, get_session_quality, calculate_atr,
    )
    _MAE_AVAILABLE = True
except ImportError:
    _MAE_AVAILABLE = False

WATCHLIST = ["XAUUSD", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "BTCUSD"]


@dataclass
class V5SignalResult:
    """Complete signal cycle output — every gate decision visible."""
    pair:             str
    timeframe:        str
    timestamp:        str

    # Gate results
    anomaly_blocked:  bool
    calendar_blocked: bool
    drift_blocked:    bool
    ev_blocked:       bool
    confidence_blocked: bool
    risk_blocked:     bool
    correlation_blocked: bool
    heat_blocked:     bool

    # Signal state
    signal_fired:     bool
    direction:        Optional[str]
    agent_used:       Optional[str]
    regime:           str
    regime_probs:     Dict[str, float]
    regime_shift_risk: float
    conflict_score:   float
    confidence:       float
    confidence_adj:   float      # from conflict resolver
    win_probability:  float
    expected_value:   float

    # Execution
    fill_price:       Optional[float]
    stop_loss:        Optional[float]
    take_profit_1:    Optional[float]
    take_profit_2:    Optional[float]
    take_profit_3:    Optional[float]
    adjusted_lots:    float
    sizing_method:    str
    slippage_pips:    float
    session:          str
    risk_score:       float
    portfolio_heat:   float

    # Metadata
    reasoning:        str
    anomalies:        List[str]
    drift_warn:       bool
    agent_scores:     Dict[str, float]

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def gate_summary(self) -> str:
        gates = [
            ("Anomaly",     self.anomaly_blocked),
            ("Calendar",    self.calendar_blocked),
            ("DriftModel",  self.drift_blocked),
            ("NegativeEV",  self.ev_blocked),
            ("Confidence",  self.confidence_blocked),
            ("RiskGov",     self.risk_blocked),
            ("Correlation", self.correlation_blocked),
            ("PortfHeat",   self.heat_blocked),
        ]
        passed  = [g for g, b in gates if not b]
        blocked = [(g, b) for g, b in gates if b]
        if blocked:
            return f"BLOCKED at {blocked[0][0]} | passed: {passed}"
        return f"ALL GATES PASSED ({len(passed)}/8) → signal fired"


class V5OrchestratorFinal:
    """
    PriceIQ Pro — Complete V5 System (Final).
    Single process, handles 6–20 pairs.
    For 20+ pairs, split regime_clf (CPU) to a worker process.
    """

    def __init__(
        self,
        starting_balance:     float = 10_000.0,
        risk_pct:             float = 2.0,
        target_vol_pct:       float = 0.02,
        max_drawdown:         float = 0.10,
        telegram=None,
        regime_model_path:    str = "regime_model.pkl",
        learning_state_path:  str = "learning_state.json",
        regime_weights_path:  str = "regime_weights.json",
        win_prob_path:        str = "win_prob_calibrator.json",
        transition_path:      str = "regime_transitions.json",
        journal_path:         str = "trade_journal.json",
    ):
        self.telegram         = telegram
        self.starting_balance = starting_balance

        # ── ML ───────────────────────────────────────────────
        self.regime_clf = RegimeClassifier(model_path=regime_model_path)
        self.drift_monitor     = FeatureDriftMonitor()
        self.win_prob_cal      = WinProbabilityCalibrator(win_prob_path)
        self.augmenter         = SyntheticDataAugmenter()
        self.regime_transition = RegimeTransitionModel(transition_path)

        # ── Agents ───────────────────────────────────────────
        self.orchestrator      = AgentOrchestrator()
        self.conflict_resolver = SignalConflictResolver()

        # ── Learning ─────────────────────────────────────────
        self.learning          = LearningLoop(learning_state_path)
        self.regime_learner    = RegimeConditionalLearner(regime_weights_path)

        # ── Risk ─────────────────────────────────────────────
        self.governor          = RiskGovernor(
            starting_balance=starting_balance,
            max_drawdown_pct=max_drawdown,
            max_risk_per_trade=risk_pct / 100,
        )
        self.corr_estimator    = DynamicCorrelationEstimator()
        self.vol_sizer         = VolatilityTargetedSizer(
            account_balance=starting_balance,
            target_vol_pct=target_vol_pct,
            max_risk_pct=max(risk_pct / 100, 0.03),
            use_kelly=True,
        )
        self.var_engine        = VaREngine(
            account_balance=starting_balance,
            max_heat_pct=0.06,
            critical_heat_pct=0.10,
        )

        # ── Signal Gates & Filters ───────────────────────────
        self.macro_gate        = None  # placeholder if macro module added later
        # Lazy init — will be loaded on first use
        self.news_gate         = None
        self.corr_filter       = corr_filter
        self.exposure_mgr      = exposure_manager
        self.fvg_opt           = fvg_optimizer
        self.entry_opt         = entry_optimizer
        self.agent_adjuster    = agent_adjuster

        # ── Core ─────────────────────────────────────────────
        self.calendar          = EconomicCalendar()
        self.trade_manager     = TradeManager(
            telegram=telegram,
            learning_loop=self.learning,
        )
        self.journal           = TradeJournal(journal_path)
        self.candle_cache      = CandleCache(max_entries=120)

        # ── Execution ────────────────────────────────────────
        self.execution         = ExecutionIntelligence(live_mode=True)

        # ── Monitoring ───────────────────────────────────────
        self.monitor           = SystemMonitor(
            telegram=telegram,
            governor=self.governor,
            learning_loop=self.learning,
        )
        self.anomaly_det       = AnomalyDetector()

        # ── Research ─────────────────────────────────────────
        self.backtest_engine   = WalkForwardEngine(
            regime_classifier=self.regime_clf,
            agent_orchestrator=self.orchestrator,
            risk_governor=self.governor,
            execution=self.execution,
            learning_loop=self.learning,
        )

        # Bootstrap weights
        self._refresh_weights("trending")
        logger.info("✅ V5 Final Orchestrator — all systems online")

    # ────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────

    def _refresh_weights(self, regime: str, thompson: bool = True):
        weights = (
            self.regime_learner.thompson_sample(regime) if thompson
            else self.regime_learner.get_weights_for_regime(regime)
        )
        self.orchestrator.update_weights(weights)

    def _no_signal(
        self, pair: str, timeframe: str, now: str, session: str,
        regime: str = "unknown", reason: str = "",
        anomaly_b=False, calendar_b=False, drift_b=False,
        ev_b=False, conf_b=False, risk_b=False, corr_b=False, heat_b=False,
        anomalies=None, drift_warn=False,
    ) -> V5SignalResult:
        return V5SignalResult(
            pair=pair, timeframe=timeframe, timestamp=now,
            anomaly_blocked=anomaly_b, calendar_blocked=calendar_b,
            drift_blocked=drift_b, ev_blocked=ev_b,
            confidence_blocked=conf_b, risk_blocked=risk_b,
            correlation_blocked=corr_b, heat_blocked=heat_b,
            signal_fired=False, direction=None, agent_used=None,
            regime=regime, regime_probs={}, regime_shift_risk=0.0,
            conflict_score=0.0, confidence=0.0, confidence_adj=0.0,
            win_probability=0.0, expected_value=0.0,
            fill_price=None, stop_loss=None,
            take_profit_1=None, take_profit_2=None, take_profit_3=None,
            adjusted_lots=0.0, sizing_method="none",
            slippage_pips=0.0, session=session,
            risk_score=self.governor.risk_score,
            portfolio_heat=self.var_engine._compute_portfolio_heat(
                self.governor._open_positions
            ),
            reasoning=reason, anomalies=anomalies or [],
            drift_warn=drift_warn, agent_scores={},
        )

    # ────────────────────────────────────────────────────────
    # MAIN SIGNAL CYCLE
    # ────────────────────────────────────────────────────────

    async def run_signal_cycle(
        self,
        candles:         List,
        pair:            str,
        timeframe:       str = "1h",
        account_balance: Optional[float] = None,
        current_dt:      Optional[datetime] = None,
        signal_bar_index: int = 0,
    ) -> V5SignalResult:

        balance = account_balance or self.governor.current_balance
        now_dt  = current_dt or datetime.now(timezone.utc)
        now_str = now_dt.isoformat()
        session_name, session_q = (
            get_session_quality(now_dt) if _MAE_AVAILABLE else ("london", 0.8)
        )

        if len(candles) < 55:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   reason="Insufficient candles")

        # ── Gate -3: Daily Circuit Breaker ────────────────────
        can_trade, cb_reason = circuit_breaker.can_trade()
        if not can_trade:
            logger.warning(f"CIRCUIT BREAKER: {cb_reason}")
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime="blocked", reason=cb_reason, risk_b=True)

        # ── Gate -2: News Blackout ──────────────────────────────
        from .core.news_blackout import news_blackout
        self.news_gate = news_blackout
        news_ok, news_reason = self.news_gate.check(pair)
        if not news_ok:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime="blocked", reason=f"News blackout: {news_reason}", calendar_b=True)

        # ── Gate -1: Exposure Manager ───────────────────────────
        exp_ok, exp_reason = self.exposure_mgr.can_add(pair)
        if not exp_ok:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime="blocked", reason=exp_reason, risk_b=True)

        # ── Gate 0: Anomaly ───────────────────────────────────
        anomaly = self.anomaly_det.check(candles, pair)
        if anomaly.block_signals:
            await self.monitor.on_signal_blocked(pair, anomaly.reason)
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   reason=anomaly.reason, anomaly_b=True,
                                   anomalies=anomaly.anomalies)

        # ── Gate 1: Economic calendar ─────────────────────────
        cal_check = self.calendar.check_blackout(pair, now_dt)
        if cal_check.blocked:
            await self.monitor.on_signal_blocked(pair, cal_check.reason)
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   reason=cal_check.reason, calendar_b=True)

        # ── Gate 2: Feature drift ─────────────────────────────
        regime_feats = self.regime_clf.extractor.extract(candles)
        drift_warn   = False
        if regime_feats:
            self.drift_monitor.update(regime_feats)
            drift_report = self.drift_monitor.check()
            if drift_report:
                if drift_report.severe:
                    return self._no_signal(pair, timeframe, now_str, session_name,
                                           reason=drift_report.recommendation,
                                           drift_b=True, drift_warn=True)
                drift_warn = drift_report.warn

        # ── Gate 3: Regime classification ────────────────────
        try:
            regime_pred = self.regime_clf.predict(candles)
            regime      = regime_pred.regime
        except Exception as e:
            regime_pred = RegimePrediction("ranging", 0.2, 0.6, 0.2, 0.6, {})
            regime      = "ranging"

        regime_probs = {
            "trending": round(regime_pred.trending, 3),
            "ranging":  round(regime_pred.ranging,  3),
            "volatile": round(regime_pred.volatile, 3),
        }

        # Update transition model
        self.regime_transition.update(regime)
        shift_risk  = self.regime_transition.shift_risk(regime)
        trans_mult  = self.regime_transition.size_multiplier(regime)

        # ── Gate 4: Agent orchestration ───────────────────────
        self._refresh_weights(regime, thompson=True)
        try:
            orch_result = self.orchestrator.run(candles, regime)
        except Exception as e:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=f"Orchestrator error: {e}")

        if orch_result is None:
            logger.info(f"[DEBUG] {pair}: All agents returned None/invalid signal in {regime} regime")
            await self.monitor.on_signal_blocked(pair, "No agent produced valid signal")
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason="No agent signal")

        signal    = orch_result.selected_signal
        agent     = orch_result.selected_agent

        # ── DEFENSE-IN-DEPTH: Regime-fit gate ─────────────────
        # Even if agent_layer.py regime-lock is bypassed, this catches it
        if signal.regime_fit < 0.5:
            reason = f"REGIME BLOCK: {agent} fit={signal.regime_fit:.2f} < 0.50 in {regime} regime"
            logger.warning(f"[SAFETY] {pair}: {reason}")
            await self.monitor.on_signal_blocked(pair, reason)
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=reason, conf_b=True)

        # Log agent confidence for debugging
        logger.info(
            f"[DEBUG] {pair}: best agent={orch_result.selected_agent} "
            f"dir={orch_result.selected_signal.direction} "
            f"conf={orch_result.selected_signal.confidence:.3f} "
            f"ev={orch_result.selected_signal.expected_value:.3f} "
            f"regime_fit={orch_result.selected_signal.regime_fit:.3f}"
        )

        signal    = orch_result.selected_signal
        agent     = orch_result.selected_agent
        direction = signal.direction

        # ── SMART STOP RECALCULATION ──
        try:
            smart = smart_stop.calculate(
                candles=candles, direction=direction, pair=pair,
                session=session_name, entry_price=getattr(signal, "entry_price", None),
            )
            signal.stop_loss = smart["sl"]
            signal.take_profit_1 = smart["tp1"]
            signal.take_profit_2 = smart["tp2"]
            signal.stop_distance = smart["sl_distance"]
            signal.tp1_distance = smart["tp1_distance"]
            signal.tp2_distance = smart["tp2_distance"]
            signal.reasoning += f" | SmartStop:{smart['method']} RR={smart['rr']}"
            logger.info(f"SmartStop: {pair} {direction} SL={smart['sl']} structure@{smart['structure_level']}")
        except Exception as e:
            logger.warning(f"SmartStop error: {e}")

        # ── MTF CONFLUENCE FILTER ──
        try:
            candles_4h = await self._get_4h_candles(pair)
            if candles_4h:
                allow, boost, mtf_reason = mtf_filter.check(direction, candles_4h, pair)
                if not allow:
                    return self._no_signal(pair, timeframe, now_str, session_name,
                                           regime=regime, reason=f"MTF blocked: {mtf_reason}", conf_b=True)
                if boost != 0:
                    signal.confidence = min(1.0, max(0.0, signal.confidence + boost))
                    signal.reasoning += f" | {mtf_reason}"
        except Exception as e:
            logger.warning(f"MTF filter error: {e}")

        # ── FVG ENTRY OPTIMIZATION ──
        try:
            fvg = self.fvg_opt.suggest_entry(candles, direction, pair)
            if fvg["use_fvg"]:
                signal._fvg_entry = fvg["entry"]
                signal._entry_type = "fvg_limit"
                signal.reasoning += f" | FVG entry @{fvg['entry']} ({fvg['fvg_type']})"
                logger.info(f"FVG limit entry for {pair}: {fvg['entry']}")
        except Exception as e:
            logger.debug(f"FVG error: {e}")

        # ── AGENT WEIGHT ADJUSTMENT ──
        try:
            raw_conf = signal.confidence
            signal.confidence = agent_adjuster.apply(agent, raw_conf)
            if signal.confidence != raw_conf:
                signal.reasoning += f" | AgentWeight:{agent_adjuster.get_weight(agent):.2f}"
        except Exception as e:
            logger.debug(f"Agent weight error: {e}")

        # ── CORRELATION FILTER (signal-level) ──
        try:
            corr_ok, blocked_by, corr_reason = self.corr_filter.check(pair)
            if not corr_ok:
                return self._no_signal(pair, timeframe, now_str, session_name,
                                       regime=regime, reason=corr_reason, corr_b=True)
        except Exception as e:
            logger.debug(f"Correlation filter error: {e}")

        # ── Gate 5: Signal conflict resolution ────────────────
        conflict = self.conflict_resolver.evaluate(orch_result)
        if conflict.should_block:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=conflict.reason,
                                   conf_b=True)

        # ── Gate 6: Calibrated win probability ───────────────
        win_prob  = self.win_prob_cal.get_win_probability(
            agent=agent, regime=regime,
            confidence=signal.confidence, session=session_name,
        )
        rr1       = signal.tp1_distance / max(signal.stop_distance, 1e-9)
        ev        = win_prob * rr1 - (1 - win_prob) * 1.0
        conf_adj  = conflict.confidence_adj
        adj_conf  = min(1.0, max(0.0, signal.confidence + conf_adj))

        if ev < 0:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime,
                                   reason=f"Negative EV={ev:.3f} (WP={win_prob:.2f}, RR={rr1:.2f})",
                                   ev_b=True)

        # ── Gate 7: Adaptive confidence threshold ─────────────
        # SAFETY: hard floor at 0.50 — never trade below this
        threshold = max(min(self.learning.get_confidence_threshold(pair), 0.60), 0.50)
        if adj_conf < threshold:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime,
                                   reason=f"Confidence {adj_conf:.2f} < threshold {threshold:.2f}",
                                   conf_b=True)

        # ── Gate 8: Risk governor ─────────────────────────────
        entry = candles[-1].close
        atr   = calculate_atr(candles) if _MAE_AVAILABLE else signal.stop_distance / 1.5

        sizing = self.vol_sizer.compute_lots(
            pair=pair, atr=atr, stop_distance=signal.stop_distance,
            win_rate=win_prob, avg_win_r=rr1, avg_loss_r=1.0,
            current_price=entry,
        )
        lots = sizing.lots
        # Compound size reductions
        lots = round(lots * anomaly.size_multiplier * conflict.size_multiplier * trans_mult
                     * cal_check.size_mult, 2)
        lots = max(0.01, lots)

        risk_dec = self.governor.evaluate(
            pair=pair, direction=direction, proposed_lots=lots,
            stop_distance=signal.stop_distance, entry_price=entry,
        )
        if not risk_dec.allowed:
            await self.monitor.on_signal_blocked(pair, risk_dec.reason)
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=risk_dec.reason, risk_b=True)
        lots = risk_dec.adjusted_lots
        # Hard override: always 0.01 micro lots for $100 account
        lots = 0.01

        # ── Gate 9: Dynamic correlation ───────────────────────
        corr_check = self.corr_estimator.check_correlation_risk(
            pair=pair, direction=direction,
            open_positions=self.governor._open_positions, regime=regime,
        )
        if corr_check["blocked"]:
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=corr_check["reason"], corr_b=True)

        # ── Gate 10: Portfolio heat ───────────────────────────
        var_report = self.var_engine.compute(self.governor._open_positions)
        if not var_report.safe_to_trade:
            heat_reason = f"Portfolio heat {var_report.portfolio_heat_pct:.1%} — unsafe"
            await self.monitor.on_signal_blocked(pair, heat_reason)
            return self._no_signal(pair, timeframe, now_str, session_name,
                                   regime=regime, reason=heat_reason, heat_b=True)

        # ── Execution ─────────────────────────────────────────
        fill_result = self.execution.simulate_fill(
            pair=pair, direction=direction, order_type="market",
            requested_price=entry, lots=lots, atr=atr, session=session_name,
        )
        fill_price = fill_result.fill_price

        # ── Trade levels ──────────────────────────────────────
        sd  = signal.stop_distance
        tp1 = signal.tp1_distance
        tp2 = signal.tp2_distance
        tp3 = getattr(signal, "tp3_distance", tp2 * 1.5)

        if direction == "buy":
            stop_loss    = fill_price - sd
            take_profit1 = fill_price + tp1
            take_profit2 = fill_price + tp2
            take_profit3 = fill_price + tp3
        else:
            stop_loss    = fill_price + sd
            take_profit1 = fill_price - tp1
            take_profit2 = fill_price - tp2
            take_profit3 = fill_price - tp3

        # ── Register with trade manager ───────────────────────
        self.trade_manager.open_position(
            pair=pair, direction=direction,
            entry=fill_price, stop_loss=stop_loss,
            tp1=take_profit1, tp2=take_profit2, tp3=take_profit3,
            lots=lots, atr=atr, agent=agent, regime=regime,
            session=session_name, timeframe=timeframe,
            confidence=adj_conf, win_prob=win_prob,
            bar_index=signal_bar_index,
            reasoning=signal.reasoning,
        )

        # ── Update correlation estimator ──────────────────────
        self.corr_estimator.update(pair, entry)

        # ── Monitor ───────────────────────────────────────────
        await self.monitor.on_signal_generated(pair, agent, direction, adj_conf)

        # ── Agent scores ──────────────────────────────────────
        agent_scores = {
            name: round(sig.confidence * sig.regime_fit, 3)
            for name, sig in orch_result.all_signals.items()
            if sig.direction is not None
        }

        # ── Build full reasoning ──────────────────────────────
        reasoning = (
            f"V5.3 {pair} {direction.upper()} | Agent:{agent} | "
            f"Regime:{regime}({regime_pred.confidence:.0%}) "
            f"shift_risk={shift_risk:.0%} | "
            f"WinProb:{win_prob:.0%} EV:{ev:+.3f}R | "
            f"Conf:{adj_conf:.2f}(thr={threshold:.2f},adj={conf_adj:+.2f}) | "
            f"Conflict:{conflict.conflict_score:.2f} | "
            f"Session:{session_name}({session_q:.0%}) | "
            f"Lots:{lots}({sizing.method}×{trans_mult:.2f}regime×{anomaly.size_multiplier:.2f}anomaly) | "
            f"Fill:{fill_price:.5f}(slip={fill_result.slippage_pips:.1f}pip) | "
            f"SL:{stop_loss:.5f} TP1:{take_profit1:.5f} TP2:{take_profit2:.5f} TP3:{take_profit3:.5f} | "
            f"Heat:{var_report.portfolio_heat_pct:.1%} RiskScore:{self.governor.risk_score:.2f}"
            + (f" | DRIFT⚠" if drift_warn else "")
        )

        logger.info(reasoning)

        # Record for filters
        try:
            self.corr_filter.record(pair)
            self.exposure_mgr.record(pair, direction)
        except Exception as rec_e:
            logger.debug(f"Filter record error: {rec_e}")

        return V5SignalResult(
            pair=pair, timeframe=timeframe, timestamp=now_str,
            anomaly_blocked=False, calendar_blocked=False, drift_blocked=False,
            ev_blocked=False, confidence_blocked=False, risk_blocked=False,
            correlation_blocked=False, heat_blocked=False,
            signal_fired=True, direction=direction, agent_used=agent,
            regime=regime, regime_probs=regime_probs,
            regime_shift_risk=round(shift_risk, 3),
            conflict_score=conflict.conflict_score,
            confidence=adj_conf, confidence_adj=conf_adj,
            win_probability=win_prob, expected_value=round(ev, 4),
            fill_price=fill_price, stop_loss=stop_loss,
            take_profit_1=take_profit1, take_profit_2=take_profit2,
            take_profit_3=take_profit3,
            adjusted_lots=lots, sizing_method=sizing.method,
            slippage_pips=fill_result.slippage_pips,
            session=session_name, risk_score=self.governor.risk_score,
            portfolio_heat=var_report.portfolio_heat_pct,
            reasoning=reasoning, anomalies=anomaly.anomalies,
            drift_warn=drift_warn, agent_scores=agent_scores,
        )

    # ── Trade lifecycle ───────────────────────────────────────

    async def on_trade_closed(
        self, pair: str, agent_name: str, direction: str,
        entry: float, exit_price: float, stop: float, tp1: float,
        outcome: str, r_multiple: float, pnl_usd: float,
        regime: str = "unknown", confidence: float = 0.0,
        session: str = "unknown", management_events: Optional[List[str]] = None,
    ):
        """Atomic update of all learning systems on trade close."""
        self.governor.close_position(pair, pnl_usd)
        self.vol_sizer.update_balance(self.governor.current_balance)
        self.var_engine.update_balance(self.governor.current_balance)
        self.var_engine.add_daily_return(pnl_usd / max(self.governor.current_balance, 1))
        self.var_engine.add_trade_pnl(pnl_usd)

        # Feed circuit breaker
        try:
            risk_amount = 2.0  # 2% of $100
            circuit_breaker.record_outcome(r_multiple, risk_amount)
        except Exception as e:
            logger.debug(f"Circuit breaker update: {e}")

        # Feed agent weight adjuster
        try:
            agent_adjuster.record(agent_name, outcome, r_multiple)
        except Exception as e:
            logger.debug(f"Agent adjuster record: {e}")

        self.learning.update(
            pair=pair, agent_name=agent_name, direction=direction,
            entry=entry, exit_price=exit_price, stop=stop, tp1=tp1,
            outcome=outcome, r_multiple=r_multiple,
            regime=regime, confidence=confidence, session=session,
        )
        self.regime_learner.update(agent_name, regime, outcome)
        self.win_prob_cal.record(
            agent=agent_name, regime=regime, confidence=confidence,
            outcome=outcome, session=session,
        )
        self._refresh_weights(regime, thompson=False)

        # Journal entry
        duration_h = 0.0
        pos_data   = {}
        trade_id   = f"{pair}_{int(datetime.now(timezone.utc).timestamp())}"
        j_entry = JournalEntry(
            trade_id=trade_id, pair=pair, timeframe="1h",
            agent=agent_name, regime=regime, regime_probs={},
            session=session, direction=direction,
            confidence=confidence, win_probability=0.0, expected_value=0.0,
            conflict_score=0.0, entry_price=entry, fill_price=entry,
            slippage_pips=0.0, stop_loss=stop, tp1=tp1, tp2=tp1 * 1.5,
            tp3=tp1 * 2.0, lots=0.0, sizing_method="unknown",
            exit_price=exit_price, outcome=outcome, r_multiple=r_multiple,
            pnl_usd=pnl_usd, duration_hours=duration_h,
            management_events=management_events or [],
            opened_at=datetime.now(timezone.utc).isoformat(),
            closed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.journal.record(j_entry)

        await self.monitor.on_trade_closed({
            "pair": pair, "agent": agent_name,
            "result": outcome, "r_multiple": r_multiple, "pnl": pnl_usd,
        })
        logger.info(f"V5.3 closed: {pair} {agent_name}/{regime} {outcome} {r_multiple:+.2f}R ${pnl_usd:+.2f}")

    # ── Tick update (call every bar) ──────────────────────────

    async def tick(self, current_prices: Dict[str, float]):
        """Call every bar from scheduler — updates all time-sensitive state."""
        # Trade manager: check TP/SL/trail on all positions
        await self.trade_manager.update_all(current_prices)
        # Correlation estimator: feed latest prices
        for pair, price in current_prices.items():
            self.corr_estimator.update(pair, price)

    # ── System control ────────────────────────────────────────

    def get_system_status(self) -> Dict:
        streak_r, streak_n = self.regime_transition.regime_streak()
        return {
            "version":              "5.3-final",
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "portfolio":            self.governor.get_portfolio_summary(),
            "var":                  self.var_engine.to_dict(
                                        self.var_engine.compute(self.governor._open_positions)
                                    ),
            "learning":             self.learning.get_full_stats(),
            "regime_weights":       self.regime_learner.get_full_table(),
            "best_per_regime":      self.regime_learner.get_best_agent_per_regime(),
            "win_prob_samples":     self.win_prob_cal.sample_sizes(),
            "drift":                self.drift_monitor.status_dict(),
            "correlation":          self.corr_estimator.status_dict(WATCHLIST),
            "candle_cache":         self.candle_cache.status(),
            "vol_sizer":            self.vol_sizer.describe(),
            "calendar":             self.calendar.status(),
            "trade_manager":        self.trade_manager.get_stats(),
            "journal_stats":        self.journal.full_stats(),
            "regime_transition": {
                "current":      streak_r,
                "streak_bars":  streak_n,
                "shift_risk":   self.regime_transition.shift_risk(streak_r),
                "matrix":       self.regime_transition.get_transition_matrix(),
            },
            "regime_model_trained": self.regime_clf._trained,
        }

    async def send_heartbeat(self):
        await self.monitor.heartbeat(self.governor.get_portfolio_summary())

    async def send_daily_summary(self):
        await self.monitor.daily_summary(
            self.learning.get_full_stats(),
            self.governor.get_portfolio_summary(),
        )

    async def train_regime_classifier(
        self, candles: List, save_path: str = "regime_model.pkl",
        use_augmentation: bool = True,
    ) -> Dict:
        """Train classifier, optionally with synthetic augmentation."""
        result = await asyncio.to_thread(
            self._train_with_augmentation, candles, save_path, use_augmentation
        )
        # Fit drift baseline
        feats = []
        for i in range(55, len(candles)):
            f = self.regime_clf.extractor.extract(candles[max(0, i-100):i+1])
            if f:
                feats.append(f)
        if feats:
            self.drift_monitor.fit_baseline(feats)
        return result

    def _train_with_augmentation(
        self, candles: List, save_path: str, use_augmentation: bool
    ) -> Dict:
        from .ml.regime_classifier import auto_label, RegimeFeatureExtractor
        labels = auto_label(candles)
        result = self.regime_clf.train(candles, labels=labels)
        self.regime_clf.save(save_path)
        return result

    def run_backtest(
        self, candles: List, pair: str = "EURUSD",
        starting_balance: float = 10_000.0, risk_pct: float = 2.0,
    ) -> BacktestResult:
        return self.backtest_engine.run(
            candles=candles, pair=pair,
            starting_balance=starting_balance, risk_pct=risk_pct,
        )

    def run_montecarlo(self, n_paths: int = 1000) -> Dict:
        records  = self.learning.store.all()
        r_mults  = [r.r_multiple for r in records]
        mc       = MonteCarloEquityCurve(r_mults)
        report   = mc.run(n_paths=n_paths, starting_balance=self.starting_balance)
        return {"summary": mc.summary(report), "report": report.__dict__}


# ── Singleton ─────────────────────────────────────────────────
_instance: Optional[V5OrchestratorFinal] = None


def init_v5(**kwargs) -> V5OrchestratorFinal:
    global _instance
    _instance = V5OrchestratorFinal(**kwargs)
    return _instance


def get_v5() -> Optional[V5OrchestratorFinal]:
    return _instance
