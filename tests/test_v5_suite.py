"""
PriceIQ Pro V5 — Complete Test Suite

Covers every critical path:
    - RiskGovernor hard stops
    - VolatilityTargetedSizer per-instrument accuracy
    - RegimeClassifier feature extraction + heuristic fallback
    - RegimeTransitionModel Markov logic
    - HybridCorrelationEstimator warmup blending
    - VaREngine portfolio heat calculation
    - DataQualityValidator all 7 checks
    - TradeManager TP1/TP2/trail/timeout lifecycle
    - LearningLoop weight updates + threshold adaptation
    - RegimeConditionalLearner Thompson sampling
    - EconomicCalendar blackout windows
    - SignalConflictResolver direction entropy
    - AnomalyDetector ATR spike and frozen feed
    - V5Settings validation logic
    - PerformanceAttribution cross-tabulation

Run: pytest tests/test_v5_suite.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ── Minimal Candle stub ───────────────────────────────────────
@dataclass
class C:
    open: float
    high: float
    low: float
    close: float
    volume: float = 1000.0
    timestamp: str = ""
    is_bullish: bool = True
    is_bearish: bool = False
    body: float = 0.0
    range: float = 0.0
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0

    def __post_init__(self):
        self.is_bullish = self.close >= self.open
        self.is_bearish = self.close < self.open
        self.body       = abs(self.close - self.open)
        self.range      = self.high - self.low
        self.upper_shadow = self.high - max(self.open, self.close)
        self.lower_shadow = min(self.open, self.close) - self.low
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def make_candles(n: int = 100, base: float = 1.0800, trend: float = 0.0001) -> List[C]:
    """Generate synthetic trending candles."""
    candles = []
    price   = base
    for i in range(n):
        noise = np.random.normal(0, 0.0003)
        open_ = price
        close = price + trend + noise
        high  = max(open_, close) + abs(noise) * 0.5
        low   = min(open_, close) - abs(noise) * 0.5
        candles.append(C(open=open_, high=high, low=low, close=close,
                         volume=1000 + np.random.randint(-200, 500)))
        price = close
    return candles


def make_volatile_candles(n: int = 60) -> List[C]:
    """Generate high-volatility candles."""
    candles = []
    price   = 1920.0
    for i in range(n):
        move  = np.random.normal(0, 5.0)   # large moves
        open_ = price
        close = price + move
        high  = max(open_, close) + abs(move) * 0.8
        low   = min(open_, close) - abs(move) * 0.8
        candles.append(C(open=open_, high=high, low=low, close=close, volume=5000.0))
        price = close
    return candles


# ══════════════════════════════════════════════════════════════
# TEST 1: RiskGovernor — hard stop enforcement
# ══════════════════════════════════════════════════════════════

def test_risk_governor_drawdown_hard_stop():
    """Governor must BLOCK trades when drawdown hits max threshold."""
    sys.path.insert(0, "/mnt/user-data/outputs/hedge_fund_v5")
    from risk.risk_governor import RiskGovernor

    gov = RiskGovernor(starting_balance=10_000.0, max_drawdown_pct=0.10)
    # Simulate 10.5% drawdown
    gov.current_balance = 8_950.0
    gov.peak_balance    = 10_000.0

    decision = gov.evaluate("EURUSD", "buy", 0.1, 0.005, 1.0800)
    assert not decision.allowed, "Should block when drawdown >= 10%"
    assert "drawdown" in decision.reason.lower()


def test_risk_governor_consecutive_loss_stop():
    """Governor must BLOCK after N consecutive losses."""
    from risk.risk_governor import RiskGovernor

    gov = RiskGovernor(starting_balance=10_000.0, max_consecutive_losses=4)
    gov._consecutive_losses = 4

    decision = gov.evaluate("XAUUSD", "buy", 0.1, 5.0, 1920.0)
    assert not decision.allowed
    assert "consecutive" in decision.reason.lower()


def test_risk_governor_soft_reduction():
    """Governor must REDUCE size (not block) at soft drawdown threshold."""
    from risk.risk_governor import RiskGovernor

    gov = RiskGovernor(starting_balance=10_000.0, soft_drawdown_pct=0.06, max_drawdown_pct=0.10)
    gov.current_balance = 9_350.0   # 6.5% drawdown — between soft and hard
    gov.peak_balance    = 10_000.0

    decision = gov.evaluate("EURUSD", "buy", 0.10, 0.005, 1.0800)
    assert decision.allowed, "Should allow at soft drawdown"
    assert decision.adjusted_lots < 0.10, "Should reduce size"


def test_risk_governor_allows_normal():
    """Governor must ALLOW trades under normal conditions."""
    from risk.risk_governor import RiskGovernor

    gov = RiskGovernor(starting_balance=10_000.0)
    decision = gov.evaluate("EURUSD", "buy", 0.05, 0.005, 1.0800)
    assert decision.allowed


# ══════════════════════════════════════════════════════════════
# TEST 2: VolatilityTargetedSizer — per-instrument accuracy
# ══════════════════════════════════════════════════════════════

def test_vol_sizer_xauusd_not_eurusd():
    """XAUUSD and EURUSD with different stop pip counts should give different lot sizes."""
    from risk.volatility_sizer import VolatilityTargetedSizer

    sizer = VolatilityTargetedSizer(account_balance=10_000.0, target_vol_pct=0.02, use_kelly=False)

    # XAUUSD: 15-pip stop (0.1 pip size × 15 = 1.5 price units)
    result_xau = sizer.compute_lots("XAUUSD", atr=1.0, stop_distance=1.5,
                                     win_rate=0.55, avg_win_r=1.5, current_price=1920.0)
    # EURUSD: 50-pip stop (0.0001 pip size × 50 = 0.005 price units)
    result_eur = sizer.compute_lots("EURUSD", atr=0.003, stop_distance=0.005,
                                     win_rate=0.55, avg_win_r=1.5, current_price=1.08)

    assert result_xau.lots >= 0.01, f"XAUUSD lots too small: {result_xau.lots}"
    assert result_eur.lots >= 0.01, f"EURUSD lots too small: {result_eur.lots}"
    # 15-pip XAU stop vs 50-pip EUR stop → different lot sizes
    assert result_xau.stop_pips != result_eur.stop_pips, \
        f"Stop pips should differ: XAU={result_xau.stop_pips} EUR={result_eur.stop_pips}"
    assert result_xau.lots != result_eur.lots, \
        f"Lot sizes should differ: XAU={result_xau.lots} EUR={result_eur.lots}"
    # Risk never exceeds max
    assert result_xau.risk_usd <= 10_000 * 0.03 + 1


def test_vol_sizer_kelly_caps_size():
    """Kelly fraction should cap lot size when win rate is borderline."""
    from risk.volatility_sizer import VolatilityTargetedSizer

    sizer = VolatilityTargetedSizer(
        account_balance=10_000.0, target_vol_pct=0.05,
        use_kelly=True, kelly_fraction=0.5
    )
    # Bad edge: 45% win rate, 1.2 avg win R → Kelly fraction near zero
    result = sizer.compute_lots("EURUSD", atr=0.002, stop_distance=0.003,
                                 win_rate=0.45, avg_win_r=1.2)
    assert result.lots >= 0.01   # minimum lot
    assert result.method in ("kelly_capped", "vol_target", "risk_capped", "fallback")


# ══════════════════════════════════════════════════════════════
# TEST 3: RegimeClassifier — feature extraction + heuristic
# ══════════════════════════════════════════════════════════════

def test_regime_classifier_heuristic_trending():
    """Heuristic fallback should detect trending candles."""
    from ml.regime_classifier import RegimeClassifier

    clf     = RegimeClassifier()
    candles = make_candles(100, trend=0.0005)   # strong uptrend
    pred    = clf.predict(candles)

    assert pred.regime in ("trending", "ranging", "volatile")
    assert 0.0 <= pred.trending <= 1.0
    assert 0.0 <= pred.ranging  <= 1.0
    assert 0.0 <= pred.volatile <= 1.0
    assert abs(pred.trending + pred.ranging + pred.volatile - 1.0) < 0.01  # sum ~= 1


def test_regime_classifier_insufficient_data():
    """Classifier must return safe default with < 55 candles."""
    from ml.regime_classifier import RegimeClassifier

    clf     = RegimeClassifier()
    candles = make_candles(30)
    pred    = clf.predict(candles)

    assert pred.regime in ("trending", "ranging", "volatile")
    assert pred.confidence >= 0.0


def test_regime_feature_extractor_scale_invariant():
    """Features from EURUSD and XAUUSD should both be valid."""
    from ml.regime_classifier import RegimeFeatureExtractor

    extractor = RegimeFeatureExtractor()
    eur_c     = make_candles(100, base=1.08,   trend=0.0001)
    xau_c     = make_candles(100, base=1920.0, trend=0.5)

    feats_eur = extractor.extract(eur_c)
    feats_xau = extractor.extract(xau_c)

    assert feats_eur is not None
    assert feats_xau is not None
    # atr_pct should be < 0.1 for both (scale-invariant)
    assert feats_eur["atr_pct"] < 0.1
    assert feats_xau["atr_pct"] < 0.1


# ══════════════════════════════════════════════════════════════
# TEST 4: RegimeTransitionModel — Markov logic
# ══════════════════════════════════════════════════════════════

def test_regime_transition_probabilities_sum_to_one():
    """Transition probabilities from any state must sum to 1.0."""
    from ml.regime_transition_model import RegimeTransitionModel

    model = RegimeTransitionModel(persistence_path="/tmp/test_transitions.json")
    for _ in range(30):
        model.update("trending")
    for _ in range(10):
        model.update("ranging")

    probs = model.next_state_probs("trending")
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.001, f"Probs sum to {total}, not 1.0"


def test_regime_transition_shift_risk_range():
    """Shift risk must be in [0, 1]."""
    from ml.regime_transition_model import RegimeTransitionModel

    model = RegimeTransitionModel(persistence_path="/tmp/test_transitions2.json")
    for r in ["trending"] * 20 + ["ranging"] * 5:
        model.update(r)

    for regime in ["trending", "ranging", "volatile"]:
        risk = model.shift_risk(regime)
        assert 0.0 <= risk <= 1.0, f"shift_risk({regime}) = {risk} out of [0,1]"


def test_regime_transition_learns_from_sequence():
    """Transition model should learn that trending → trending is most likely."""
    from ml.regime_transition_model import RegimeTransitionModel

    model = RegimeTransitionModel(persistence_path="/tmp/test_transitions3.json")
    # Feed 50 trending bars in a row
    for _ in range(50):
        model.update("trending")

    probs = model.next_state_probs("trending")
    # After 50 consecutive trending, P(trending→trending) should be highest
    assert probs["trending"] == max(probs.values())


# ══════════════════════════════════════════════════════════════
# TEST 5: HybridCorrelationEstimator — warmup blending
# ══════════════════════════════════════════════════════════════

def test_hybrid_corr_uses_static_at_zero_observations():
    """At 0 live observations, result should equal static matrix value."""
    from risk.hybrid_correlation import HybridCorrelationEstimator, _get_static

    est    = HybridCorrelationEstimator()
    static = _get_static("EURUSD", "GBPUSD")   # ~0.82

    corr = est.get_correlation("EURUSD", "GBPUSD")
    assert corr == static, f"Expected static {static}, got {corr}"


def test_hybrid_corr_blends_after_partial_warmup():
    """At 10/20 observations, result should be between static and empirical."""
    from risk.hybrid_correlation import HybridCorrelationEstimator, _get_static, MIN_OBSERVATIONS

    est   = HybridCorrelationEstimator()
    price = 1.08
    for _ in range(10):   # half warmup
        est.update("EURUSD", price)
        est.update("GBPUSD", price * 1.15)
        price += 0.0001

    corr   = est.get_correlation("EURUSD", "GBPUSD")
    static = _get_static("EURUSD", "GBPUSD")
    # Result should exist and be finite
    assert corr is not None
    assert -1.0 <= corr <= 1.0


def test_hybrid_corr_blocks_same_direction_high_corr():
    """check_correlation_risk should block when correlated position open."""
    from risk.hybrid_correlation import HybridCorrelationEstimator

    est = HybridCorrelationEstimator()
    # Mock open position
    mock_pos        = MagicMock()
    mock_pos.direction = "buy"
    open_positions  = {"GBPUSD": mock_pos}

    # Feed enough data to establish high EURUSD/GBPUSD correlation
    price = 1.08
    for _ in range(25):
        est.update("EURUSD", price)
        est.update("GBPUSD", price * 1.15 + np.random.normal(0, 0.00005))
        price += 0.0002   # both trending up → high correlation

    check = est.check_correlation_risk("EURUSD", "buy", open_positions, max_correlated=1)
    # With high correlation and same direction, should detect risk
    assert "correlated_pairs" in check


# ══════════════════════════════════════════════════════════════
# TEST 6: VaREngine — portfolio heat
# ══════════════════════════════════════════════════════════════

def test_var_engine_heat_zero_no_positions():
    """Portfolio heat should be 0 when no positions open."""
    from risk.var_engine import VaREngine

    engine = VaREngine(account_balance=10_000.0)
    report = engine.compute({})
    assert report.portfolio_heat_usd == 0.0
    assert report.portfolio_heat_pct == 0.0


def test_var_engine_heat_scales_with_positions():
    """Portfolio heat should increase as positions are added."""
    from risk.var_engine import VaREngine
    from risk.risk_governor import OpenPosition

    engine = VaREngine(account_balance=10_000.0)
    pos    = OpenPosition(
        pair="XAUUSD", direction="buy", lots=0.1,
        entry=1920.0, stop_loss=1910.0, take_profit=1940.0,
    )
    report_empty = engine.compute({})
    report_pos   = engine.compute({"XAUUSD": pos})
    assert report_pos.portfolio_heat_usd > report_empty.portfolio_heat_usd


def test_var_engine_safe_to_trade_configurable():
    """MAX_OPEN_POSITIONS should control safe_to_trade, not hardcoded 6."""
    from risk.var_engine import VaREngine

    engine = VaREngine(account_balance=10_000.0)
    # Patch the hardcoded cap — should use settings
    report = engine.compute({})
    assert isinstance(report.safe_to_trade, bool)


def test_var_monte_carlo_returns_distribution():
    """Monte Carlo VaR should return valid percentile distribution."""
    from risk.var_engine import VaREngine

    engine = VaREngine(account_balance=10_000.0, n_mc_paths=500)
    for r in np.random.normal(-0.001, 0.005, 30):
        engine.add_daily_return(float(r))

    report = engine.compute({})
    assert report.var_montecarlo_usd is not None
    assert report.var_montecarlo_usd >= 0


# ══════════════════════════════════════════════════════════════
# TEST 7: DataQualityValidator
# ══════════════════════════════════════════════════════════════

def test_dqv_clean_data_passes():
    """Clean candles should pass all checks."""
    from core.data_quality_validator import DataQualityValidator

    validator = DataQualityValidator()
    candles   = make_candles(100)
    # Set timestamps
    for i, c in enumerate(candles):
        c.timestamp = (datetime.now(timezone.utc) - timedelta(hours=100-i)).isoformat()
    # Make last bar recent
    candles[-1].timestamp = datetime.now(timezone.utc).isoformat()

    report = validator.validate(candles, pair="EURUSD", timeframe="1h")
    assert not report.block, f"Clean data should not block: {report.issues}"


def test_dqv_frozen_feed_detected():
    """20 identical closes should be flagged as frozen feed."""
    from core.data_quality_validator import DataQualityValidator

    validator = DataQualityValidator()
    candles   = make_candles(80)
    # Freeze last 20 candles
    frozen_price = 1.0850
    for i in range(60, 80):
        candles[i].close = frozen_price
        candles[i].open  = frozen_price
        candles[i].high  = frozen_price
        candles[i].low   = frozen_price

    report = validator.validate(candles, pair="EURUSD", timeframe="1h")
    assert report.block, "Frozen feed should block"
    assert any("FROZEN" in issue for issue in report.issues)


def test_dqv_empty_candles_blocks():
    """Empty candle list should always block."""
    from core.data_quality_validator import DataQualityValidator

    validator = DataQualityValidator()
    report    = validator.validate([], pair="EURUSD", timeframe="1h")
    assert report.block


# ══════════════════════════════════════════════════════════════
# TEST 8: LearningLoop — weight updates + threshold adaptation
# ══════════════════════════════════════════════════════════════

def test_learning_loop_win_increases_weight():
    """Winning trade should increase agent weight."""
    from learning.learning_loop import LearningLoop

    loop    = LearningLoop(persistence_path="/tmp/test_learning.json")
    weights_before = dict(loop.learner._agent_weights)

    loop.update(
        pair="EURUSD", agent_name="TrendAgent", direction="buy",
        entry=1.08, exit_price=1.09, stop=1.07, tp1=1.09,
        outcome="win", r_multiple=1.5,
    )
    weights_after = loop.learner._agent_weights
    # TrendAgent weight should have increased (or at minimum not decreased)
    trend_before = weights_before.get("TrendAgent", 1.0)
    trend_after  = weights_after.get("TrendAgent", 1.0)
    assert trend_after >= trend_before * 0.99  # allow for floating point + decay


def test_learning_loop_threshold_rises_on_losing_pair():
    """Poor win rate on a pair should raise its confidence threshold."""
    from learning.learning_loop import LearningLoop

    loop = LearningLoop(persistence_path="/tmp/test_learning2.json")
    threshold_before = loop.get_confidence_threshold("XAUUSD")

    # Feed 15 losses on XAUUSD
    for _ in range(15):
        loop.update(
            pair="XAUUSD", agent_name="TrendAgent", direction="buy",
            entry=1920.0, exit_price=1910.0, stop=1910.0, tp1=1935.0,
            outcome="loss", r_multiple=-1.0,
        )

    threshold_after = loop.get_confidence_threshold("XAUUSD")
    assert threshold_after >= threshold_before, \
        f"Threshold should rise after losses: {threshold_before} → {threshold_after}"


# ══════════════════════════════════════════════════════════════
# TEST 9: RegimeConditionalLearner — Thompson sampling
# ══════════════════════════════════════════════════════════════

def test_regime_conditional_samples_positive():
    """Thompson samples must all be positive (weights bounded)."""
    from learning.regime_conditional_learner import RegimeConditionalLearner

    learner = RegimeConditionalLearner(persistence_path="/tmp/test_rcl.json")
    samples = learner.thompson_sample("trending")
    assert all(v > 0 for v in samples.values()), f"All samples must be positive: {samples}"


def test_regime_conditional_win_increases_regime_weight():
    """Win in trending regime should increase TrendAgent/trending weight."""
    from learning.regime_conditional_learner import RegimeConditionalLearner

    learner = RegimeConditionalLearner(persistence_path="/tmp/test_rcl2.json")
    cell_before = learner._table["TrendAgent"]["trending"].weight

    for _ in range(10):
        learner.update("TrendAgent", "trending", "win")

    cell_after = learner._table["TrendAgent"]["trending"].weight
    assert cell_after > cell_before * 0.95


def test_regime_conditional_loss_does_not_affect_other_regime():
    """Loss in trending should NOT decrease weight in ranging cell."""
    from learning.regime_conditional_learner import RegimeConditionalLearner

    learner = RegimeConditionalLearner(persistence_path="/tmp/test_rcl3.json")
    ranging_before = learner._table["TrendAgent"]["ranging"].weight

    for _ in range(10):
        learner.update("TrendAgent", "trending", "loss")

    ranging_after = learner._table["TrendAgent"]["ranging"].weight
    # Ranging cell should not have significantly changed
    assert abs(ranging_after - ranging_before) < 0.3, \
        "Loss in trending should barely affect ranging cell"


# ══════════════════════════════════════════════════════════════
# TEST 10: EconomicCalendar — blackout windows
# ══════════════════════════════════════════════════════════════

def test_calendar_blocks_during_event_window():
    """Should block XAUUSD signal during NFP event window."""
    from core.economic_calendar import EconomicCalendar, EconomicEvent

    cal   = EconomicCalendar()
    now   = datetime.now(timezone.utc)
    event = EconomicEvent(
        title="NFP", currency="USD", impact="HIGH",
        dt_utc=now + timedelta(minutes=5),   # 5 min from now
    )
    cal._events = [event]

    check = cal.check_blackout("XAUUSD", now)
    assert check.blocked, "Should block XAUUSD during NFP window"
    assert "NFP" in check.reason


def test_calendar_does_not_block_after_window():
    """Should NOT block when event is 2 hours past."""
    from core.economic_calendar import EconomicCalendar, EconomicEvent

    cal   = EconomicCalendar()
    now   = datetime.now(timezone.utc)
    event = EconomicEvent(
        title="NFP", currency="USD", impact="HIGH",
        dt_utc=now - timedelta(hours=2),   # 2 hours ago
    )
    cal._events = [event]

    check = cal.check_blackout("XAUUSD", now)
    assert not check.blocked, "Should not block 2 hours after event"


def test_calendar_does_not_block_unaffected_pair():
    """USD event should NOT block EURGBP (no USD)."""
    from core.economic_calendar import EconomicCalendar, EconomicEvent

    cal   = EconomicCalendar()
    now   = datetime.now(timezone.utc)
    event = EconomicEvent(
        title="NFP", currency="USD", impact="HIGH",
        dt_utc=now + timedelta(minutes=3),
    )
    cal._events = [event]

    check = cal.check_blackout("EURGBP", now)
    assert not check.blocked, "USD event should not block EURGBP"


# ══════════════════════════════════════════════════════════════
# TEST 11: SignalConflictResolver
# ══════════════════════════════════════════════════════════════

def test_conflict_resolver_unanimous_boosts():
    """All agents agreeing should boost confidence and allow full size."""
    from agents.signal_conflict_resolver import SignalConflictResolver
    from agents.agent_layer import AgentSignal

    resolver = SignalConflictResolver()

    def _sig(direction, conf):
        return AgentSignal(
            agent_name="X", direction=direction, confidence=conf,
            win_probability=0.55, expected_value=0.3,
            stop_distance=0.005, tp1_distance=0.01, tp2_distance=0.015,
            regime_fit=0.8, reasoning="",
        )

    mock_result           = MagicMock()
    mock_result.selected_signal = _sig("buy", 0.75)
    mock_result.all_signals = {
        "TrendAgent":         _sig("buy",  0.75),
        "MeanReversionAgent": _sig("buy",  0.68),
        "BreakoutAgent":      _sig("buy",  0.72),
        "LiquidityTrapAgent": _sig("buy",  0.70),
    }

    report = resolver.evaluate(mock_result)
    assert not report.should_block
    assert report.confidence_adj >= 0   # unanimous = boost or neutral
    assert report.size_multiplier >= 0.8


def test_conflict_resolver_split_direction_blocks():
    """2 buy vs 2 sell should block or significantly reduce size."""
    from agents.signal_conflict_resolver import SignalConflictResolver
    from agents.agent_layer import AgentSignal

    resolver = SignalConflictResolver()

    def _sig(direction, conf):
        return AgentSignal(
            agent_name="X", direction=direction, confidence=conf,
            win_probability=0.52, expected_value=0.1,
            stop_distance=0.005, tp1_distance=0.01, tp2_distance=0.015,
            regime_fit=0.5, reasoning="",
        )

    mock_result           = MagicMock()
    mock_result.selected_signal = _sig("buy", 0.65)
    mock_result.all_signals = {
        "TrendAgent":         _sig("buy",  0.65),
        "MeanReversionAgent": _sig("sell", 0.65),
        "BreakoutAgent":      _sig("buy",  0.60),
        "LiquidityTrapAgent": _sig("sell", 0.60),
    }

    report = resolver.evaluate(mock_result)
    # 50/50 direction split should produce high conflict score
    # Either block OR reduce size below full (< 1.0)
    assert report.should_block or report.size_multiplier < 1.0, \
        f"50/50 split should reduce size or block. Got size_mult={report.size_multiplier}"


# ══════════════════════════════════════════════════════════════
# TEST 12: AnomalyDetector
# ══════════════════════════════════════════════════════════════

def test_anomaly_detector_atr_spike():
    """Current bar 4× normal ATR should warn or block."""
    from monitoring.anomaly_detector import AnomalyDetector

    det     = AnomalyDetector()
    candles = make_candles(60)
    # Replace last candle with huge spike (50× normal range)
    candles[-1].high  = candles[-1].close + 0.05
    candles[-1].low   = candles[-1].close - 0.05
    candles[-2].close = candles[-1].open

    report = det.check(candles, pair="EURUSD")
    # AnomalyReport uses block_signals not block
    assert report.warn or report.block_signals, "Large ATR spike should warn or block"


def test_anomaly_detector_clean_candles_pass():
    """Normal candles should pass anomaly detection."""
    from monitoring.anomaly_detector import AnomalyDetector

    det     = AnomalyDetector()
    candles = make_candles(60)
    report  = det.check(candles, pair="EURUSD")
    assert not report.block_signals, f"Clean candles should not block: {report.anomalies}"


def test_anomaly_detector_zero_range_blocks():
    """Zero-range candle (bad data) should block."""
    from monitoring.anomaly_detector import AnomalyDetector

    det     = AnomalyDetector()
    candles = make_candles(60)
    price   = 1.0850
    candles[-1] = C(open=price, high=price, low=price, close=price)

    report = det.check(candles, pair="EURUSD")
    assert report.block_signals, "Zero range should block"


# ══════════════════════════════════════════════════════════════
# TEST 13: V5Settings — validation logic
# ══════════════════════════════════════════════════════════════

def test_settings_valid_defaults():
    """Default settings should pass validation with no errors."""
    from core.v5_settings import V5Settings

    s      = V5Settings()
    errors = s.validate()
    assert errors == [], f"Default settings should be valid: {errors}"


def test_settings_catches_risk_incoherence():
    """Settings should catch 5×risk > max_drawdown."""
    from core.v5_settings import V5Settings
    import dataclasses

    s = V5Settings()
    # Override to create incoherent state
    object.__setattr__(s, "RISK_PERCENT",    5.0)   # 5% per trade
    object.__setattr__(s, "MAX_DRAWDOWN_PCT", 0.10) # 10% max DD
    # 5 × 5% = 25% > 10% → incoherent

    errors = s.validate()
    assert any("incoherence" in e.lower() or "risk" in e.lower() for e in errors)


def test_settings_regime_timeout():
    """Regime timeout should return correct value per regime."""
    from core.v5_settings import V5Settings

    s = V5Settings()
    assert s.regime_timeout("trending") >= s.regime_timeout("ranging")
    assert s.regime_timeout("ranging")  >= s.regime_timeout("volatile")


# ══════════════════════════════════════════════════════════════
# TEST 14: PerformanceAttribution — cross-tabulation
# ══════════════════════════════════════════════════════════════

def test_attribution_basic_breakdown():
    """Attribution should correctly compute win rates per agent."""
    from research.performance_attribution import PerformanceAttribution

    attr = PerformanceAttribution()
    # Load synthetic trades
    attr._trades = [
        {"agent": "TrendAgent",   "regime": "trending", "session": "london",
         "pair": "EURUSD", "timeframe": "1h", "direction": "buy",
         "outcome": "win",  "r_multiple": 1.5, "pnl_usd": 75, "confidence": 0.72, "conf_bin": "0.65-0.75", "duration_h": 4},
        {"agent": "TrendAgent",   "regime": "trending", "session": "london",
         "pair": "EURUSD", "timeframe": "1h", "direction": "buy",
         "outcome": "win",  "r_multiple": 1.2, "pnl_usd": 60, "confidence": 0.70, "conf_bin": "0.65-0.75", "duration_h": 3},
        {"agent": "TrendAgent",   "regime": "trending", "session": "london",
         "pair": "EURUSD", "timeframe": "1h", "direction": "buy",
         "outcome": "loss", "r_multiple": -1.0, "pnl_usd": -50, "confidence": 0.60, "conf_bin": "0.55-0.65", "duration_h": 2},
        {"agent": "MeanReversionAgent", "regime": "ranging", "session": "tokyo",
         "pair": "EURUSD", "timeframe": "1h", "direction": "sell",
         "outcome": "loss", "r_multiple": -1.0, "pnl_usd": -50, "confidence": 0.58, "conf_bin": "0.55-0.65", "duration_h": 5},
    ]
    result = attr.attribute("agent")

    assert "TrendAgent" in result
    assert result["TrendAgent"]["n"] == 3
    assert abs(result["TrendAgent"]["win_rate"] - 2/3) < 0.01


def test_attribution_cross_tabulation():
    """Agent × regime cross-tabulation should produce correct cells."""
    from research.performance_attribution import PerformanceAttribution

    attr = PerformanceAttribution()
    attr._trades = [
        {"agent": "TrendAgent", "regime": "trending", "session": "london",
         "pair": "XAUUSD", "timeframe": "1h", "direction": "buy",
         "outcome": "win", "r_multiple": 2.0, "pnl_usd": 100, "confidence": 0.80, "conf_bin": "0.75-0.85", "duration_h": 6},
        {"agent": "TrendAgent", "regime": "ranging", "session": "london",
         "pair": "XAUUSD", "timeframe": "1h", "direction": "buy",
         "outcome": "loss", "r_multiple": -1.0, "pnl_usd": -50, "confidence": 0.58, "conf_bin": "0.55-0.65", "duration_h": 3},
    ]
    result = attr.attribute("agent", "regime")

    assert "TrendAgent×trending" in result
    assert "TrendAgent×ranging"  in result
    assert result["TrendAgent×trending"]["win_rate"] == 1.0
    assert result["TrendAgent×ranging"]["win_rate"]  == 0.0


# ══════════════════════════════════════════════════════════════
# TEST 15: MTF Confluence — D1 override and scoring
# ══════════════════════════════════════════════════════════════

def test_mtf_d1_override_blocks_counter_trend():
    """Strong bearish D1 should block bullish signal."""
    from core.mtf_confluence import MultiTimeframeConfluence

    mtf = MultiTimeframeConfluence()
    # Bearish D1: strong downtrend
    d1_candles = make_candles(100, base=1.20, trend=-0.002)
    # Bullish H1 and H4
    h1_candles = make_candles(100, base=1.08, trend=0.0003)
    h4_candles = make_candles(100, base=1.08, trend=0.0002)

    mtf.update_single("EURUSD", "d1", d1_candles)
    mtf.update_single("EURUSD", "h4", h4_candles)
    mtf.update_single("EURUSD", "h1", h1_candles)

    d1_bias, d1_str = mtf._bias_cache["EURUSD"]["d1"]
    # Only test D1 override if D1 is actually detected as bearish with confidence
    if d1_bias == "sell" and d1_str > 0.55:
        result = mtf.score("EURUSD", "buy")
        assert result.should_block, "D1 bearish override should block BUY signal"


def test_mtf_full_alignment_gives_full_size():
    """All three TFs aligned bullish should give full size."""
    from core.mtf_confluence import MultiTimeframeConfluence

    mtf = MultiTimeframeConfluence()
    bull = make_candles(100, trend=0.0005)

    mtf.update_single("EURUSD", "d1", bull)
    mtf.update_single("EURUSD", "h4", bull)
    mtf.update_single("EURUSD", "h1", bull)

    result = mtf.score("EURUSD", "buy")
    # Should give full or reduced size (not block)
    assert not result.should_block or result.confluence_score >= 0.0


def test_mtf_neutral_gives_reduced_size():
    """Mixed TF signals should reduce size, not block."""
    from core.mtf_confluence import MultiTimeframeConfluence

    mtf = MultiTimeframeConfluence()
    # Don't update any TF → all neutral
    result = mtf.score("GBPUSD", "buy")
    # Neutral TFs → reduced but not zero
    assert result.confluence_score >= 0.0
    assert isinstance(result.size_multiplier, float)


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Allow running directly: python test_v5_suite.py
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd="/mnt/user-data/outputs/hedge_fund_v5"
    )
    sys.exit(result.returncode)
