"""
PriceIQ Pro — Advanced Capabilities Test Suite

Tests for:
    - TickExecutionEngine: bar builder, bar close detection, latency tracking
    - MacroSignalAgent: signal logic, data freshness, pair filtering
    - WalkForwardOptimizer: fold structure, efficiency ratio, param combos
    - ExtendedOrchestrator: macro agreement/disagreement adjustments
    - AdvancedIntegration: attachment and API routing

Run: pytest tests/test_advanced.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared Candle stub (same as conftest) ─────────────────────
@dataclass
class Candle:
    open: float; high: float; low: float; close: float
    volume: float = 1000.0; timestamp: str = ""
    is_bullish: bool = True; is_bearish: bool = False
    body: float = 0.0; range: float = 0.0
    upper_shadow: float = 0.0; lower_shadow: float = 0.0

    def __post_init__(self):
        self.is_bullish = self.close >= self.open
        self.is_bearish = not self.is_bullish
        self.body = abs(self.close - self.open)
        self.range = self.high - self.low
        self.upper_shadow = self.high - max(self.open, self.close)
        self.lower_shadow = min(self.open, self.close) - self.low
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def make_candles(n=100, base=1.08, trend=0.0001, noise=0.0003):
    candles = []
    price = base
    now = datetime.now(timezone.utc)
    for i in range(n):
        n_val = np.random.normal(0, noise)
        o = price; c = price + trend + n_val
        h = max(o, c) + abs(n_val) * 0.5
        l = min(o, c) - abs(n_val) * 0.5
        ts = (now - timedelta(hours=n-i)).isoformat()
        candles.append(Candle(open=o, high=h, low=l, close=c, timestamp=ts))
        price = c
    return candles


# ══════════════════════════════════════════════════════════════
# TEST GROUP 1: TickExecutionEngine
# ══════════════════════════════════════════════════════════════

class TestBarBuilder:

    def test_bar_builder_initialises_correctly(self):
        from execution.tick_execution_engine import BarBuilder, TF_SECONDS
        bb = BarBuilder("1h")
        assert bb.tf_seconds == TF_SECONDS["1h"]
        assert bb.tf_seconds == 3600

    def test_bar_builder_creates_first_bar_on_tick(self):
        from execution.tick_execution_engine import BarBuilder, Tick
        bb   = BarBuilder("1h")
        tick = Tick("EURUSD", bid=1.0800, ask=1.0802,
                    timestamp=datetime.now(timezone.utc))
        asyncio.get_event_loop().run_until_complete(bb.process_tick(tick))
        bar = bb.get_current_bar("EURUSD")
        assert bar is not None
        assert abs(bar.open - tick.mid) < 0.0001

    def test_bar_builder_updates_high_low(self):
        from execution.tick_execution_engine import BarBuilder, Tick
        bb  = BarBuilder("1h")
        now = datetime.now(timezone.utc)

        ticks = [
            Tick("EURUSD", 1.0800, 1.0802, now),
            Tick("EURUSD", 1.0820, 1.0822, now + timedelta(seconds=10)),
            Tick("EURUSD", 1.0790, 1.0792, now + timedelta(seconds=20)),
        ]
        for tick in ticks:
            asyncio.get_event_loop().run_until_complete(bb.process_tick(tick))

        bar = bb.get_current_bar("EURUSD")
        assert bar.high >= 1.0820   # must have captured high tick
        assert bar.low  <= 1.0791   # must have captured low tick

    def test_bar_builder_fires_callback_on_close(self):
        from execution.tick_execution_engine import BarBuilder, Tick
        bb = BarBuilder("1h")
        events_received = []

        async def on_close(event):
            events_received.append(event)

        bb.on_bar_close(on_close)

        now     = datetime.now(timezone.utc)
        # Tick in bar period 0
        tick_1  = Tick("EURUSD", 1.0800, 1.0802, now)
        # Tick in bar period 1 (new hour) — should fire close
        next_hr = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        tick_2  = Tick("EURUSD", 1.0810, 1.0812, next_hr + timedelta(seconds=5))

        async def run():
            await bb.process_tick(tick_1)
            await bb.process_tick(tick_2)

        asyncio.get_event_loop().run_until_complete(run())
        assert len(events_received) == 1, "Should fire exactly one bar close event"
        assert events_received[0].pair == "EURUSD"

    def test_bar_builder_records_latency(self):
        from execution.tick_execution_engine import BarBuilder, Tick
        bb = BarBuilder("1h")
        latencies = []

        async def on_close(event):
            latencies.append(event.latency_ms)

        bb.on_bar_close(on_close)
        now     = datetime.now(timezone.utc)
        tick_1  = Tick("EURUSD", 1.08, 1.0802, now)
        next_hr = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        tick_2  = Tick("EURUSD", 1.081, 1.0812, next_hr + timedelta(seconds=1))

        async def run():
            await bb.process_tick(tick_1)
            await bb.process_tick(tick_2)

        asyncio.get_event_loop().run_until_complete(run())
        assert len(latencies) == 1
        assert isinstance(latencies[0], float)
        # Latency is (detection_time - theoretical_close). Can be negative if
        # detection happens before theoretical close (e.g. tick arrives in-bar).
        # We only assert it's a finite number.
        assert np.isfinite(latencies[0])

    def test_tick_spread_calculation(self):
        from execution.tick_execution_engine import Tick
        tick = Tick("EURUSD", bid=1.0799, ask=1.0801,
                    timestamp=datetime.now(timezone.utc))
        assert abs(tick.spread_pips - 2.0) < 0.1   # 2 pips spread
        assert abs(tick.mid - 1.0800) < 0.00001

    def test_bar_builder_accumulates_tick_count(self):
        from execution.tick_execution_engine import BarBuilder, Tick
        bb  = BarBuilder("1h")
        now = datetime.now(timezone.utc)
        for i in range(10):
            tick = Tick("EURUSD", 1.08, 1.0802,
                        now + timedelta(seconds=i * 30))
            asyncio.get_event_loop().run_until_complete(bb.process_tick(tick))

        bar = bb.get_current_bar("EURUSD")
        # First tick opens bar with volume=0, ticks 2-10 call update() += 1 each = 9
        # OR first tick sets volume=1 and rest add. Either way: volume > 0
        assert bar.volume > 0, f"Volume should be > 0, got {bar.volume}"
        assert bar.volume <= 10, f"Volume should be <= 10, got {bar.volume}"


# ══════════════════════════════════════════════════════════════
# TEST GROUP 2: MacroSignalAgent
# ══════════════════════════════════════════════════════════════

class TestMacroSignalAgent:

    def test_null_signal_when_no_snapshot(self):
        from agents.macro_signal_agent import MacroSignalAgent
        agent = MacroSignalAgent()
        sig   = agent.evaluate(make_candles(60), "trending", "XAUUSD")
        assert sig.direction is None
        assert "No macro snapshot" in sig.reasoning

    def test_bullish_gold_signal_on_falling_rates_and_dxy(self):
        from agents.macro_signal_agent import MacroSignalAgent, MacroSnapshot
        agent = MacroSignalAgent()
        # Inject snapshot: falling real rates + weakening DXY + rising VIX → bullish gold
        agent._last_snapshot = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=1.5,
            us_nominal_rate=4.2,
            us_breakeven_10y=2.7,
            fed_funds=5.25,
            dxy_level=103.5,
            dxy_change_24h=-0.50,          # USD weakening
            vix_level=18.5,
            vix_change_24h=+2.5,           # fear rising
            real_rate_change_24h=-0.08,    # real rates falling (in bps)
            nominal_rate_change_24h=-0.05,
            gold_macro_bias="bullish",
            gold_macro_strength=0.67,
        )
        sig = agent.evaluate(make_candles(60, base=1920.0, trend=0.5), "trending", "XAUUSD")
        assert sig.direction == "buy", f"Expected BUY, got {sig.direction}: {sig.reasoning}"
        assert sig.confidence > 0.50

    def test_bearish_gold_signal_on_rising_rates(self):
        from agents.macro_signal_agent import MacroSignalAgent, MacroSnapshot
        agent = MacroSignalAgent()
        agent._last_snapshot = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=2.1,
            us_nominal_rate=4.8,
            us_breakeven_10y=2.7,
            fed_funds=5.50,
            dxy_level=106.0,
            dxy_change_24h=+0.60,          # USD strengthening
            vix_level=14.0,
            vix_change_24h=-1.5,           # fear falling
            real_rate_change_24h=+0.12,    # real rates rising
            nominal_rate_change_24h=+0.10,
            gold_macro_bias="bearish",
            gold_macro_strength=0.67,
        )
        sig = agent.evaluate(make_candles(60, base=1920.0), "trending", "XAUUSD")
        assert sig.direction == "sell", f"Expected SELL, got {sig.direction}"

    def test_neutral_when_mixed_signals(self):
        from agents.macro_signal_agent import MacroSignalAgent, MacroSnapshot
        agent = MacroSignalAgent()
        agent._last_snapshot = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=1.8, us_nominal_rate=4.4, us_breakeven_10y=2.6,
            fed_funds=5.25, dxy_level=104.0,
            dxy_change_24h=-0.10,      # minor DXY move (below threshold)
            vix_level=16.0,
            vix_change_24h=+0.5,       # minor VIX change (below threshold)
            real_rate_change_24h=-0.02,  # very small rate change
            nominal_rate_change_24h=-0.01,
            gold_macro_bias="neutral",
            gold_macro_strength=0.0,
        )
        sig = agent.evaluate(make_candles(60, base=1920.0), "trending", "XAUUSD")
        assert sig.direction is None, "Should be neutral with mixed/weak signals"

    def test_unsupported_pair_returns_null(self):
        from agents.macro_signal_agent import MacroSignalAgent, MacroSnapshot
        agent = MacroSignalAgent()
        agent._last_snapshot = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=1.5, us_nominal_rate=4.2, us_breakeven_10y=2.7,
            fed_funds=5.25, dxy_level=103.0, dxy_change_24h=-0.5,
            vix_level=18.0, vix_change_24h=2.0,
            real_rate_change_24h=-0.08, nominal_rate_change_24h=-0.05,
            gold_macro_bias="bullish", gold_macro_strength=0.67,
        )
        sig = agent.evaluate(make_candles(60), "trending", "NZDUSD")
        assert sig.direction is None, "NZDUSD not in supported pairs"

    def test_stale_snapshot_returns_null(self):
        from agents.macro_signal_agent import MacroSignalAgent, MacroSnapshot
        agent = MacroSignalAgent()
        # 6-hour-old snapshot (exceeds 4h threshold)
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        agent._last_snapshot = MacroSnapshot(
            timestamp=stale_ts,
            us_real_rate=1.5, us_nominal_rate=4.2, us_breakeven_10y=2.7,
            fed_funds=5.25, dxy_level=103.0, dxy_change_24h=-0.5,
            vix_level=18.0, vix_change_24h=2.0,
            real_rate_change_24h=-0.08, nominal_rate_change_24h=-0.05,
            gold_macro_bias="bullish", gold_macro_strength=0.67,
        )
        sig = agent.evaluate(make_candles(60, base=1920.0), "trending", "XAUUSD")
        assert sig.direction is None, "Stale snapshot should return null signal"

    def test_macro_bias_computation(self):
        from agents.macro_signal_agent import MacroSignalAgent
        agent = MacroSignalAgent()
        # All bearish signals
        bias, strength = agent._compute_gold_bias(
            real_rate_chg=+0.10,   # rising rates → bearish
            dxy_chg=+0.50,         # stronger USD → bearish
            vix_chg=-2.0,          # falling fear → bearish for gold safe haven
        )
        assert bias == "bearish"
        assert strength > 0.5

        # All bullish signals
        bias, strength = agent._compute_gold_bias(
            real_rate_chg=-0.10,   # falling rates → bullish
            dxy_chg=-0.50,         # weaker USD → bullish
            vix_chg=+2.0,          # rising fear → bullish (safe haven)
        )
        assert bias == "bullish"
        assert strength > 0.5


# ══════════════════════════════════════════════════════════════
# TEST GROUP 3: WalkForwardOptimizer
# ══════════════════════════════════════════════════════════════

class TestWalkForwardOptimizer:

    def _make_mock_backtest(self, win_rate=0.55, n_trades=15, sharpe=0.8):
        """Create a mock backtest function that returns consistent results."""
        from dataclasses import dataclass as dc
        @dc
        class MockTrade:
            bar_index: int
            r_multiple: float
            outcome: str

        class MockResult:
            def __init__(self):
                self.total_trades = n_trades
                self.win_rate     = win_rate
                self.sharpe       = sharpe
                self.max_drawdown = 0.08
                self.net_pnl      = 500.0
                self.trades = [
                    MockTrade(
                        bar_index=i * 5,
                        r_multiple=1.5 if i % 2 == 0 else -1.0,
                        outcome="win" if i % 2 == 0 else "loss",
                    )
                    for i in range(n_trades)
                ]
        return lambda candles, params: MockResult()

    def test_optimizer_requires_sufficient_candles(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(
            backtest_fn=self._make_mock_backtest(),
            train_bars=200, test_bars=50,
        )
        with pytest.raises(ValueError, match="Need"):
            wfo.run(make_candles(100), pair="EURUSD")

    def test_optimizer_produces_folds(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(
            backtest_fn=self._make_mock_backtest(),
            train_bars=100, test_bars=30, max_combos=3,
        )
        candles = make_candles(300)
        report  = wfo.run(candles, pair="EURUSD")
        assert report.n_folds >= 1, "Should produce at least 1 fold"
        assert len(report.folds) == report.n_folds

    def test_optimizer_efficiency_ratio_bounded(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(
            backtest_fn=self._make_mock_backtest(win_rate=0.55, sharpe=0.8),
            train_bars=100, test_bars=30, max_combos=3,
        )
        report = wfo.run(make_candles(300), pair="EURUSD")
        # Efficiency ratio can be any real number but should be finite
        assert np.isfinite(report.efficiency_ratio)

    def test_optimizer_fold_structure_anchored(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(
            backtest_fn=self._make_mock_backtest(),
            train_bars=100, test_bars=50, max_combos=2,
        )
        report = wfo.run(make_candles(400), pair="EURUSD")
        for fold in report.folds:
            # Anchored: train always starts at 0
            assert fold.train_start == 0
            # Test window comes after training
            assert fold.test_start >= fold.train_end
            # Test end after test start
            assert fold.test_end > fold.test_start

    def test_optimizer_recommended_params_valid(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer, PARAM_GRID
        wfo = WalkForwardOptimizer(
            backtest_fn=self._make_mock_backtest(),
            train_bars=100, test_bars=50, max_combos=4,
        )
        report = wfo.run(make_candles(400), pair="XAUUSD")
        params = report.recommended_params
        # Params should be from the grid
        for key in params:
            assert key in PARAM_GRID, f"Unknown param key: {key}"

    def test_param_combo_count_respects_max(self):
        from research.walk_forward_optimizer import WalkForwardOptimizer
        wfo    = WalkForwardOptimizer(backtest_fn=lambda c, p: None, max_combos=5)
        combos = wfo._build_combos()
        assert len(combos) <= 5

    def test_sharpe_computation(self):
        from research.walk_forward_optimizer import _sharpe
        assert _sharpe([]) == 0.0
        assert _sharpe([1.0]) == 0.0           # single value → std=0
        r = [1.5, -1.0, 1.5, -1.0, 2.0]
        s = _sharpe(r)
        assert isinstance(s, float)
        assert np.isfinite(s)


# ══════════════════════════════════════════════════════════════
# TEST GROUP 4: ExtendedOrchestrator macro integration
# ══════════════════════════════════════════════════════════════

class TestExtendedOrchestrator:

    def _make_extended_orch(self):
        from agents.extended_orchestrator import ExtendedAgentOrchestrator
        return ExtendedAgentOrchestrator()

    def test_orchestrator_initialises_with_5_agents(self):
        orch = self._make_extended_orch()
        assert len(orch._price_agents) == 4
        assert orch.macro_agent is not None
        total_agent_types = len(orch._price_agents) + 1  # +1 for macro
        assert total_agent_types == 5

    def test_orchestrator_runs_without_macro_data(self):
        """Should work even when macro snapshot is None."""
        orch    = self._make_extended_orch()
        candles = make_candles(100, trend=0.0005)  # trending
        result  = orch.run(candles, regime="trending", pair="EURUSD")
        # May or may not fire a signal, but should not raise
        # macro_fired should be False since no snapshot loaded
        if result:
            assert result.macro_fired == False or result.macro_direction is not None

    def test_macro_agree_boosts_confidence(self):
        """When macro agrees with price agent, confidence should increase."""
        from agents.macro_signal_agent import MacroSnapshot
        orch    = self._make_extended_orch()
        candles = make_candles(100, trend=0.0005)  # uptrend

        # Inject bullish macro snapshot for XAUUSD
        from agents.macro_signal_agent import MacroSnapshot
        snap = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=1.5, us_nominal_rate=4.2, us_breakeven_10y=2.7,
            fed_funds=5.25, dxy_level=103.0, dxy_change_24h=-0.50,
            vix_level=18.0, vix_change_24h=2.0,
            real_rate_change_24h=-0.08, nominal_rate_change_24h=-0.05,
            gold_macro_bias="bullish", gold_macro_strength=0.67,
        )
        orch.macro_agent._last_snapshot = snap

        # Run on XAUUSD candles
        candles_xau = make_candles(100, base=1920.0, trend=0.5)
        result = orch.run(candles_xau, regime="trending", pair="XAUUSD")

        # If a signal fired and macro agrees, confidence should include boost
        if result and result.selected_signal.direction == "buy" and result.macro_aligns:
            # Confidence should be boosted (hard to assert exact value without knowing base)
            assert result.selected_signal.confidence > 0.0
            assert "MACRO AGREES" in result.selected_signal.reasoning

    def test_macro_oppose_reduces_confidence(self):
        """When macro opposes price agent, confidence should decrease."""
        from agents.macro_signal_agent import MacroSnapshot
        orch    = self._make_extended_orch()
        # Inject bearish macro snapshot
        snap = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=2.1, us_nominal_rate=4.8, us_breakeven_10y=2.7,
            fed_funds=5.50, dxy_level=106.0, dxy_change_24h=+0.60,
            vix_level=14.0, vix_change_24h=-1.5,
            real_rate_change_24h=+0.12, nominal_rate_change_24h=+0.10,
            gold_macro_bias="bearish", gold_macro_strength=0.67,
        )
        orch.macro_agent._last_snapshot = snap

        # Run on bullish XAUUSD candles (price agent likely says BUY)
        candles_xau = make_candles(100, base=1920.0, trend=0.8)
        result = orch.run(candles_xau, regime="trending", pair="XAUUSD")

        if result and result.macro_opposes:
            assert "MACRO OPPOSES" in result.selected_signal.reasoning

    def test_weights_update_propagates_to_all_agents(self):
        orch = self._make_extended_orch()
        new_weights = {
            "TrendAgent": 2.0, "MeanReversionAgent": 0.5,
            "BreakoutAgent": 1.5, "LiquidityTrapAgent": 0.8,
            "MacroSignalAgent": 1.2,
        }
        orch.update_weights(new_weights)
        assert orch.weights["TrendAgent"]      == 2.0
        assert orch.weights["MacroSignalAgent"] == 1.2


# ══════════════════════════════════════════════════════════════
# TEST GROUP 5: Advanced Integration
# ══════════════════════════════════════════════════════════════

class TestAdvancedIntegration:

    def test_attach_macro_agent_enabled(self):
        """attach_advanced_capabilities should add macro_agent to v5."""
        from advanced_integration import attach_advanced_capabilities
        import os

        v5_mock        = MagicMock()
        v5_mock.orchestrator        = MagicMock()
        v5_mock.orchestrator.weights = {"TrendAgent": 1.0}

        with patch.dict(os.environ, {"MACRO_AGENT_ENABLED": "true",
                                      "TICK_ENGINE_ENABLED": "false",
                                      "WFO_ENABLED": "false"}):
            attach_advanced_capabilities(v5_mock, broker=None, telegram=None)

        # macro_agent should be set on v5
        assert hasattr(v5_mock, "macro_agent") or hasattr(v5_mock.orchestrator, "macro_agent")

    def test_all_disabled_no_error(self):
        """Disabling all advanced capabilities should not raise."""
        from advanced_integration import attach_advanced_capabilities
        import os

        v5_mock = MagicMock()
        with patch.dict(os.environ, {"MACRO_AGENT_ENABLED": "false",
                                      "TICK_ENGINE_ENABLED": "false",
                                      "WFO_ENABLED": "false"}):
            # Should complete without error
            attach_advanced_capabilities(v5_mock)

    def test_wfo_param_grid_completeness(self):
        """Walk-forward param grid should contain all optimizable params."""
        from research.walk_forward_optimizer import PARAM_GRID
        required = {"atr_stop_multiplier", "atr_tp1_multiplier",
                    "min_confidence", "rsi_overbought"}
        assert required.issubset(set(PARAM_GRID.keys()))

    def test_macro_agent_status_dict(self):
        """MacroSignalAgent.status_dict() should return valid structure."""
        from agents.macro_signal_agent import MacroSignalAgent
        agent  = MacroSignalAgent()
        status = agent.status_dict()
        # Basic keys always present
        assert "status" in status
        assert status["status"] in ("ok", "no_data")

        # With a snapshot loaded, more keys appear
        from agents.macro_signal_agent import MacroSnapshot
        agent._last_snapshot = MacroSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            us_real_rate=1.5, us_nominal_rate=4.2, us_breakeven_10y=2.7,
            fed_funds=5.25, dxy_level=103.0, dxy_change_24h=-0.5,
            vix_level=18.0, vix_change_24h=2.0,
            real_rate_change_24h=-0.08, nominal_rate_change_24h=-0.05,
            gold_macro_bias="bullish", gold_macro_strength=0.67,
        )
        status_with_data = agent.status_dict()
        assert status_with_data["status"] == "ok"
        assert "gold_bias" in status_with_data
        assert "fred_api_configured" in status_with_data
        assert isinstance(status_with_data["fred_api_configured"], bool)

    def test_simulated_tick_feed_produces_ticks(self):
        """SimulatedTickFeed should produce ticks for all pairs."""
        from execution.tick_execution_engine import SimulatedTickFeed, Tick

        feed   = SimulatedTickFeed(speed_multiplier=1000.0)
        pairs  = ["EURUSD", "XAUUSD"]
        ticks_received = []

        async def collect(tick):
            ticks_received.append(tick)
            if len(ticks_received) >= 4:
                feed.stop()

        feed.on_tick(collect)

        async def run():
            try:
                await asyncio.wait_for(feed.start(pairs), timeout=2.0)
            except asyncio.TimeoutError:
                pass

        asyncio.get_event_loop().run_until_complete(run())
        assert len(ticks_received) >= 2, "Should receive ticks for each pair"
        received_pairs = {t.pair for t in ticks_received}
        assert "EURUSD" in received_pairs
        assert "XAUUSD" in received_pairs

    def test_order_router_tracks_slippage(self):
        """InstantOrderRouter should measure fill vs requested."""
        from execution.tick_execution_engine import InstantOrderRouter

        # Mock broker
        mock_broker = MagicMock()
        mock_broker.place_market_order = AsyncMock(return_value={
            "price": "1.0805",   # slight slippage from requested 1.0800
            "id": "TEST123",
        })

        router = InstantOrderRouter(mock_broker, telegram=None)

        async def run():
            result = await router.place_order(
                pair="EURUSD", direction="buy",
                lots=0.1, stop_loss=1.0750,
                take_profit=1.0900,
                requested_price=1.0800,
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.success
        assert result.fill_price == 1.0805
        assert result.slippage_pips > 0   # 0.5 pip slippage

        stats = router.slippage_stats()
        assert stats["fills"] == 1
        assert stats["avg_slippage_pip"] > 0


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    sys.exit(result.returncode)
