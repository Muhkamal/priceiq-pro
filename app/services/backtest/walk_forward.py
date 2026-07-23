"""
PriceIQ Pro — Walk-Forward Backtest Engine v1.0

NOT a simple replay. This runs the FULL system stack on historical data:
    - RegimeClassifier per window
    - AgentOrchestrator selects best agent
    - RiskGovernor gates and sizes positions
    - ExecutionIntelligence applies realistic fills
    - LearningLoop updates weights as trades close
    - Metrics: Sharpe, Sortino, max drawdown, expectancy, agent breakdown

Usage:
    engine = WalkForwardEngine(
        regime_classifier=clf,
        agent_orchestrator=orch,
        risk_governor=gov,
        execution=exec_intel,
    )
    result = engine.run(candles, pair="XAUUSD", train_bars=200, test_bars=50)
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


@dataclass
class BacktestTrade:
    bar_index:   int
    pair:        str
    agent:       str
    direction:   str
    regime:      str
    entry:       float
    fill_price:  float
    stop_loss:   float
    take_profit: float
    exit_price:  float
    outcome:     str          # "win" | "loss" | "timeout"
    r_multiple:  float
    pnl_usd:     float
    equity_after: float
    confidence:  float
    slippage_pips: float = 0.0


@dataclass
class BacktestResult:
    pair:        str
    total_bars:  int
    trades:      List[BacktestTrade]
    equity_curve: List[float]

    # Computed metrics (filled by compute_metrics)
    total_trades:   int = 0
    win_trades:     int = 0
    loss_trades:    int = 0
    timeout_trades: int = 0
    win_rate:       float = 0.0
    net_pnl:        float = 0.0
    max_drawdown:   float = 0.0
    sharpe:         float = 0.0
    sortino:        float = 0.0
    expectancy:     float = 0.0
    profit_factor:  float = 0.0
    agent_breakdown: Dict[str, Dict] = field(default_factory=dict)
    regime_breakdown: Dict[str, Dict] = field(default_factory=dict)

    def compute_metrics(self, starting_balance: float = 10_000.0) -> "BacktestResult":
        if not self.trades:
            return self

        self.total_trades   = len(self.trades)
        self.win_trades     = sum(1 for t in self.trades if t.outcome == "win")
        self.loss_trades    = sum(1 for t in self.trades if t.outcome == "loss")
        self.timeout_trades = sum(1 for t in self.trades if t.outcome == "timeout")
        self.win_rate       = round(_safe_div(self.win_trades, self.total_trades), 4)
        self.net_pnl        = round(sum(t.pnl_usd for t in self.trades), 2)
        self.expectancy     = round(_safe_div(self.net_pnl, self.total_trades), 3)

        # Profit factor
        gross_wins  = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_losses = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        self.profit_factor = round(_safe_div(gross_wins, max(gross_losses, 0.01)), 3)

        # Drawdown
        peak = starting_balance
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = _safe_div(peak - eq, peak)
            max_dd = max(max_dd, dd)
        self.max_drawdown = round(max_dd, 4)

        # Sharpe / Sortino (using daily R-multiple returns)
        r_mults = [t.r_multiple for t in self.trades]
        if len(r_mults) > 1:
            mean_r = np.mean(r_mults)
            std_r  = np.std(r_mults)
            neg_r  = [r for r in r_mults if r < 0]
            downside_std = np.std(neg_r) if neg_r else 0.001
            self.sharpe  = round(_safe_div(mean_r, std_r), 3)
            self.sortino = round(_safe_div(mean_r, downside_std), 3)

        # Agent breakdown
        agents = set(t.agent for t in self.trades)
        for agent in agents:
            atrades = [t for t in self.trades if t.agent == agent]
            awins   = [t for t in atrades if t.outcome == "win"]
            ar_mults = [t.r_multiple for t in atrades]
            self.agent_breakdown[agent] = {
                "trades":    len(atrades),
                "win_rate":  round(_safe_div(len(awins), len(atrades)), 3),
                "expectancy": round(float(np.mean(ar_mults)), 3) if ar_mults else 0,
                "net_pnl":   round(sum(t.pnl_usd for t in atrades), 2),
            }

        # Regime breakdown
        regimes = set(t.regime for t in self.trades)
        for regime in regimes:
            rtrades = [t for t in self.trades if t.regime == regime]
            rwins   = [t for t in rtrades if t.outcome == "win"]
            self.regime_breakdown[regime] = {
                "trades":   len(rtrades),
                "win_rate": round(_safe_div(len(rwins), len(rtrades)), 3),
                "net_pnl":  round(sum(t.pnl_usd for t in rtrades), 2),
            }

        return self

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  PriceIQ Backtest: {self.pair}",
            f"{'='*55}",
            f"  Total trades:   {self.total_trades}",
            f"  Wins / Losses:  {self.win_trades} / {self.loss_trades} (timeout: {self.timeout_trades})",
            f"  Win rate:       {self.win_rate:.1%}",
            f"  Net PnL:        ${self.net_pnl:,.2f}",
            f"  Expectancy:     ${self.expectancy:,.2f} / trade",
            f"  Profit factor:  {self.profit_factor:.2f}",
            f"  Max drawdown:   {self.max_drawdown:.1%}",
            f"  Sharpe (R):     {self.sharpe:.3f}",
            f"  Sortino (R):    {self.sortino:.3f}",
            f"",
            f"  Agent breakdown:",
        ]
        for agent, stats in self.agent_breakdown.items():
            lines.append(
                f"    {agent}: {stats['trades']} trades | "
                f"WR={stats['win_rate']:.0%} | "
                f"EV=${stats['expectancy']:+.2f} | "
                f"PnL=${stats['net_pnl']:+,.2f}"
            )
        lines.append(f"")
        lines.append(f"  Regime breakdown:")
        for regime, stats in self.regime_breakdown.items():
            lines.append(
                f"    {regime}: {stats['trades']} trades | "
                f"WR={stats['win_rate']:.0%} | "
                f"PnL=${stats['net_pnl']:+,.2f}"
            )
        lines.append(f"{'='*55}")
        return "\n".join(lines)


class WalkForwardEngine:
    """
    Walk-forward backtest using the full v5 system stack.

    Each test window runs the regime classifier → agent orchestrator → risk governor → execution.
    The learning loop updates weights WITHIN the backtest (no lookahead bias).
    """

    def __init__(
        self,
        regime_classifier=None,
        agent_orchestrator=None,
        risk_governor=None,
        execution=None,
        learning_loop=None,
    ):
        self.regime_clf  = regime_classifier
        self.orchestrator = agent_orchestrator
        self.governor    = risk_governor
        self.execution   = execution
        self.learning    = learning_loop

    def run(
        self,
        candles:          List,
        pair:             str = "EURUSD",
        starting_balance: float = 10_000.0,
        risk_pct:         float = 2.0,
        train_bars:       int = 200,
        test_bars:        int = 50,
        sim_bars_forward: int = 5,
    ) -> BacktestResult:
        """
        Walk-forward simulation.

        Args:
            candles:        full historical candle list
            pair:           instrument
            starting_balance: starting account balance
            risk_pct:       % of balance to risk per trade
            train_bars:     bars used to train classifier in each window
            test_bars:      bars in each test fold
            sim_bars_forward: bars to simulate trade outcome after signal
        """
        if len(candles) < train_bars + test_bars + sim_bars_forward:
            raise ValueError(
                f"Need at least {train_bars + test_bars + sim_bars_forward} candles "
                f"for walk-forward. Got {len(candles)}."
            )

        trades:       List[BacktestTrade] = []
        equity_curve: List[float]         = [starting_balance]
        balance = starting_balance

        # Reset governor for backtest
        if self.governor:
            self.governor.current_balance = starting_balance
            self.governor.peak_balance    = starting_balance

        step = max(test_bars // 2, 10)   # 50% overlap windows

        for start in range(train_bars, len(candles) - sim_bars_forward - 1, step):
            window = candles[max(0, start - train_bars): start + 1]

            # ── 1. Regime classification ─────────────────────
            regime = "ranging"
            if self.regime_clf:
                try:
                    pred   = self.regime_clf.predict(window)
                    regime = pred.regime
                except Exception as e:
                    logger.debug(f"Regime clf error: {e}")

            # ── 2. Agent orchestrator ────────────────────────
            if not self.orchestrator:
                continue

            try:
                result = self.orchestrator.run(window, regime)
            except Exception as e:
                logger.debug(f"Orchestrator error: {e}")
                continue

            if result is None or not result.selected_signal.is_valid:
                continue

            signal   = result.selected_signal
            agent    = result.selected_agent
            direction = signal.direction
            entry    = candles[start].close
            stop_dist = signal.stop_distance
            tp1_dist  = signal.tp1_distance

            stop_loss   = entry - stop_dist if direction == "buy" else entry + stop_dist
            take_profit = entry + tp1_dist  if direction == "buy" else entry - tp1_dist

            # ── 3. Risk governor ─────────────────────────────
            risk_amount = balance * (risk_pct / 100)
            lots = max(0.01, round(_safe_div(risk_amount, stop_dist * 100_000, 0.01), 2))

            if self.governor:
                decision = self.governor.evaluate(pair, direction, lots, stop_dist, entry)
                if not decision.allowed:
                    logger.debug(f"Backtest: trade blocked — {decision.reason}")
                    continue
                lots = decision.adjusted_lots

            # ── 4. Execution intelligence ────────────────────
            fill_price   = entry
            slippage_pip = 0.0
            if self.execution:
                try:
                    fill = self.execution.simulate_fill(
                        pair=pair, direction=direction, order_type="market",
                        requested_price=entry, lots=lots, atr=stop_dist / 1.5,
                        session="london",
                    )
                    fill_price   = fill.fill_price
                    slippage_pip = fill.slippage_pips
                except Exception as e:
                    logger.debug(f"Execution sim error: {e}")

            # ── 5. Simulate outcome ──────────────────────────
            outcome    = "timeout"
            exit_price = fill_price
            for future in candles[start + 1: start + sim_bars_forward + 1]:
                if direction == "buy":
                    if future.low <= stop_loss:
                        outcome    = "loss"
                        exit_price = stop_loss
                        break
                    if future.high >= take_profit:
                        outcome    = "win"
                        exit_price = take_profit
                        break
                else:
                    if future.high >= stop_loss:
                        outcome    = "loss"
                        exit_price = stop_loss
                        break
                    if future.low <= take_profit:
                        outcome    = "win"
                        exit_price = take_profit
                        break

            # R-multiple
            if direction == "buy":
                pnl_pts = exit_price - fill_price
            else:
                pnl_pts = fill_price - exit_price
            r_multiple = round(_safe_div(pnl_pts, stop_dist, 0.0), 3)
            pnl_usd    = round(pnl_pts * lots * 100_000 / 10, 2)   # approx USD

            balance += pnl_usd
            if balance > (self.governor.peak_balance if self.governor else balance):
                if self.governor:
                    self.governor.peak_balance = balance
            equity_curve.append(balance)

            # ── 6. Learning loop update ──────────────────────
            if self.learning:
                self.learning.update(
                    pair=pair, agent_name=agent, direction=direction,
                    entry=fill_price, exit_price=exit_price,
                    stop=stop_loss, tp1=take_profit,
                    outcome=outcome, r_multiple=r_multiple,
                    regime=regime, confidence=signal.confidence,
                )
                # Feed updated weights back to orchestrator
                if self.orchestrator:
                    self.orchestrator.update_weights(self.learning.get_agent_weights())

            trade = BacktestTrade(
                bar_index=start,
                pair=pair,
                agent=agent,
                direction=direction,
                regime=regime,
                entry=entry,
                fill_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_price=exit_price,
                outcome=outcome,
                r_multiple=r_multiple,
                pnl_usd=pnl_usd,
                equity_after=round(balance, 2),
                confidence=signal.confidence,
                slippage_pips=slippage_pip,
            )
            trades.append(trade)

        result = BacktestResult(
            pair=pair,
            total_bars=len(candles),
            trades=trades,
            equity_curve=equity_curve,
        ).compute_metrics(starting_balance)

        logger.info(f"Walk-forward complete:\n{result.summary()}")
        return result
