"""
PriceIQ Pro — Parameter Sensitivity & Monte Carlo Research Engine v1.0

Two tools in one file:

1. ParameterSensitivityTester
   Sweeps key hyperparameters across a grid and measures how much
   performance degrades. Overfit systems peak sharply; robust systems
   are flat across the grid.

   Parameters swept:
       atr_stop_multiplier: [1.0, 1.25, 1.5, 1.75, 2.0]
       atr_tp_multiplier:   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
       min_confidence:      [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
       rsi_overbought:      [65, 68, 70, 72, 75]

2. MonteCarloEquityCurve
   Bootstrap-resamples the trade sequence N times.
   Produces: median final equity, 5th/95th percentile, max DD distribution,
   probability of ruin (equity drops below 50% of start).

Usage:
    # Sensitivity test:
    tester = ParameterSensitivityTester(backtest_fn=your_backtest)
    report = tester.run(candles, pair="XAUUSD", n_jobs=1)
    print(report.stability_score)   # 0–1, higher = more robust

    # Monte Carlo:
    mc = MonteCarloEquityCurve(trades=backtest_result.trades)
    report = mc.run(n_paths=1000, starting_balance=10_000)
    print(report.median_final)
    print(report.prob_ruin)
"""

from __future__ import annotations

import logging
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 1. PARAMETER SENSITIVITY TESTER
# ════════════════════════════════════════════════════════════

PARAM_GRID = {
    "atr_stop_multiplier": [1.0, 1.25, 1.5, 1.75, 2.0],
    "atr_tp1_multiplier":  [1.5, 2.0, 2.5, 3.0, 3.5],
    "min_confidence":      [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "rsi_overbought":      [65, 68, 70, 72, 75],
}


@dataclass
class SensitivityResult:
    """Result for one parameter combination."""
    params:      Dict[str, float]
    win_rate:    float
    net_pnl:     float
    max_drawdown: float
    sharpe:      float
    n_trades:    int


@dataclass
class SensitivityReport:
    """Full sensitivity sweep report."""
    base_params:     Dict[str, float]
    results:         List[SensitivityResult]
    stability_score: float    # 0–1, higher = flatter peak = more robust
    best_params:     Dict[str, float]
    worst_params:    Dict[str, float]
    param_importance: Dict[str, float]   # which param drives variance most
    recommendation:  str


class ParameterSensitivityTester:
    """
    Runs the backtest across a parameter grid and measures robustness.
    Stability score = 1 - (std(net_pnl) / mean(net_pnl)) — inverse of CV.
    """

    def __init__(self, backtest_fn: Callable):
        """
        Args:
            backtest_fn: callable(candles, params_dict) → BacktestResult
                         Must accept ATR/confidence/RSI overrides as kwargs.
        """
        self.backtest_fn = backtest_fn

    def run(
        self,
        candles:     List,
        pair:        str = "EURUSD",
        param_grid:  Optional[Dict] = None,
        max_combos:  int = 50,   # cap to avoid runaway compute
    ) -> SensitivityReport:
        """
        Run sensitivity sweep. Samples up to max_combos parameter combinations.
        """
        grid      = param_grid or PARAM_GRID
        all_keys  = list(grid.keys())
        all_vals  = list(grid.values())

        # Full cartesian product, then sample if too large
        combos = list(itertools.product(*all_vals))
        if len(combos) > max_combos:
            idx    = np.random.choice(len(combos), max_combos, replace=False)
            combos = [combos[i] for i in idx]

        logger.info(f"Sensitivity test: {len(combos)} parameter combinations on {pair}")

        results: List[SensitivityResult] = []

        for combo in combos:
            params = dict(zip(all_keys, combo))
            try:
                bt = self.backtest_fn(candles, params)
                results.append(SensitivityResult(
                    params=params,
                    win_rate=bt.win_rate,
                    net_pnl=bt.net_pnl,
                    max_drawdown=bt.max_drawdown,
                    sharpe=bt.sharpe,
                    n_trades=bt.total_trades,
                ))
            except Exception as e:
                logger.debug(f"Sensitivity backtest failed for {params}: {e}")
                continue

        if not results:
            return SensitivityReport(
                base_params={}, results=[], stability_score=0.0,
                best_params={}, worst_params={},
                param_importance={}, recommendation="No results — check backtest function",
            )

        pnls = [r.net_pnl for r in results]
        mean_pnl = np.mean(pnls)
        std_pnl  = np.std(pnls)

        # Stability: low CV = flat performance across params = robust
        cv = abs(std_pnl / mean_pnl) if mean_pnl != 0 else 1.0
        stability = max(0.0, min(1.0, 1.0 - cv))

        best  = max(results, key=lambda r: r.net_pnl)
        worst = min(results, key=lambda r: r.net_pnl)

        # Parameter importance: variance of PnL when each param varies
        importance = {}
        for key in all_keys:
            groups = {}
            for r in results:
                val = r.params[key]
                groups.setdefault(val, []).append(r.net_pnl)
            group_means = [np.mean(v) for v in groups.values() if v]
            importance[key] = round(float(np.std(group_means)), 2) if len(group_means) > 1 else 0.0

        if stability > 0.7:
            rec = "ROBUST: Performance is stable across parameter variations. Proceed with confidence."
        elif stability > 0.4:
            rec = "MODERATE: Some parameter sensitivity. Use conservative values near the centre of the grid."
        else:
            rec = "FRAGILE: Performance peaks sharply at specific parameters — likely overfit. Reduce complexity."

        logger.info(f"Sensitivity complete: stability={stability:.3f} best_pnl=${best.net_pnl:.0f}")

        return SensitivityReport(
            base_params={"atr_stop_multiplier": 1.5, "min_confidence": 0.55},
            results=results,
            stability_score=round(stability, 4),
            best_params=best.params,
            worst_params=worst.params,
            param_importance=importance,
            recommendation=rec,
        )

    def summary_table(self, report: SensitivityReport) -> str:
        lines = [
            f"Parameter Sensitivity Report",
            f"{'='*50}",
            f"Combinations tested: {len(report.results)}",
            f"Stability score:     {report.stability_score:.3f} (1.0 = perfectly robust)",
            f"",
            f"Best params:  {report.best_params}",
            f"  → PnL: ${max(r.net_pnl for r in report.results):.0f}",
            f"Worst params: {report.worst_params}",
            f"  → PnL: ${min(r.net_pnl for r in report.results):.0f}",
            f"",
            f"Parameter importance (PnL variance when varied):",
        ]
        for param, imp in sorted(report.param_importance.items(), key=lambda x: -x[1]):
            lines.append(f"  {param:<30} {imp:.2f}")
        lines.append(f"")
        lines.append(f"Recommendation: {report.recommendation}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 2. MONTE CARLO EQUITY CURVE BOOTSTRAPPER
# ════════════════════════════════════════════════════════════

@dataclass
class MonteCarloReport:
    n_paths:         int
    starting_balance: float
    final_equity:    Dict[str, float]   # percentiles: p5, p25, p50, p75, p95
    max_drawdown:    Dict[str, float]   # distribution of max DDs
    prob_ruin:       float              # P(equity drops below 50% of start)
    prob_profit:     float              # P(equity > start at end)
    consecutive_loss_dist: Dict[str, float]   # p50/p95 max consecutive losses
    recommendation:  str
    n_trades:        int


class MonteCarloEquityCurve:
    """
    Bootstrap-resamples trade sequence to estimate distribution of outcomes.

    Method:
        1. Take list of trade R-multiples (e.g. [+1.5, -1.0, +2.0, -1.0])
        2. Randomly resample with replacement N times
        3. Simulate equity curve for each path
        4. Compute percentile distribution across all paths

    This answers: "Is a 55% win rate on 40 trades statistically meaningful?"
    (If median and p5 are both profitable → yes. If p5 < starting → fragile.)
    """

    def __init__(self, r_multiples: List[float], risk_per_trade_pct: float = 0.02):
        """
        Args:
            r_multiples: list of R-multiples from completed trades
                         (+1.5 = won 1.5R, -1.0 = lost 1R)
            risk_per_trade_pct: fraction of balance risked per trade
        """
        self.r_multiples        = r_multiples
        self.risk_per_trade_pct = risk_per_trade_pct

    def run(
        self,
        n_paths:          int   = 1_000,
        starting_balance: float = 10_000.0,
        ruin_threshold:   float = 0.50,    # equity below 50% = "ruin"
    ) -> MonteCarloReport:
        """Run Monte Carlo simulation."""
        if len(self.r_multiples) < 5:
            return MonteCarloReport(
                n_paths=0, starting_balance=starting_balance,
                final_equity={}, max_drawdown={},
                prob_ruin=0.0, prob_profit=0.0,
                consecutive_loss_dist={},
                recommendation="Insufficient trades for Monte Carlo (need ≥ 5)",
                n_trades=len(self.r_multiples),
            )

        n_trades  = len(self.r_multiples)
        r_arr     = np.array(self.r_multiples)
        final_equities = []
        max_drawdowns  = []
        max_consec_losses = []
        ruin_count    = 0
        profit_count  = 0

        for _ in range(n_paths):
            # Resample trade sequence with replacement
            sampled = np.random.choice(r_arr, size=n_trades, replace=True)

            equity  = starting_balance
            peak    = starting_balance
            max_dd  = 0.0
            consec_loss = 0
            max_consec  = 0

            for r in sampled:
                trade_pnl = equity * self.risk_per_trade_pct * r
                equity   += trade_pnl
                equity    = max(equity, 0.01)   # floor at near-zero

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

                if r < 0:
                    consec_loss += 1
                    max_consec   = max(max_consec, consec_loss)
                else:
                    consec_loss = 0

            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            max_consec_losses.append(max_consec)

            if equity < starting_balance * ruin_threshold:
                ruin_count += 1
            if equity > starting_balance:
                profit_count += 1

        fe  = np.array(final_equities)
        mdd = np.array(max_drawdowns)
        mcl = np.array(max_consec_losses)

        prob_ruin   = ruin_count   / n_paths
        prob_profit = profit_count / n_paths

        def pctiles(arr, pcts):
            return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in pcts}

        final_dist = pctiles(fe, [5, 10, 25, 50, 75, 90, 95])
        dd_dist    = {k: round(v * 100, 2) for k, v in pctiles(mdd, [50, 75, 95]).items()}
        cl_dist    = pctiles(mcl, [50, 95])

        # Recommendation
        if prob_ruin > 0.20:
            rec = "HIGH RISK: >20% probability of significant drawdown. Reduce position size or improve strategy."
        elif prob_profit < 0.60:
            rec = "MARGINAL: <60% paths profitable. Edge may not be statistically significant."
        elif final_dist["p5"] > starting_balance:
            rec = "ROBUST: Even the worst 5% of simulated paths are profitable. Strong edge confirmed."
        elif final_dist["p25"] > starting_balance:
            rec = "GOOD: Bottom quartile of paths profitable. Solid edge with some tail risk."
        else:
            rec = "MODERATE: Median profitable but significant downside tail. Consider tighter risk controls."

        logger.info(
            f"Monte Carlo ({n_paths} paths, {n_trades} trades): "
            f"p50=${final_dist['p50']:,.0f} p5=${final_dist['p5']:,.0f} "
            f"ruin={prob_ruin:.1%} profit={prob_profit:.1%}"
        )

        return MonteCarloReport(
            n_paths=n_paths,
            starting_balance=starting_balance,
            final_equity=final_dist,
            max_drawdown=dd_dist,
            prob_ruin=round(prob_ruin, 4),
            prob_profit=round(prob_profit, 4),
            consecutive_loss_dist=cl_dist,
            recommendation=rec,
            n_trades=n_trades,
        )

    def summary(self, report: MonteCarloReport) -> str:
        lines = [
            f"Monte Carlo Equity Curve ({report.n_paths} paths, {report.n_trades} trades)",
            f"{'='*55}",
            f"Starting balance: ${report.starting_balance:,.2f}",
            f"",
            f"Final equity distribution:",
            f"  p5  (worst 5%):  ${report.final_equity.get('p5',  0):>10,.2f}",
            f"  p25 (low end):   ${report.final_equity.get('p25', 0):>10,.2f}",
            f"  p50 (median):    ${report.final_equity.get('p50', 0):>10,.2f}",
            f"  p75 (high end):  ${report.final_equity.get('p75', 0):>10,.2f}",
            f"  p95 (best 5%):   ${report.final_equity.get('p95', 0):>10,.2f}",
            f"",
            f"Max drawdown distribution:",
            f"  Median:  {report.max_drawdown.get('p50', 0):.1f}%",
            f"  p95:     {report.max_drawdown.get('p95', 0):.1f}%",
            f"",
            f"Max consecutive losses:",
            f"  Median:  {report.consecutive_loss_dist.get('p50', 0):.0f}",
            f"  p95:     {report.consecutive_loss_dist.get('p95', 0):.0f}",
            f"",
            f"P(profit):   {report.prob_profit:.1%}",
            f"P(ruin):     {report.prob_ruin:.1%}",
            f"",
            f"{'='*55}",
            f"{report.recommendation}",
        ]
        return "\n".join(lines)
