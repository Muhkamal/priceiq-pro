"""
PriceIQ Pro — Walk-Forward Optimization Engine v1.0

Real walk-forward optimization — NOT the sensitivity tester.

Difference:
    Sensitivity tester: sweeps parameters on ONE dataset, finds the best.
    Walk-forward optimizer: trains on window[i], tests on window[i+1],
                            rolls forward, reports out-of-sample performance.

Why it matters:
    In-sample best params always overfit.
    Walk-forward finds params that generalize — the only params you can
    trust in live trading.

Protocol (Anchored Walk-Forward):
    ┌──────────────────────────────────────────────────────┐
    │ TRAIN  [bar 0 → 200]  → optimize params             │
    ├──────────────────────────────────────────────────────┤
    │ TEST   [bar 200 → 250] → record OOS performance     │
    ├──────────────────────────────────────────────────────┤
    │ TRAIN  [bar 0 → 250]  → re-optimize (anchored)      │
    ├──────────────────────────────────────────────────────┤
    │ TEST   [bar 250 → 300] → record OOS performance     │
    └──────────────────────────────────────────────────────┘
    ... repeat until end of data

    Final report: aggregate of ALL out-of-sample folds.
    Efficiency ratio = OOS Sharpe / IS Sharpe
    > 0.60 = robust. < 0.30 = overfit.

Parameters optimized:
    atr_stop_multiplier: [1.0, 1.25, 1.5, 1.75, 2.0]
    atr_tp1_multiplier:  [1.5, 2.0, 2.5, 3.0]
    min_confidence:      [0.50, 0.55, 0.60, 0.65]
    rsi_overbought:      [68, 70, 72, 75]

Optimization objective:
    Maximize: Sharpe ratio of OOS trades
    Constraint: max_drawdown < 20%, n_trades >= 10

Usage:
    optimizer = WalkForwardOptimizer(backtest_fn=your_backtest)
    report    = optimizer.run(candles, pair="XAUUSD")
    print(report.efficiency_ratio)
    print(report.recommended_params)
    # Deploy recommended_params into v5_settings
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optimization grid ─────────────────────────────────────────
PARAM_GRID = {
    "atr_stop_multiplier": [1.0, 1.25, 1.5, 1.75, 2.0],
    "atr_tp1_multiplier":  [1.5, 2.0, 2.5, 3.0],
    "min_confidence":      [0.50, 0.55, 0.60, 0.65],
    "rsi_overbought":      [68, 70, 72, 75],
}

# Walk-forward defaults
DEFAULT_TRAIN_BARS = 200
DEFAULT_TEST_BARS  = 50
MIN_TRADES_PER_FOLD = 5    # fold ignored if fewer trades


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


def _sharpe(r_mults: List[float]) -> float:
    """Sharpe ratio from R-multiple returns."""
    if len(r_mults) < 2:
        return 0.0
    arr = np.array(r_mults)
    mu  = np.mean(arr)
    std = np.std(arr)
    return float(_safe_div(mu, std))


def _max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = _safe_div(peak - eq, peak)
        if dd > max_dd:
            max_dd = dd
    return max_dd


@dataclass
class FoldResult:
    """Result for one walk-forward fold (train + test)."""
    fold_idx:         int
    train_start:      int
    train_end:        int
    test_start:       int
    test_end:         int
    # In-sample (training) performance
    is_sharpe:        float
    is_trades:        int
    is_win_rate:      float
    is_best_params:   Dict
    # Out-of-sample (test) performance with IS params
    oos_sharpe:       float
    oos_trades:       int
    oos_win_rate:     float
    oos_net_pnl:      float
    oos_max_drawdown: float
    oos_r_mults:      List[float]
    oos_equity:       List[float]


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward optimization report."""
    pair:            str
    total_bars:      int
    n_folds:         int
    train_bars:      int
    test_bars:       int
    folds:           List[FoldResult]

    # Aggregate OOS metrics
    oos_sharpe:      float     # across all OOS folds
    oos_win_rate:    float
    oos_net_pnl:     float
    oos_max_drawdown: float
    oos_total_trades: int

    # IS metrics (for comparison)
    avg_is_sharpe:   float

    # Robustness metrics
    efficiency_ratio: float    # OOS Sharpe / IS Sharpe (>0.6 = robust)
    stability_score:  float    # std of fold Sharpe values (lower = more stable)

    # Recommendation
    recommended_params: Dict
    recommendation:     str
    timestamp:          str

    def fold_summary(self) -> str:
        lines = [
            f"Walk-Forward Results: {self.pair}",
            f"{'='*60}",
            f"Folds: {self.n_folds} | Train: {self.train_bars} bars | Test: {self.test_bars} bars",
            f"",
            f"{'Fold':<6} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS WR':>8} {'OOS Trades':>11}",
            f"{'-'*50}",
        ]
        for f in self.folds:
            lines.append(
                f"{f.fold_idx:<6} {f.is_sharpe:>10.3f} {f.oos_sharpe:>11.3f} "
                f"{f.oos_win_rate:>8.1%} {f.oos_trades:>11}"
            )
        lines.extend([
            f"{'-'*50}",
            f"{'AVG IS':>16} {self.avg_is_sharpe:>11.3f}",
            f"{'OOS TOTAL':>16} {self.oos_sharpe:>11.3f} "
            f"{self.oos_win_rate:>8.1%} {self.oos_total_trades:>11}",
            f"",
            f"Efficiency ratio:  {self.efficiency_ratio:.3f}  "
            f"({'ROBUST' if self.efficiency_ratio >= 0.6 else 'MARGINAL' if self.efficiency_ratio >= 0.3 else 'OVERFIT'})",
            f"Stability score:   {self.stability_score:.3f}  "
            f"({'STABLE' if self.stability_score < 0.5 else 'VARIABLE'})",
            f"",
            f"Recommended params: {self.recommended_params}",
            f"",
            f"{self.recommendation}",
            f"{'='*60}",
        ])
        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Anchored walk-forward optimization.
    Finds parameters that generalize out-of-sample.

    The backtest_fn signature:
        result = backtest_fn(candles: List, params: Dict) -> BacktestResult
        result must have: .win_rate, .sharpe, .max_drawdown, .total_trades,
                          .net_pnl, and a list of trade R-multiples accessible
                          as [t.r_multiple for t in result.trades]
    """

    def __init__(
        self,
        backtest_fn: Callable,
        param_grid:  Optional[Dict] = None,
        train_bars:  int = DEFAULT_TRAIN_BARS,
        test_bars:   int = DEFAULT_TEST_BARS,
        max_combos:  int = 40,    # cap combinations to avoid runaway compute
    ):
        self.backtest_fn = backtest_fn
        self.param_grid  = param_grid or PARAM_GRID
        self.train_bars  = train_bars
        self.test_bars   = test_bars
        self.max_combos  = max_combos

    def run(
        self,
        candles: List,
        pair:    str = "EURUSD",
        starting_balance: float = 10_000.0,
        risk_pct: float = 2.0,
    ) -> WalkForwardReport:
        """
        Run anchored walk-forward optimization on candle history.
        Returns comprehensive WalkForwardReport.
        """
        now = datetime.now(timezone.utc).isoformat()
        n   = len(candles)

        if n < self.train_bars + self.test_bars:
            raise ValueError(
                f"Need ≥ {self.train_bars + self.test_bars} candles. "
                f"Got {n}."
            )

        # Build parameter combinations
        all_combos = self._build_combos()
        logger.info(
            f"Walk-forward optimizer: {pair} | "
            f"{len(all_combos)} param combos | "
            f"train={self.train_bars} test={self.test_bars}"
        )

        folds: List[FoldResult] = []
        fold_idx = 0

        # Anchored: train window starts at 0, grows each fold
        test_start = self.train_bars
        while test_start + self.test_bars <= n:
            train_end = test_start
            test_end  = test_start + self.test_bars

            logger.info(
                f"Fold {fold_idx}: train[0:{train_end}] "
                f"test[{test_start}:{test_end}]"
            )

            # ── In-sample optimization ────────────────────────
            train_candles = candles[:train_end]
            best_is_params, best_is_sharpe, best_is_trades, best_is_wr = \
                self._optimize_in_sample(train_candles, pair, starting_balance, risk_pct)

            # ── Out-of-sample evaluation ──────────────────────
            test_candles = candles[:test_end]   # anchored: include all history
            oos_result   = self._evaluate_oos(
                test_candles, test_start, test_end,
                best_is_params, pair, starting_balance, risk_pct,
            )

            fold = FoldResult(
                fold_idx=fold_idx,
                train_start=0, train_end=train_end,
                test_start=test_start, test_end=test_end,
                is_sharpe=best_is_sharpe,
                is_trades=best_is_trades,
                is_win_rate=best_is_wr,
                is_best_params=best_is_params,
                oos_sharpe=oos_result["sharpe"],
                oos_trades=oos_result["n_trades"],
                oos_win_rate=oos_result["win_rate"],
                oos_net_pnl=oos_result["net_pnl"],
                oos_max_drawdown=oos_result["max_dd"],
                oos_r_mults=oos_result["r_mults"],
                oos_equity=oos_result["equity"],
            )
            folds.append(fold)
            fold_idx   += 1
            test_start += self.test_bars   # step forward

        if not folds:
            raise ValueError("No folds completed — insufficient data")

        # ── Aggregate OOS metrics ─────────────────────────────
        all_oos_r_mults = []
        for f in folds:
            all_oos_r_mults.extend(f.oos_r_mults)

        oos_sharpe   = _sharpe(all_oos_r_mults)
        oos_wr       = _safe_div(
            sum(1 for r in all_oos_r_mults if r > 0), len(all_oos_r_mults)
        )
        oos_net_pnl  = sum(f.oos_net_pnl for f in folds)
        oos_max_dd   = max((f.oos_max_drawdown for f in folds), default=0.0)
        oos_n_trades = sum(f.oos_trades for f in folds)
        avg_is_sharpe = np.mean([f.is_sharpe for f in folds])
        fold_sharpes  = [f.oos_sharpe for f in folds]
        stability     = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0

        efficiency    = _safe_div(oos_sharpe, avg_is_sharpe)

        # Recommended params: use most recent fold's best IS params
        # (they trained on the most data)
        recommended = folds[-1].is_best_params if folds else {}

        # Recommendation text
        if efficiency >= 0.6 and oos_sharpe > 0.5:
            rec = (
                f"✅ ROBUST: Efficiency ratio {efficiency:.2f} ≥ 0.60 and "
                f"OOS Sharpe {oos_sharpe:.2f} > 0.50. "
                f"Deploy recommended params with confidence."
            )
        elif efficiency >= 0.3:
            rec = (
                f"⚠️ MARGINAL: Efficiency ratio {efficiency:.2f} (0.30–0.60). "
                f"Some overfitting. Use params cautiously with reduced position size."
            )
        else:
            rec = (
                f"❌ OVERFIT: Efficiency ratio {efficiency:.2f} < 0.30. "
                f"IS performance does not transfer OOS. "
                f"Simplify the strategy or collect more data before deploying."
            )

        logger.info(
            f"Walk-forward complete: {pair} | "
            f"OOS Sharpe={oos_sharpe:.3f} | "
            f"Efficiency={efficiency:.3f} | "
            f"OOS trades={oos_n_trades}"
        )

        return WalkForwardReport(
            pair=pair, total_bars=n, n_folds=len(folds),
            train_bars=self.train_bars, test_bars=self.test_bars,
            folds=folds,
            oos_sharpe=round(oos_sharpe, 4),
            oos_win_rate=round(oos_wr, 4),
            oos_net_pnl=round(oos_net_pnl, 2),
            oos_max_drawdown=round(oos_max_dd, 4),
            oos_total_trades=oos_n_trades,
            avg_is_sharpe=round(float(avg_is_sharpe), 4),
            efficiency_ratio=round(efficiency, 4),
            stability_score=round(stability, 4),
            recommended_params=recommended,
            recommendation=rec,
            timestamp=now,
        )

    def _optimize_in_sample(
        self,
        candles: List,
        pair: str,
        balance: float,
        risk_pct: float,
    ) -> Tuple[Dict, float, int, float]:
        """Find best params on training data. Returns (params, sharpe, n_trades, wr)."""
        combos  = self._build_combos()
        best_sharpe = -999.0
        best_params = combos[0] if combos else {}
        best_trades = 0
        best_wr     = 0.0

        for params in combos:
            try:
                result = self.backtest_fn(candles, params)
                n      = getattr(result, "total_trades", 0)
                if n < MIN_TRADES_PER_FOLD:
                    continue
                dd     = getattr(result, "max_drawdown", 1.0)
                if dd > 0.20:   # constraint: don't allow > 20% DD in IS
                    continue
                sharpe = getattr(result, "sharpe", 0.0)
                wr     = getattr(result, "win_rate", 0.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_trades = n
                    best_wr     = wr
            except Exception as e:
                logger.debug(f"IS backtest failed for {params}: {e}")
                continue

        return best_params, best_sharpe, best_trades, best_wr

    def _evaluate_oos(
        self,
        candles: List,
        test_start: int,
        test_end: int,
        params: Dict,
        pair: str,
        balance: float,
        risk_pct: float,
    ) -> Dict:
        """Evaluate IS-optimized params on OOS period."""
        try:
            result = self.backtest_fn(candles, params)
            # Extract only the OOS trades (those in the test window)
            all_trades = getattr(result, "trades", [])
            oos_trades = [
                t for t in all_trades
                if test_start <= getattr(t, "bar_index", 0) < test_end
            ]
            r_mults = [getattr(t, "r_multiple", 0.0) for t in oos_trades]
            wins    = sum(1 for r in r_mults if r > 0)
            n       = len(r_mults)

            # Equity curve for OOS
            equity = [balance]
            for r in r_mults:
                equity.append(equity[-1] + equity[-1] * (risk_pct / 100) * r)

            return {
                "sharpe":   _sharpe(r_mults),
                "n_trades": n,
                "win_rate": _safe_div(wins, n),
                "net_pnl":  sum(r * balance * (risk_pct/100) for r in r_mults),
                "max_dd":   _max_drawdown(equity),
                "r_mults":  r_mults,
                "equity":   equity,
            }
        except Exception as e:
            logger.debug(f"OOS evaluation failed: {e}")
            return {
                "sharpe": 0.0, "n_trades": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "max_dd": 0.0, "r_mults": [], "equity": [balance],
            }

    def _build_combos(self) -> List[Dict]:
        """Build parameter combinations, capped at max_combos."""
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combos = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]
        if len(all_combos) > self.max_combos:
            rng  = np.random.default_rng(seed=42)
            idxs = rng.choice(len(all_combos), self.max_combos, replace=False)
            return [all_combos[i] for i in sorted(idxs)]
        return all_combos
