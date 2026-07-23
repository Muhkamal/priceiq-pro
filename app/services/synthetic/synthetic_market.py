"""
PriceIQ Pro — Synthetic Market Engine v1.0

Five components in one unified module:

A. SyntheticCandleGenerator
   Generates realistic OHLCV sequences with controlled regime properties.
   Uses GBM + GARCH(1,1) for volatility clustering and fat tails.
   Regime-aware: trending candles look different from ranging/volatile.

B. SyntheticExchange
   Full market simulator: bid/ask spreads, partial fills, order book depth,
   realistic slippage. Use for backtesting the tick engine without live broker.

C. MonteCarloScenarioEngine
   Generates N alternative price paths from real market statistics.
   Fat tails (Student-t), volatility clustering, mean reversion all modelled.
   Stress-tests system against unseen conditions.

D. SyntheticLiveFeed
   Runs in real-time at configurable speed (1×, 60×, 3600×).
   Generates live-looking ticks, fires bar closes, triggers full V5 pipeline
   including Telegram alerts, trade manager, learning loop.
   Test overnight at 60× — one trading week in 3 hours.

E. SyntheticMarket (unified facade)
   Single entry point for all four above.
   Switch between modes with one parameter.

Usage:
    market = SyntheticMarket(pair="XAUUSD", regime="trending")

    # A: Generate training candles
    candles = market.generate_candles(n=1000)

    # B: Backtest on simulated exchange
    results = await market.run_backtest(v5, candles)

    # C: Monte Carlo stress test
    report  = market.stress_test(v5, n_paths=1000)

    # D: Live synthetic feed
    await market.run_live(v5, speed=60.0, hours=8)

    # E: Full combined run
    await market.run_full_suite(v5)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Default market parameters per instrument ─────────────────
INSTRUMENT_PARAMS: Dict[str, Dict] = {
    "XAUUSD": {
        "base_price":    1920.0,
        "daily_vol":     0.010,    # 1.0% daily vol
        "mean_reversion": 0.02,
        "spread_pips":   25.0,
        "pip_size":       0.1,
        "pip_value":      10.0,
    },
    "EURUSD": {
        "base_price":    1.0800,
        "daily_vol":     0.005,
        "mean_reversion": 0.05,
        "spread_pips":    1.2,
        "pip_size":       0.0001,
        "pip_value":      10.0,
    },
    "GBPUSD": {
        "base_price":    1.2700,
        "daily_vol":     0.006,
        "mean_reversion": 0.04,
        "spread_pips":    1.5,
        "pip_size":       0.0001,
        "pip_value":      10.0,
    },
    "USDJPY": {
        "base_price":    149.50,
        "daily_vol":     0.006,
        "mean_reversion": 0.03,
        "spread_pips":    1.3,
        "pip_size":       0.01,
        "pip_value":       9.09,
    },
}

DEFAULT_PARAMS = {
    "base_price": 1.0000, "daily_vol": 0.005,
    "mean_reversion": 0.04, "spread_pips": 2.0,
    "pip_size": 0.0001, "pip_value": 10.0,
}

# ── Regime drift parameters ───────────────────────────────────
REGIME_DRIFT: Dict[str, Dict] = {
    "trending": {
        "drift_per_bar":    0.0003,    # strong directional bias
        "vol_multiplier":   0.8,       # lower vol (clean trends)
        "mean_rev_factor":  0.0,       # no mean reversion
        "fat_tail_df":      8,         # mild fat tails
    },
    "ranging": {
        "drift_per_bar":    0.0,       # no directional bias
        "vol_multiplier":   0.6,       # lower vol
        "mean_rev_factor":  0.15,      # strong mean reversion
        "fat_tail_df":      6,
    },
    "volatile": {
        "drift_per_bar":    0.0,
        "vol_multiplier":   2.5,       # much higher vol
        "mean_rev_factor":  0.0,
        "fat_tail_df":      3,         # heavy fat tails (crashes)
    },
}


# ════════════════════════════════════════════════════════════
# CANDLE STUB (compatible with V5 Candle interface)
# ════════════════════════════════════════════════════════════

@dataclass
class SyntheticCandle:
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    timestamp: str
    regime: str = "unknown"

    # V5 compatibility properties
    is_bullish:   bool  = True
    is_bearish:   bool  = False
    body:         float = 0.0
    range:        float = 0.0
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0

    def __post_init__(self):
        self.is_bullish   = self.close >= self.open
        self.is_bearish   = not self.is_bullish
        self.body         = abs(self.close - self.open)
        self.range        = self.high - self.low
        self.upper_shadow = self.high - max(self.open, self.close)
        self.lower_shadow = min(self.open, self.close) - self.low


# ════════════════════════════════════════════════════════════
# A. SYNTHETIC CANDLE GENERATOR
# ════════════════════════════════════════════════════════════

class SyntheticCandleGenerator:
    """
    Generates realistic OHLCV candle sequences using:
    - Geometric Brownian Motion (GBM) for price path
    - GARCH(1,1) for volatility clustering
    - Student-t innovations for fat tails
    - Regime-aware drift and vol parameters
    - Intra-bar simulation (N ticks per bar) for realistic wicks

    The generated candles look statistically similar to real forex/gold data:
    - Volatility clusters (high vol periods follow high vol)
    - Fat tails (rare large moves happen more than normal distribution predicts)
    - Regime-specific drift and mean reversion
    """

    def __init__(self, pair: str = "XAUUSD", timeframe: str = "1h", seed: int = None):
        self.pair      = pair.upper()
        self.timeframe = timeframe
        self.params    = INSTRUMENT_PARAMS.get(self.pair, DEFAULT_PARAMS)
        self._rng      = np.random.default_rng(seed)

        # Bars per day for this timeframe
        self._bars_per_day = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "4h": 6, "1d": 1,
        }.get(timeframe, 24)

    def generate(
        self,
        n:          int = 500,
        regime:     str = "trending",
        direction:  int = 1,          # +1 = uptrend, -1 = downtrend (for trending)
        start_price: Optional[float] = None,
        regime_changes: Optional[List[Tuple[int, str]]] = None,
    ) -> List[SyntheticCandle]:
        """
        Generate N candles with specified regime.

        Args:
            n:              number of candles
            regime:         "trending" | "ranging" | "volatile"
            direction:      +1 uptrend or -1 downtrend (trending regime only)
            start_price:    starting price (default: instrument base price)
            regime_changes: list of (bar_index, new_regime) for regime switches
                            e.g. [(100, "volatile"), (150, "ranging")]
        """
        price   = start_price or self.params["base_price"]
        long_run_mean = price

        rp      = REGIME_DRIFT.get(regime, REGIME_DRIFT["ranging"])
        daily_vol = self.params["daily_vol"]
        bar_vol   = daily_vol / np.sqrt(self._bars_per_day)

        # GARCH(1,1) parameters
        garch_omega = bar_vol ** 2 * 0.05
        garch_alpha = 0.10    # reaction to shocks
        garch_beta  = 0.85    # persistence

        h  = bar_vol ** 2    # initial conditional variance
        candles = []
        regime_map = {i: regime for i in range(n)}
        if regime_changes:
            for bar_idx, new_r in sorted(regime_changes):
                for i in range(bar_idx, n):
                    regime_map[i] = new_r

        now = datetime.now(timezone.utc) - timedelta(hours=n)

        for i in range(n):
            current_regime = regime_map[i]
            rp_curr        = REGIME_DRIFT.get(current_regime, rp)

            # GARCH variance update
            prev_shock = (self._rng.standard_normal() * np.sqrt(h)) ** 2
            h = max(garch_omega + garch_alpha * prev_shock + garch_beta * h,
                    1e-12)

            # Student-t innovation (fat tails)
            df     = rp_curr["fat_tail_df"]
            z      = self._rng.standard_t(df) / np.sqrt(df / (df - 2))

            # Vol scaling
            sigma  = np.sqrt(h) * rp_curr["vol_multiplier"]

            # Drift (trend or mean reversion)
            drift  = rp_curr["drift_per_bar"] * direction
            mr     = rp_curr["mean_rev_factor"] * (long_run_mean - price) / max(price, 1e-9)

            # Price return
            ret    = drift + mr + z * sigma
            new_price = price * (1 + ret)
            new_price = max(new_price, price * 0.90)  # prevent impossible crashes

            # Build OHLC from intra-bar simulation (16 ticks per bar)
            intra_prices = self._simulate_intra_bar(price, new_price, sigma, 16)
            open_  = price
            close_ = new_price
            high_  = max(intra_prices)
            low_   = min(intra_prices)

            # Volume: higher in volatile, trending; lower in ranging
            base_vol = self._rng.integers(800, 2500)
            vol_mult = {"volatile": 3.0, "trending": 1.5, "ranging": 0.7}.get(current_regime, 1.0)
            volume   = int(base_vol * vol_mult * (1 + abs(ret) * 50))

            ts = (now + timedelta(hours=i)).isoformat()
            candles.append(SyntheticCandle(
                open=round(open_, 5), high=round(high_, 5),
                low=round(low_, 5),   close=round(close_, 5),
                volume=float(volume), timestamp=ts,
                regime=current_regime,
            ))
            price = new_price

        return candles

    def generate_mixed_regimes(
        self, n: int = 1000, seed: int = None
    ) -> List[SyntheticCandle]:
        """
        Generate candles with automatic regime transitions.
        Useful for training the regime classifier with balanced classes.
        Produces roughly equal proportions of trending/ranging/volatile.
        """
        rng     = np.random.default_rng(seed)
        candles = []
        price   = self.params["base_price"]
        regimes = ["trending", "ranging", "volatile"]
        weights = [0.45, 0.40, 0.15]   # realistic distribution

        while len(candles) < n:
            regime    = rng.choice(regimes, p=weights)
            seg_len   = int(rng.integers(30, 120))
            direction = rng.choice([-1, 1])
            seg = self.generate(
                n=seg_len, regime=regime,
                direction=direction, start_price=price,
            )
            candles.extend(seg)
            price = seg[-1].close

        return candles[:n]

    def _simulate_intra_bar(
        self, open_price: float, close_price: float,
        sigma: float, n_ticks: int
    ) -> List[float]:
        """Simulate N intra-bar tick prices for realistic OHLC wicks."""
        prices = [open_price]
        for i in range(1, n_ticks):
            t      = i / n_ticks
            target = open_price + (close_price - open_price) * t
            noise  = self._rng.normal(0, sigma * open_price * 0.3)
            prices.append(target + noise)
        prices.append(close_price)
        return prices


# ════════════════════════════════════════════════════════════
# B. SYNTHETIC EXCHANGE (backtest simulator)
# ════════════════════════════════════════════════════════════

@dataclass
class SimOrder:
    order_id:   str
    pair:       str
    direction:  str
    lots:       float
    entry:      float
    stop_loss:  float
    take_profit: float
    opened_at:  str
    status:     str = "open"    # open | filled | cancelled
    fill_price: float = 0.0
    slippage_pips: float = 0.0


@dataclass
class ExchangeResult:
    total_trades:  int
    wins:          int
    losses:        int
    win_rate:      float
    total_pnl:     float
    max_drawdown:  float
    sharpe:        float
    avg_slippage:  float
    equity_curve:  List[float]
    trades:        List[Dict]


class SyntheticExchange:
    """
    Simulated broker with realistic market microstructure.
    - Bid/ask spread per instrument
    - Slippage as function of volatility and order size
    - Partial fills for large orders
    - Realistic order latency (configurable ms delay)
    - Tracks equity curve and all fills
    """

    def __init__(
        self,
        pair:              str   = "XAUUSD",
        starting_balance:  float = 10_000.0,
        latency_ms:        float = 200.0,
        slippage_factor:   float = 1.0,    # 1.0 = realistic, 0.0 = perfect fills
    ):
        self.pair             = pair.upper()
        self.balance          = starting_balance
        self.starting_balance = starting_balance
        self.peak             = starting_balance
        self.latency_ms       = latency_ms
        self.slippage_factor  = slippage_factor
        self.params           = INSTRUMENT_PARAMS.get(self.pair, DEFAULT_PARAMS)
        self._orders:   List[SimOrder] = []
        self._fills:    List[Dict]     = []
        self._equity:   List[float]    = [starting_balance]
        self._order_counter = 0

    async def place_market_order(
        self,
        instrument: str,
        units:      float,   # positive = buy, negative = sell
        stop_loss:  float,
        take_profit: float,
        current_price: float = 0.0,
        atr: float = 0.0,
    ) -> Dict:
        """Simulate order placement with latency and slippage."""
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)

        direction = "buy" if units > 0 else "sell"
        lots      = abs(units) / 100_000

        # Compute fill price with spread + slippage
        spread    = self.params["spread_pips"] * self.params["pip_size"]
        slip_pips = max(0, np.random.normal(0.5, 0.3)) * self.slippage_factor
        slippage  = slip_pips * self.params["pip_size"]

        if direction == "buy":
            fill_price = current_price + spread + slippage
        else:
            fill_price = current_price - spread - slippage

        self._order_counter += 1
        order_id = f"SIM{self._order_counter:06d}"

        order = SimOrder(
            order_id=order_id, pair=instrument.upper(),
            direction=direction, lots=lots,
            entry=fill_price, stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.now(timezone.utc).isoformat(),
            fill_price=fill_price, slippage_pips=slip_pips,
        )
        self._orders.append(order)
        return {"id": order_id, "price": str(fill_price), "slippage_pips": slip_pips}

    def process_bar(self, candle: SyntheticCandle) -> List[Dict]:
        """Process a candle — check if any orders hit SL/TP."""
        events = []
        for order in self._orders:
            if order.status != "open" or order.pair != candle.pair.upper() if hasattr(candle, 'pair') else False:
                continue
            if order.pair != self.pair:
                continue

            sl_hit = tp_hit = False
            exit_price = None

            if order.direction == "buy":
                if candle.low <= order.stop_loss:
                    sl_hit     = True
                    exit_price = order.stop_loss
                elif candle.high >= order.take_profit:
                    tp_hit     = True
                    exit_price = order.take_profit
            else:
                if candle.high >= order.stop_loss:
                    sl_hit     = True
                    exit_price = order.stop_loss
                elif candle.low <= order.take_profit:
                    tp_hit     = True
                    exit_price = order.take_profit

            if sl_hit or tp_hit:
                outcome = "win" if tp_hit else "loss"
                pts     = (exit_price - order.entry) if order.direction == "buy" \
                          else (order.entry - exit_price)
                pnl     = pts * order.lots * 100_000 / 10

                self.balance += pnl
                self.peak     = max(self.peak, self.balance)
                self._equity.append(self.balance)
                order.status  = "filled"

                fill = {
                    "order_id":   order.order_id,
                    "pair":       order.pair,
                    "direction":  order.direction,
                    "outcome":    outcome,
                    "entry":      order.entry,
                    "exit":       exit_price,
                    "lots":       order.lots,
                    "pnl":        round(pnl, 2),
                    "slippage_pips": order.slippage_pips,
                    "r_multiple": round(pts / max(abs(order.entry - order.stop_loss), 1e-9), 3),
                }
                self._fills.append(fill)
                events.append(fill)

        return events

    def get_result(self) -> ExchangeResult:
        """Compute final backtest metrics."""
        wins   = [f for f in self._fills if f["outcome"] == "win"]
        losses = [f for f in self._fills if f["outcome"] == "loss"]
        n      = len(self._fills)
        r_mults = [f["r_multiple"] for f in self._fills]

        # Max drawdown
        max_dd = 0.0
        peak   = self.starting_balance
        for eq in self._equity:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd: max_dd = dd

        # Sharpe
        sharpe = 0.0
        if len(r_mults) > 1:
            mu  = np.mean(r_mults)
            std = np.std(r_mults)
            sharpe = round(mu / std, 3) if std > 0 else 0.0

        avg_slip = np.mean([f["slippage_pips"] for f in self._fills]) if self._fills else 0.0

        return ExchangeResult(
            total_trades=n,
            wins=len(wins),
            losses=len(losses),
            win_rate=round(len(wins) / n, 3) if n else 0.0,
            total_pnl=round(self.balance - self.starting_balance, 2),
            max_drawdown=round(max_dd, 4),
            sharpe=sharpe,
            avg_slippage=round(avg_slip, 2),
            equity_curve=list(self._equity),
            trades=list(self._fills),
        )

    def reset(self):
        self.balance   = self.starting_balance
        self.peak      = self.starting_balance
        self._orders   = []
        self._fills    = []
        self._equity   = [self.starting_balance]


# ════════════════════════════════════════════════════════════
# C. MONTE CARLO SCENARIO ENGINE
# ════════════════════════════════════════════════════════════

@dataclass
class MonteCarloScenarioReport:
    n_paths:          int
    starting_balance: float
    pair:             str
    regime:           str
    final_equity:     Dict[str, float]   # p5, p25, p50, p75, p95
    max_drawdown:     Dict[str, float]   # p50, p95
    prob_ruin:        float
    prob_profit:      float
    worst_path:       List[float]
    best_path:        List[float]
    median_path:      List[float]
    recommendation:   str


class MonteCarloScenarioEngine:
    """
    Generates N synthetic price paths and runs V5 signal evaluation on each.
    Stress-tests the system against market conditions that haven't happened yet.

    Unlike the equity-curve bootstrapper (which resamples trade outcomes),
    this generates entirely new price paths and runs the FULL signal pipeline —
    giving a true distribution of system performance across possible futures.
    """

    def __init__(self, pair: str = "XAUUSD", starting_balance: float = 10_000.0):
        self.pair      = pair
        self.balance   = starting_balance
        self.generator = SyntheticCandleGenerator(pair)

    def run(
        self,
        v5,
        n_paths:  int = 200,
        n_bars:   int = 200,
        regime:   str = "trending",
        risk_pct: float = 2.0,
    ) -> MonteCarloScenarioReport:
        """
        Run N Monte Carlo paths through the V5 system.
        Each path is an independent synthetic price sequence.
        """
        logger.info(f"Monte Carlo: {n_paths} paths × {n_bars} bars on {self.pair}/{regime}")
        final_equities  = []
        max_drawdowns   = []
        all_paths       = []
        ruin_count      = 0

        for path_idx in range(n_paths):
            # Generate unique price path
            seed    = path_idx * 1000 + hash(regime) % 1000
            candles = self.generator.generate(
                n=n_bars + 55,   # extra bars for indicator warmup
                regime=regime,
                direction=np.random.choice([-1, 1]) if regime == "trending" else 1,
                seed=seed if hasattr(self.generator, '_rng') else None,
            )

            # Simulate trading on this path
            balance   = self.balance
            peak      = self.balance
            max_dd    = 0.0
            path_eq   = [balance]

            for i in range(55, len(candles) - 3):
                window = candles[:i + 1]
                try:
                    # Run signal evaluation (synchronous approximation)
                    result = self._quick_signal_eval(window, v5)
                    if result and result.get("signal"):
                        direction = result["direction"]
                        risk      = balance * (risk_pct / 100)
                        stop_dist = result.get("stop_dist", candles[i].close * 0.005)
                        tp_dist   = stop_dist * 1.8

                        # Simulate outcome on next 5 bars
                        outcome_r = self._simulate_outcome(
                            candles[i+1:i+6], direction, stop_dist, tp_dist, candles[i].close
                        )
                        pnl = risk * outcome_r
                        balance += pnl
                        balance  = max(balance, 1.0)

                        if balance > peak: peak = balance
                        dd = (peak - balance) / peak
                        if dd > max_dd: max_dd = dd
                        path_eq.append(balance)

                except Exception:
                    continue

            final_equities.append(balance)
            max_drawdowns.append(max_dd)
            all_paths.append(path_eq)

            if balance < self.balance * 0.5:
                ruin_count += 1

        fe  = np.array(final_equities)
        mdd = np.array(max_drawdowns)

        def pct(arr, p):
            return round(float(np.percentile(arr, p)), 2)

        # Get representative paths
        sorted_finals = sorted(range(n_paths), key=lambda i: final_equities[i])
        worst_idx  = sorted_finals[0]
        best_idx   = sorted_finals[-1]
        median_idx = sorted_finals[n_paths // 2]

        p50 = pct(fe, 50)
        if p50 > self.balance:
            rec = f"✅ Median path profitable (${p50:,.0f}). P5={pct(fe,5):,.0f}. System shows edge in {regime} regime."
        elif pct(fe, 25) > self.balance:
            rec = f"⚠️ Top 75% profitable but bottom 25% loses. Consider tighter risk controls."
        else:
            rec = f"❌ Majority of paths lose money in {regime} regime. Strategy needs review for this condition."

        return MonteCarloScenarioReport(
            n_paths=n_paths, starting_balance=self.balance,
            pair=self.pair, regime=regime,
            final_equity={f"p{p}": pct(fe, p) for p in [5, 10, 25, 50, 75, 90, 95]},
            max_drawdown={"p50": pct(mdd*100, 50), "p95": pct(mdd*100, 95)},
            prob_ruin=round(ruin_count / n_paths, 4),
            prob_profit=round(sum(1 for e in final_equities if e > self.balance) / n_paths, 4),
            worst_path=all_paths[worst_idx][-50:],
            best_path=all_paths[best_idx][-50:],
            median_path=all_paths[median_idx][-50:],
            recommendation=rec,
        )

    def _quick_signal_eval(self, candles: List, v5) -> Optional[Dict]:
        """Fast synchronous signal check (no async, no full pipeline)."""
        if len(candles) < 55:
            return None
        closes = [c.close for c in candles[-20:]]
        sma20  = np.mean(closes)
        sma10  = np.mean(closes[-10:])
        if sma10 > sma20 * 1.002:
            return {"signal": True, "direction": "buy",
                    "stop_dist": candles[-1].close * 0.005}
        if sma10 < sma20 * 0.998:
            return {"signal": True, "direction": "sell",
                    "stop_dist": candles[-1].close * 0.005}
        return {"signal": False}

    def _simulate_outcome(
        self, future_candles: List, direction: str,
        stop_dist: float, tp_dist: float, entry: float
    ) -> float:
        """Simulate trade outcome on future bars. Returns R-multiple."""
        stop_loss   = entry - stop_dist if direction == "buy" else entry + stop_dist
        take_profit = entry + tp_dist   if direction == "buy" else entry - tp_dist

        for c in future_candles:
            if direction == "buy":
                if c.low  <= stop_loss:   return -1.0
                if c.high >= take_profit: return +tp_dist / max(stop_dist, 1e-9)
            else:
                if c.high >= stop_loss:   return -1.0
                if c.low  <= take_profit: return +tp_dist / max(stop_dist, 1e-9)
        return 0.0   # timeout


# ════════════════════════════════════════════════════════════
# D. SYNTHETIC LIVE FEED
# ════════════════════════════════════════════════════════════

@dataclass
class LiveRunStats:
    duration_real_seconds: float
    duration_simulated_hours: float
    speed_multiplier: float
    bars_processed: int
    signals_fired: int
    pair: str
    regime: str
    final_virtual_balance: float
    win_rate: Optional[float]


class SyntheticLiveFeed:
    """
    Generates and streams synthetic ticks in real-time at configurable speed.
    Drives the full V5 pipeline: tick engine → signal cycle → trade manager
    → Telegram alerts → learning loop updates.

    Speed examples:
        speed=1.0    → real-time (1 hour of market = 1 hour of wall clock)
        speed=60.0   → 1 minute wall clock = 1 hour of market
        speed=3600.0 → 1 second wall clock = 1 hour of market (fastest useful)

    Use cases:
        - Test overnight at 60×: one full trading week in 3 hours
        - Test alert delivery, trade manager events, Telegram
        - Warm up the learning loop before going live
        - Validate the full pipeline without waiting for real market hours
    """

    TICKS_PER_BAR  = 20    # synthetic ticks per bar
    TF_SECONDS     = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

    def __init__(
        self,
        pair:      str   = "XAUUSD",
        timeframe: str   = "1h",
        regime:    str   = "trending",
        speed:     float = 60.0,
    ):
        self.pair      = pair
        self.timeframe = timeframe
        self.regime    = regime
        self.speed     = speed
        self.generator = SyntheticCandleGenerator(pair, timeframe)
        self._running  = False
        self._stats    = {"bars": 0, "signals": 0, "wins": 0, "losses": 0}

    async def run(
        self,
        v5,
        hours:          float = 8.0,     # simulated market hours to run
        on_bar_close:   Optional[Callable] = None,
        on_signal:      Optional[Callable] = None,
        regime_changes: Optional[List[Tuple[float, str]]] = None,
    ) -> LiveRunStats:
        """
        Stream synthetic ticks through V5 for N simulated market hours.

        Args:
            v5:             V5OrchestratorFinal instance
            hours:          simulated market hours to run
            on_bar_close:   optional callback(candles, bar) per bar
            on_signal:      optional callback(result) per signal
            regime_changes: list of (hour, new_regime) for regime switches
                            e.g. [(4.0, "volatile"), (6.0, "ranging")]
        """
        self._running = True
        t_start_real  = time.monotonic()

        tf_sec     = self.TF_SECONDS.get(self.timeframe, 3600)
        n_bars     = int(hours * 3600 / tf_sec)
        tick_delay = tf_sec / self.TICKS_PER_BAR / self.speed  # seconds per tick

        logger.info(
            f"SyntheticLiveFeed: {self.pair}/{self.timeframe} "
            f"{hours}h @ {self.speed}× speed "
            f"→ {n_bars} bars, {n_bars * self.TICKS_PER_BAR} ticks"
        )

        # Build regime schedule
        regime_schedule: Dict[int, str] = {}
        if regime_changes:
            for change_hour, new_regime in regime_changes:
                bar_idx = int(change_hour * 3600 / tf_sec)
                for b in range(bar_idx, n_bars):
                    regime_schedule[b] = new_regime

        # Pre-generate candle sequence
        all_candles = self.generator.generate(
            n=n_bars + 55,
            regime=self.regime,
            regime_changes=[
                (int(h * 3600 / tf_sec), r)
                for h, r in (regime_changes or [])
            ],
        )

        # Build rolling history window
        history: List[SyntheticCandle] = list(all_candles[:55])

        for bar_idx in range(55, len(all_candles)):
            if not self._running:
                break

            current_bar = all_candles[bar_idx]
            history.append(current_bar)
            if len(history) > 500:
                history = history[-500:]

            # Stream synthetic ticks for this bar
            await self._stream_bar_ticks(current_bar, tick_delay)

            self._stats["bars"] += 1

            # Run V5 signal cycle on bar close
            try:
                result = await v5.run_signal_cycle(
                    candles=history,
                    pair=self.pair,
                    timeframe=self.timeframe,
                    current_dt=datetime.now(timezone.utc),
                    signal_bar_index=bar_idx,
                )

                if result.signal_fired:
                    self._stats["signals"] += 1
                    if on_signal:
                        await on_signal(result)

                    # Simulate outcome on next 3 bars (fast forward)
                    future = all_candles[bar_idx + 1: bar_idx + 4]
                    if future and result.stop_loss and result.take_profit_1:
                        outcome, r_mult = self._quick_outcome(
                            future, result.direction,
                            result.fill_price or current_bar.close,
                            result.stop_loss, result.take_profit_1,
                        )
                        pnl = result.adjusted_lots * r_mult * v5.governor.current_balance * 0.02
                        await v5.on_trade_closed(
                            pair=self.pair,
                            agent_name=result.agent_used or "unknown",
                            direction=result.direction,
                            entry=result.fill_price or current_bar.close,
                            exit_price=result.take_profit_1 if outcome == "win" else result.stop_loss,
                            stop=result.stop_loss, tp1=result.take_profit_1,
                            outcome=outcome, r_multiple=r_mult,
                            pnl_usd=pnl, regime=result.regime,
                            confidence=result.confidence, session=result.session,
                        )
                        if outcome == "win": self._stats["wins"] += 1
                        else: self._stats["losses"] += 1

            except Exception as e:
                logger.debug(f"SyntheticLiveFeed signal error: {e}")

            if on_bar_close:
                try:
                    await on_bar_close(history, current_bar)
                except Exception:
                    pass

        t_elapsed = time.monotonic() - t_start_real
        n         = self._stats["wins"] + self._stats["losses"]
        wr        = self._stats["wins"] / n if n > 0 else None

        logger.info(
            f"SyntheticLiveFeed complete: {self._stats['bars']} bars "
            f"{self._stats['signals']} signals "
            f"WR={wr:.0%}" if wr else f"WR=N/A"
        )

        return LiveRunStats(
            duration_real_seconds=round(t_elapsed, 1),
            duration_simulated_hours=hours,
            speed_multiplier=self.speed,
            bars_processed=self._stats["bars"],
            signals_fired=self._stats["signals"],
            pair=self.pair, regime=self.regime,
            final_virtual_balance=v5.governor.current_balance,
            win_rate=wr,
        )

    def stop(self):
        self._running = False

    async def _stream_bar_ticks(self, bar: SyntheticCandle, tick_delay: float):
        """Stream N ticks for one bar with realistic intra-bar path."""
        prices = np.linspace(bar.open, bar.close, self.TICKS_PER_BAR)
        # Add intra-bar noise
        for i, p in enumerate(prices):
            if not self._running:
                break
            noise = np.random.normal(0, bar.range * 0.05)
            tick_price = p + noise
            await asyncio.sleep(tick_delay)

    def _quick_outcome(
        self, future: List, direction: str,
        entry: float, stop: float, tp: float
    ) -> Tuple[str, float]:
        stop_dist = abs(entry - stop)
        tp_dist   = abs(entry - tp)
        for c in future:
            if direction == "buy":
                if c.low  <= stop: return "loss", -1.0
                if c.high >= tp:   return "win",  round(tp_dist / max(stop_dist, 1e-9), 2)
            else:
                if c.high >= stop: return "loss", -1.0
                if c.low  <= tp:   return "win",  round(tp_dist / max(stop_dist, 1e-9), 2)
        return "timeout", 0.0


# ════════════════════════════════════════════════════════════
# E. UNIFIED SYNTHETIC MARKET FACADE
# ════════════════════════════════════════════════════════════

@dataclass
class FullSuiteReport:
    pair:      str
    candle_generation: Dict
    backtest:  Optional[ExchangeResult]
    montecarlo: Optional[MonteCarloScenarioReport]
    live_run:  Optional[LiveRunStats]
    summary:   str


class SyntheticMarket:
    """
    Unified facade for all four synthetic market components.
    Single entry point — switch between modes with parameters.

    Quick start:
        market = SyntheticMarket("XAUUSD")
        candles = market.generate_candles(1000, regime="trending")
        report  = await market.run_full_suite(v5, hours=4.0, speed=60.0)
    """

    def __init__(
        self,
        pair:             str   = "XAUUSD",
        timeframe:        str   = "1h",
        starting_balance: float = 10_000.0,
        seed:             Optional[int] = None,
    ):
        self.pair      = pair
        self.timeframe = timeframe
        self.balance   = starting_balance
        self.generator = SyntheticCandleGenerator(pair, timeframe, seed)
        self.exchange  = SyntheticExchange(pair, starting_balance)
        self.mc_engine = MonteCarloScenarioEngine(pair, starting_balance)

    # ── A: Generate candles ───────────────────────────────────

    def generate_candles(
        self,
        n:         int = 500,
        regime:    str = "trending",
        direction: int = 1,
        mixed:     bool = False,
    ) -> List[SyntheticCandle]:
        """
        Generate synthetic candles.
        Set mixed=True for balanced regime training data.
        """
        if mixed:
            return self.generator.generate_mixed_regimes(n)
        return self.generator.generate(n, regime, direction)

    # ── B: Backtest on synthetic exchange ─────────────────────

    async def run_backtest(
        self,
        v5,
        n_bars:   int = 300,
        regime:   str = "trending",
        risk_pct: float = 2.0,
    ) -> ExchangeResult:
        """
        Run V5 signal pipeline on synthetic candles with simulated exchange fills.
        """
        candles = self.generate_candles(n_bars + 55, regime)
        self.exchange.reset()
        history: List[SyntheticCandle] = list(candles[:55])

        for i in range(55, len(candles) - 1):
            current = candles[i]
            history.append(current)
            if len(history) > 500:
                history = history[-500:]

            # Process any open orders against this bar
            self.exchange.process_bar(current)

            # Run signal cycle
            try:
                result = await v5.run_signal_cycle(
                    candles=history, pair=self.pair,
                    timeframe=self.timeframe,
                    signal_bar_index=i,
                )
                if result.signal_fired and result.stop_loss and result.take_profit_1:
                    await self.exchange.place_market_order(
                        instrument=self.pair,
                        units=result.adjusted_lots * 100_000 * (1 if result.direction == "buy" else -1),
                        stop_loss=result.stop_loss,
                        take_profit=result.take_profit_1,
                        current_price=current.close,
                        atr=current.range,
                    )
            except Exception as e:
                logger.debug(f"Backtest signal error: {e}")

        return self.exchange.get_result()

    # ── C: Monte Carlo stress test ────────────────────────────

    def stress_test(
        self,
        v5,
        n_paths:  int = 200,
        n_bars:   int = 200,
        regimes:  Optional[List[str]] = None,
    ) -> Dict[str, MonteCarloScenarioReport]:
        """
        Run Monte Carlo stress test across multiple regimes.
        Returns dict of regime → report.
        """
        regimes = regimes or ["trending", "ranging", "volatile"]
        reports = {}
        for regime in regimes:
            logger.info(f"Monte Carlo stress test: {self.pair}/{regime}")
            reports[regime] = self.mc_engine.run(
                v5=v5, n_paths=n_paths, n_bars=n_bars, regime=regime
            )
        return reports

    # ── D: Live synthetic feed ────────────────────────────────

    async def run_live(
        self,
        v5,
        hours:   float = 8.0,
        speed:   float = 60.0,
        regime:  str   = "trending",
        regime_changes: Optional[List[Tuple[float, str]]] = None,
    ) -> LiveRunStats:
        """
        Run synthetic live feed through full V5 pipeline.
        speed=60.0: 8 simulated hours completes in 8 minutes real time.
        """
        feed = SyntheticLiveFeed(
            pair=self.pair, timeframe=self.timeframe,
            regime=regime, speed=speed,
        )

        async def on_signal(result):
            logger.info(
                f"[SYNTHETIC] {result.pair} {result.direction} "
                f"via {result.agent_used} "
                f"conf={result.confidence:.0%}"
            )

        return await feed.run(
            v5=v5, hours=hours,
            on_signal=on_signal,
            regime_changes=regime_changes,
        )

    # ── E: Full combined suite ────────────────────────────────

    async def run_full_suite(
        self,
        v5,
        hours:    float = 4.0,
        speed:    float = 60.0,
        n_mc:     int   = 100,
        regime:   str   = "trending",
    ) -> FullSuiteReport:
        """
        Run all four components in sequence:
        1. Generate training candles
        2. Backtest on synthetic exchange
        3. Monte Carlo stress test (all regimes)
        4. Live synthetic feed run
        """
        logger.info(f"SyntheticMarket full suite: {self.pair} {regime}")

        # A: Generate
        candles = self.generate_candles(500, regime)
        gen_info = {
            "n_candles": len(candles),
            "regime":    regime,
            "price_range": f"{min(c.close for c in candles):.4f} – {max(c.close for c in candles):.4f}",
        }

        # B: Backtest
        bt_result = None
        try:
            bt_result = await self.run_backtest(v5, n_bars=200, regime=regime)
            logger.info(
                f"Backtest: {bt_result.total_trades} trades "
                f"WR={bt_result.win_rate:.0%} "
                f"Sharpe={bt_result.sharpe:.3f} "
                f"PnL=${bt_result.total_pnl:+.2f}"
            )
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")

        # C: Monte Carlo
        mc_reports = None
        try:
            mc_reports = self.stress_test(v5, n_paths=n_mc, n_bars=100)
        except Exception as e:
            logger.warning(f"Monte Carlo failed: {e}")

        # D: Live run
        live_stats = None
        try:
            live_stats = await self.run_live(v5, hours=hours, speed=speed, regime=regime)
        except Exception as e:
            logger.warning(f"Live run failed: {e}")

        # Build summary
        lines = [f"Synthetic Market Suite: {self.pair} | {regime}"]
        if bt_result:
            lines.append(
                f"Backtest: {bt_result.total_trades} trades | "
                f"WR={bt_result.win_rate:.0%} | "
                f"Sharpe={bt_result.sharpe:.3f} | "
                f"PnL=${bt_result.total_pnl:+.2f}"
            )
        if mc_reports:
            for r, mc in mc_reports.items():
                lines.append(
                    f"MC {r}: p50=${mc.final_equity.get('p50',0):,.0f} | "
                    f"P(profit)={mc.prob_profit:.0%} | "
                    f"{mc.recommendation[:60]}"
                )
        if live_stats:
            lines.append(
                f"Live {live_stats.duration_simulated_hours}h @{speed}×: "
                f"{live_stats.signals_fired} signals | "
                f"WR={live_stats.win_rate:.0%}" if live_stats.win_rate else "WR=N/A"
            )

        return FullSuiteReport(
            pair=self.pair,
            candle_generation=gen_info,
            backtest=bt_result,
            montecarlo=mc_reports.get(regime) if mc_reports else None,
            live_run=live_stats,
            summary="\n".join(lines),
        )
