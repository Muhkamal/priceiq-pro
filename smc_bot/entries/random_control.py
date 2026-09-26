"""
RandomControl — a frozen, seeded null-hypothesis baseline.

PURPOSE
-------
Every real entry module must beat this control before its result is treated
as evidence of edge. It runs through the exact same funnel as every other
module (same kill-zone gate, same zone gate, same RR>=2.0 gate, same cost
model, same trade management / partials / time-stop) with two differences
only: which bars it fires on, and which direction it takes. Both of those
decisions are made by a seeded RNG, not by market structure.

FROZEN SPEC (experiments.jsonl, exp-CTRL, 2026-09-26 - do not edit after
the first real comparison run; amendment requires a CORRECTION ledger entry
predating any affected run)
-----------------------------------------------------------------------
- seed: 20260926, fixed per instance. Changing it after seeing how a real
  module compares invalidates the control.
- fire_prob: 0.02 per eligible bar (a bar that already passed ctx/zone/
  kill-zone gating in the runner, same as every other module). Frozen
  BEFORE any control output existed. On the XAUUSD train window this yields
  ~99 control trades (~4,965 gate-passing bars x 0.02) - enough for a
  stable 95th-percentile bootstrap bound.
- direction: uniform 50/50, drawn from the same RNG stream.
- stop distance: ATR(14) on the M5 series at signal time, in the same
  units as price. Volatility-normalized, contains no directional or
  structural information.
- target: fixed at RR=2.0 (entry +/- 2*ATR), so the RR gate is satisfied
  by construction. Isolates "does the market's actual path pay this
  stop/target combination more often than chance" as the only thing tested.

USAGE
-----
Only ever instantiated by backtest/runner.py under `--control`, which
replaces the module list entirely (never combined with real modules in the
same run; never addable via system.yaml entry_modules; never in main.py's
live MODULE_REGISTRY).
"""
import random
from typing import Optional
import pandas as pd
from .base import EntryModule

FROZEN_SEED = 20260926
FROZEN_FIRE_PROB = 0.02
FROZEN_RR = 2.0
ATR_LOOKBACK = 14


class RandomControl(EntryModule):
    name = "RandomControl"

    def __init__(self, seed: int = FROZEN_SEED, fire_prob: float = FROZEN_FIRE_PROB,
                 rr: float = FROZEN_RR, atr_lookback: int = ATR_LOOKBACK):
        self.rng = random.Random(seed)
        self.fire_prob = fire_prob
        self.rr = rr
        self.atr_lookback = atr_lookback
        self._seed_used = seed  # recorded so the ledger entry can quote it verbatim

    def _atr(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.atr_lookback + 1:
            return None
        window = df.iloc[-(self.atr_lookback + 1):]
        highs = window["high"].values
        lows = window["low"].values
        closes = window["close"].values
        trs = []
        for i in range(1, len(window)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)

    def check(self, df: pd.DataFrame, ctx, config) -> Optional[dict]:
        # Deterministic draw order: one fire-decision draw per eligible bar,
        # then (only if firing) one direction draw. Identical RNG stream
        # across --verify's two runs regardless of upstream skips.
        fires = self.rng.random() < self.fire_prob
        if not fires:
            return None

        direction = "BUY" if self.rng.random() < 0.5 else "SELL"

        atr = self._atr(df)
        if atr is None or atr <= 0:
            return None

        ref_price = float(df.iloc[-1]["close"])
        if direction == "BUY":
            sl = ref_price - atr
            tp = ref_price + self.rr * atr
        else:
            sl = ref_price + atr
            tp = ref_price - self.rr * atr

        return {
            "module": self.name,
            "direction": direction,
            "entry": "next_open",
            "ref_price": ref_price,
            "sl": float(sl),
            "tp": float(tp),
            "reason": f"RandomControl seed={self._seed_used} fire_prob={self.fire_prob}",
        }
