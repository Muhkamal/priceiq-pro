"""TSMOM (time-series momentum / Donchian breakout) - exp-011. BACKTEST-ONLY until gate.

FROZEN INTERNAL GATES (listed per exp-008 meta-lesson):
  entry    : H1 close > max(high, prior 50 H1 bars) -> BUY; < min(low, 50) -> SELL
  stop     : 2.0 x ATR(14,H1) from cost-adjusted effective entry
  target   : 6.0 x ATR (nominal RR=3; post-cost RR>2 clears the runner gate by
             construction, exp-CTRL-impl convention)
  cooldown : 24 bars per pair (no stacking)
Runner machinery applies on top: shared zone/kill/dq gates, BE-at-4R, 20/30/50
slices, time-stop (exp-011 runs use --time-stop-bars 240 = 10 days at H1).
Exit anatomy thus: stop, BE scratch, 4R partial, 6R target, or 10-day time-stop.

Prior: Moskowitz/Ooi/Pedersen 2012 (TSMOM, 58 instruments); Hurst/Ooi/Pedersen
('A Century of Evidence on Trend-Following'); Menkhoff et al (FX momentum).
Counterparty (exp-011 registration): trend-faders and flow-hedgers transacting
against established macro repricing; behavioral underreaction is persistent.
"""
import pandas as pd
from .base import EntryModule
from ..core.pairs import get_spread


class TSMomentum(EntryModule):
    name = "TSMOM"
    DONCHIAN = 50
    ATR_LEN = 14
    ATR_STOP = 2.0
    ATR_TP = 6.0
    COOLDOWN = 24

    def __init__(self):
        self._cool = {}

    def check(self, df, ctx, config):
        if ctx is None or not ctx.pair or len(df) < self.DONCHIAN + self.ATR_LEN + 5:
            return None
        if not ctx.in_kill_zone:
            return None
        last_i = len(df) - 1
        if last_i - self._cool.get(ctx.pair, -10**9) < self.COOLDOWN:
            return None
        close = float(df.iloc[-1]["close"])
        prior = df.iloc[-(self.DONCHIAN + 1):-1]
        hi = float(prior["high"].max())
        lo = float(prior["low"].min())
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.ATR_LEN).mean().iloc[-1]
        if not pd.notna(atr) or atr <= 0 or not (hi > lo):
            return None
        direction = None
        if close > hi:
            direction = "BUY"
        elif close < lo:
            direction = "SELL"
        if direction is None:
            return None
        spread = get_spread(ctx.pair) or 0.0
        cost = 1.25 * spread
        ref = close + cost if direction == "BUY" else close - cost
        if direction == "BUY":
            sl, tp = ref - self.ATR_STOP * atr, ref + self.ATR_TP * atr
        else:
            sl, tp = ref + self.ATR_STOP * atr, ref - self.ATR_TP * atr
        self._cool[ctx.pair] = last_i
        return {"direction": direction, "ref_price": ref, "sl": sl, "tp": tp,
                "reason": f"TSMOM {direction} close {close:.5f} vs 50-bar [{lo:.5f},{hi:.5f}] ATR {atr:.5f}",
                "module": self.name, "entry": "next_open"}
