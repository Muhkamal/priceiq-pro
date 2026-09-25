"""RangeSweepFade (sunscreen shop): fade failed sweeps of PDH/PDL back to range EQ.

Hypothesis (2026-09-25): XAUUSD continuation showed MFE median 0.46R / MAE
median -1.57R - a mean-reversion fingerprint. This module trades the ranging
regime trend modules refuse: liquidity sweep of the previous day high/low,
rejection close back inside the range, target the equilibrium midpoint.

Design notes:
- Regime gate: engine reports no DOL (ranging) AND current close inside PDH/PDL.
- Trigger: within the last 3 completed M5 bars, a bar wicked beyond PDH (PDL)
  and CLOSED back inside. Wick pierce + body-close back = rejection.
- Current bar must be back inside too (no live sweep in progress).
- TP = ctx.equilibrium; SL beyond sweep extreme + invalidation buffer;
  downstream signed-RR gate (>= MIN_RR_TO_DOL) arbitrates as usual.
- Cooldown 12 bars/pair so a slow bleed isn't re-faded bar after bar.
- Same kill-zone discipline as every other module.
- BACKTEST-ONLY until it earns its way through train -> OOS like everything else.
"""
from .base import EntryModule
from ..core.pairs import buffer_price


class RangeSweepFade(EntryModule):
    name = "RangeSweepFade"
    LOOKBACK = 3    # sweep must be among the last 3 completed bars
    COOLDOWN = 12   # M5 bars between signals on the same pair

    def __init__(self):
        self._cool = {}

    def check(self, df, ctx, config):
        if ctx is None or not ctx.pair or len(df) < 20:
            return None
        if not ctx.in_kill_zone:
            return None
        # Regime: ranging per the engine (no active draw on liquidity)
        if ctx.dol is not None:
            return None
        try:
            pdh, pdl, eq = float(ctx.pdh), float(ctx.pdl), float(ctx.equilibrium)
        except (TypeError, ValueError):
            return None
        if not (pdl < pdh) or not (pdl < eq < pdh):
            return None

        last_i = len(df) - 1
        if last_i - self._cool.get(ctx.pair, -10**9) < self.COOLDOWN:
            return None

        c = df.iloc[-1]
        close = float(c["close"])
        if not (pdl < close < pdh):        # no sweep-in-progress entries
            return None

        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)

        # PDH swept and rejected -> SELL back toward EQ
        sweep_hi = None
        for j in range(max(0, last_i - self.LOOKBACK + 1), last_i + 1):
            b = df.iloc[j]
            if float(b["high"]) > pdh and float(b["close"]) < pdh:
                sweep_hi = max(sweep_hi or 0.0, float(b["high"]))
        if sweep_hi is not None and (sweep_hi + buf - close) > 0:
            self._cool[ctx.pair] = last_i
            return {"direction": "SELL", "ref_price": close,
                    "sl": sweep_hi + buf, "tp": eq,
                    "reason": f"PDH sweep-reject {sweep_hi:.2f}, fade to EQ {eq:.2f}",
                    "module": self.name, "entry": "next_open"}

        # PDL swept and rejected -> BUY back toward EQ
        sweep_lo = None
        for j in range(max(0, last_i - self.LOOKBACK + 1), last_i + 1):
            b = df.iloc[j]
            if float(b["low"]) < pdl and float(b["close"]) > pdl:
                sweep_lo = min(sweep_lo if sweep_lo is not None else float("inf"),
                               float(b["low"]))
        if sweep_lo is not None and (close - (sweep_lo - buf)) > 0:
            self._cool[ctx.pair] = last_i
            return {"direction": "BUY", "ref_price": close,
                    "sl": sweep_lo - buf, "tp": eq,
                    "reason": f"PDL sweep-reject {sweep_lo:.2f}, fade to EQ {eq:.2f}",
                    "module": self.name, "entry": "next_open"}
        return None
