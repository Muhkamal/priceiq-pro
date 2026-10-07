"""POI_Retest (exp-012) - canonical SMC sweep-reversal entry. BACKTEST-ONLY.

Implements the frozen exp-012 spec (experiments.jsonl):
  sweep      : low takes out the prior SWING-bar low within SWEEP_LOOKBACK, bar
               closes bullish (reclaim) [symmetric for shorts]
  displacement: first post-sweep bar with body/range >= 0.6 and range >= 1.5x
               ATR(14,M5), within 10 bars of the sweep
  POI        : last opposite-close candle before the displacement bar (order
               block); DEAD if price traded beyond its far edge after the
               displacement (mitigation) - unmitigated required
  tap        : current bar re-enters [OB low, OB high]
  confirm    : post-tap M5 CHoCH proxy - a micro low swept (2nd half of the
               last 2xMICRO bars takes out 1st-half lows) AND current close
               above the prior MICRO-bar high [symmetric inverted]
  entry/exit : next-open fill; SL = sweep extreme +/- 0.25xATR buffer;
               TP = opposing liquidity (pdh / 4h DOL beyond entry);
               runner RR>=2 gate arbitrates
  state      : one trade per sweep event per pair; cooldown 24 bars
EFFICIENCY: all math on a 320-bar tail window - full-series work per call is
O(n) and would make the 10y M5 run take days.
"""
import pandas as pd
from .base import EntryModule
from ..core.pairs import get_spread


class PoiRetracement(EntryModule):
    name = "POI_Retest"
    SWING = 20
    SWEEP_LOOKBACK = 96
    MICRO = 10
    BODY = 0.6
    IMPULSE_ATR = 1.5
    ATR_LEN = 14
    SL_BUF = 0.25
    COOLDOWN = 24
    TAIL = 320

    def __init__(self):
        self._cool = {}
        self._traded_sweeps = {}

    def _atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.rolling(self.ATR_LEN).mean().iloc[-1]

    def check(self, df_full, ctx, config):
        if ctx is None or not ctx.pair or len(df_full) < self.TAIL:
            return None
        if not ctx.in_kill_zone:
            return None
        pair = ctx.pair
        last_i = len(df_full) - 1
        if last_i - self._cool.get(pair, -10**9) < self.COOLDOWN:
            return None
        df = df_full.iloc[-self.TAIL:].reset_index(drop=True)
        n = len(df)
        atr = self._atr(df)
        if not pd.notna(atr) or atr <= 0:
            return None
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        close = float(c.iloc[-1])
        traded = self._traded_sweeps.setdefault(pair, set())

        for long in (True, False):
            # --- locate most recent qualifying sweep bar k ---
            k = None
            lo = max(self.SWING, n - 2 - self.SWEEP_LOOKBACK)
            for kk in range(n - 3, lo, -1):
                prior = l.iloc[kk - self.SWING:kk]
                if long:
                    if l.iloc[kk] < prior.min() and c.iloc[kk] > o.iloc[kk]:
                        k = kk
                        break
                else:
                    if h.iloc[kk] > prior.max() and c.iloc[kk] < o.iloc[kk]:
                        k = kk
                        break
            if k is None or k in traded:
                continue
            sweep_extreme = float(l.iloc[k]) if long else float(h.iloc[k])
            # --- displacement bar within k+1 .. k+10 ---
            j = None
            for jj in range(k + 1, min(k + 11, n - 1)):
                rng = h.iloc[jj] - l.iloc[jj]
                body = abs(c.iloc[jj] - o.iloc[jj])
                if rng >= self.IMPULSE_ATR * atr and body / rng >= self.BODY \
                        and ((c.iloc[jj] > o.iloc[jj]) == long):
                    j = jj
                    break
            if j is None:
                continue
            # --- order block: last opposite-close candle in [k, j-1] ---
            m = None
            for mm in range(j - 1, k - 1, -1):
                if (c.iloc[mm] < o.iloc[mm]) == long:   # opposite to the move
                    m = mm
                    break
            if m is None:
                continue
            ob_lo, ob_hi = float(l.iloc[m]), float(h.iloc[m])
            # --- unmitigated: nothing beyond OB far edge after j ---
            after = df.iloc[j + 1:]
            if len(after) == 0:
                continue
            if long and after["low"].min() < ob_lo:
                continue
            if not long and after["high"].max() > ob_hi:
                continue
            # --- tap: current bar inside the zone ---
            if not (float(l.iloc[-1]) <= ob_hi and close >= ob_lo):
                continue
            # --- confirmation: micro low swept, close through micro high ---
            if n < 2 * self.MICRO + 1:
                continue
            first = l.iloc[n - 2 * self.MICRO:n - self.MICRO]
            second = l.iloc[n - self.MICRO:]
            if long:
                swept = second.min() < first.min()
                through = close > float(h.iloc[n - self.MICRO - 1:n - 1].max())
                ok = swept and through
            else:
                swept = second.max() > first.max()
                through = close < float(l.iloc[n - self.MICRO - 1:n - 1].min())
                ok = swept and through
            if not ok:
                continue
            # --- opposing liquidity target ---
            ref = close + 1.25 * (get_spread(pair) or 0.0) if long else \
                  close - 1.25 * (get_spread(pair) or 0.0)
            tp = None
            for cand in (getattr(ctx, "pdh", None), getattr(ctx, "dol", None)):
                try:
                    cand = float(cand)
                except (TypeError, ValueError):
                    continue
                if long and cand > ref:
                    tp = cand if tp is None else max(tp, cand)
                if not long and cand < ref:
                    tp = cand if tp is None else min(tp, cand)
            if tp is None:
                continue
            sl = sweep_extreme - self.SL_BUF * atr if long else \
                 sweep_extreme + self.SL_BUF * atr
            traded.add(k)
            self._cool[pair] = last_i
            return {"direction": "BUY" if long else "SELL", "ref_price": ref,
                    "sl": float(sl), "tp": float(tp),
                    "reason": f"POI_Retest {'LONG' if long else 'SHORT'} k={k} j={j} "
                              f"OB[{ob_lo:.5f},{ob_hi:.5f}]",
                    "module": self.name, "entry": "next_open"}
        return None
