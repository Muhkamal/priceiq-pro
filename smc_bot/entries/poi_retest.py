"""POI_Retest (exp-012) - canonical SMC sweep-reversal entry. BACKTEST-ONLY.

Frozen spec: experiments.jsonl exp-012 + exp-012-corr-1 (zero-fire rule,
plumbing exception) + exp-012-corr-2 (TP = NEAREST opposing pool beyond
cost-adjusted entry; RR>=2 gate applies; nearest < 2R -> rr_skip).
Dedup: one trade per sweep EVENT, keyed by absolute timestamp (tail-window
indices shift every bar - exp-012-corr-1 pre-flight fix).
Rejection counters (_rej) expose why candidates die, per the zero-fire
attribution contingency.
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
        self._rej = {}

    def _bump(self, reason):
        self._rej[reason] = self._rej.get(reason, 0) + 1

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
            self._bump("no_atr")
            return None
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        close = float(c.iloc[-1])
        traded = self._traded_sweeps.setdefault(pair, set())

        for long in (True, False):
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
            if k is None:
                self._bump("no_sweep")
                continue
            abs_k = df_full.index[last_i - (n - 1 - k)]
            if abs_k in traded:
                self._bump("sweep_already_traded")
                continue
            sweep_extreme = float(l.iloc[k]) if long else float(h.iloc[k])
            j = None
            for jj in range(k + 1, min(k + 11, n - 1)):
                rng = h.iloc[jj] - l.iloc[jj]
                body = abs(c.iloc[jj] - o.iloc[jj])
                if rng >= self.IMPULSE_ATR * atr and body / rng >= self.BODY \
                        and ((c.iloc[jj] > o.iloc[jj]) == long):
                    j = jj
                    break
            if j is None:
                self._bump("no_displacement")
                continue
            m = None
            for mm in range(j - 1, k - 1, -1):
                if (c.iloc[mm] < o.iloc[mm]) == long:
                    m = mm
                    break
            if m is None:
                self._bump("no_ob")
                continue
            ob_lo, ob_hi = float(l.iloc[m]), float(h.iloc[m])
            after = df.iloc[j + 1:]
            if len(after) == 0:
                self._bump("no_after")
                continue
            if long and after["low"].min() < ob_lo:
                self._bump("ob_mitigated")
                continue
            if not long and after["high"].max() > ob_hi:
                self._bump("ob_mitigated")
                continue
            if not (float(l.iloc[-1]) <= ob_hi and close >= ob_lo):
                self._bump("no_tap")
                continue
            if n < 2 * self.MICRO + 1:
                self._bump("no_micro")
                continue
            first = l.iloc[n - 2 * self.MICRO:n - self.MICRO]
            second = l.iloc[n - self.MICRO:]
            if long:
                ok = second.min() < first.min() and \
                     close > float(h.iloc[n - self.MICRO - 1:n - 1].max())
            else:
                ok = second.max() > first.max() and \
                     close < float(l.iloc[n - self.MICRO - 1:n - 1].min())
            if not ok:
                self._bump("no_confirm")
                continue
            ref = close + 1.25 * (get_spread(pair) or 0.0) if long else \
                  close - 1.25 * (get_spread(pair) or 0.0)
            tp = None
            for cand in (getattr(ctx, "pdh", None), getattr(ctx, "dol", None)):
                try:
                    cand = float(cand)
                except (TypeError, ValueError):
                    continue
                if long and cand > ref:
                    tp = cand if tp is None else min(tp, cand)   # corr-2: NEAREST
                if not long and cand < ref:
                    tp = cand if tp is None else max(tp, cand)
            if tp is None:
                self._bump("no_tp_candidate")
                continue
            sl = sweep_extreme - self.SL_BUF * atr if long else \
                 sweep_extreme + self.SL_BUF * atr
            traded.add(abs_k)
            self._cool[pair] = last_i
            return {"direction": "BUY" if long else "SELL", "ref_price": ref,
                    "sl": float(sl), "tp": float(tp),
                    "reason": f"POI_Retest {'LONG' if long else 'SHORT'} OB[{ob_lo:.5f},{ob_hi:.5f}]",
                    "module": self.name, "entry": "next_open"}
        return None
