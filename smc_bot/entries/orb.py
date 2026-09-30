"""ORB (Opening Range Breakout) - exp-010. BACKTEST-ONLY until it earns the gate.

Hypothesis (peer-reviewed prior: Zarattini/Barbon/Aziz, SSRN 4729284, US equities
2016-23 - TRANSFER to spot gold is the hypothesis, not the prior). Volume filter
from the source paper DROPPED: OTC forex carries no centralized tape.

FROZEN INTERNAL GATES (listed per exp-008 meta-lesson):
  opening range : first 3 M5 bars from 07:00 UTC (07:00-07:15, London)
  trigger window: closes beyond range from 07:15 while in kill zone (shared gate)
  direction     : agnostic (no bias filter beyond the runner's shared zone gate)
  stop          : opposite side of the opening range
  target        : 2.0R at cost-adjusted effective entry (post-cost RR=2 by
                  construction, exp-CTRL-impl convention)
  lock          : one signal per pair per day
Counterparty (exp-010 registration): Asian-session position holders and pre-London
hedgers forced to re-mark at the open; stale-position exits + breakout stop flow
fuel the move.
"""
from .base import EntryModule
from ..core.pairs import get_spread


class OpeningRangeBreakout(EntryModule):
    name = "ORB"
    OR_BARS = 3          # 07:00-07:15 UTC
    OR_START_HOUR = 7
    RR = 2.0

    def __init__(self):
        self._state = {}   # pair -> {date, hi, lo, fired}
        self._cool = {}

    def check(self, df, ctx, config):
        if ctx is None or not ctx.pair or len(df) < 20:
            return None
        if not ctx.in_kill_zone:
            return None
        t = df.index[-1]
        d = t.date()
        st = self._state.get(ctx.pair)
        if st is None or st["date"] != d:
            day = df[df.index.date == d]
            ors = day[day.index.hour >= self.OR_START_HOUR].iloc[:self.OR_BARS]
            if len(ors) < self.OR_BARS:
                return None                     # range not complete yet
            st = {"date": d, "hi": float(ors["high"].max()),
                  "lo": float(ors["low"].min()), "fired": False}
            self._state[ctx.pair] = st
        if st["fired"]:
            return None
        if t.hour < self.OR_START_HOUR or (t.hour == self.OR_START_HOUR and t.minute < 15):
            return None                         # range still building
        hi, lo = st["hi"], st["lo"]
        if not (hi > lo):
            return None
        last_i = len(df) - 1
        if last_i - self._cool.get(ctx.pair, -10**9) < 12:
            return None
        close = float(df.iloc[-1]["close"])
        spread = get_spread(ctx.pair) or 0.0
        cost = 1.25 * spread
        if close > hi:
            ref = close + cost
            risk = ref - lo
            if risk <= 0:
                return None
            st["fired"] = True
            self._cool[ctx.pair] = last_i
            return {"direction": "BUY", "ref_price": ref, "sl": lo,
                    "tp": ref + self.RR * risk,
                    "reason": f"ORB London {d} close {close:.5f} > range hi {hi:.5f}",
                    "module": self.name, "entry": "next_open"}
        if close < lo:
            ref = close - cost
            risk = hi - ref
            if risk <= 0:
                return None
            st["fired"] = True
            self._cool[ctx.pair] = last_i
            return {"direction": "SELL", "ref_price": ref, "sl": hi,
                    "tp": ref - self.RR * risk,
                    "reason": f"ORB London {d} close {close:.5f} < range lo {lo:.5f}",
                    "module": self.name, "entry": "next_open"}
        return None
