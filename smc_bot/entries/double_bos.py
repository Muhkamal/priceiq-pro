"""RVG 'Double Breakout': 1 BO (body close through resistance) -> 2 BO (close
through a higher swing = expansion proof) -> entry on confirmed retest of the
1 BO flip zone.

Design notes (agreed, do not regress):
- Golden-zone/POI gates INTENTIONALLY not applied: momentum-continuation module,
  diversifies signal source into trending regimes where CHoCH modules are silent.
- Breakouts require BODY CLOSE beyond level; wick pierces are false breaks.
- Close back through level1 before retest = failed breakout -> stand down.
- Bounded retest window (doublebo_retest_window, default 48 M5 bars).
- Stop anchored to pre-breakout extreme + retest candle, never at the level.
- TP stays bot-standard DOL; signed-RR gate arbitrates downstream.
- State is keyed PER PAIR: one module instance serves all pairs in live mode.
"""
from .base import EntryModule
from ..core.pairs import buffer_price

# Retest touch tolerance as a fraction of the level (5 basis points):
# how close price must come to level1 to count as a retest.
# ~6 pips on EURUSD, ~22 points on V75 at current prices.
# Named so tuning is findable; promote to per-pair config only if
# Experiment F shows the touch rate is pair-sensitive.
RETEST_TOL_FRAC = 0.0005


class DoubleBreakout(EntryModule):
    name = "Double_BOS"

    def __init__(self):
        self.k = 3  # fractal half-width; swing at i confirmed only when i+k <= last bar
        self._states = {}

    def _new_state(self):
        return {"stage": 0, "level1": None, "level2": None,
                "cons_extreme": None, "bars_since_bo": 0}

    def _swings(self, df, direction):
        """Confirmed fractal swings within the last 96 bars (older = stale by rule)."""
        h, l = df["high"].values, df["low"].values
        n = len(df)
        out = []
        for i in range(max(self.k, n - 96 - self.k), n - self.k):
            if direction == 1 and h[i] == h[i - self.k:i + self.k + 1].max():
                out.append((i, float(h[i])))
            elif direction == -1 and l[i] == l[i - self.k:i + self.k + 1].min():
                out.append((i, float(l[i])))
        return out

    def _find_next_swing(self, df, current_level, direction):
        """Find the next confirmed swing target beyond current_level.
        Returns None if no further swing is found (fallback to DOL downstream)."""
        swings = self._swings(df, direction)
        for i, lvl in swings:
            if direction == 1 and lvl > current_level:
                return lvl
            elif direction == -1 and lvl < current_level:
                return lvl
        return None

    def check(self, df, ctx, config):
        if ctx is None or not ctx.pair or len(df) < 2 * self.k + 10:
            return None
        if not ctx.in_kill_zone or ctx.dol is None:
            return None
        st = self._states.setdefault(ctx.pair, self._new_state())
        window = getattr(config.params, "doublebo_retest_window", 48)
        c = df.iloc[-1]
        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)
        if ctx.bias == "BULLISH":
            return self._run(df, ctx, c, buf, window, 1, st, config)
        if ctx.bias == "BEARISH":
            return self._run(df, ctx, c, buf, window, -1, st, config)
        return None

    def _run(self, df, ctx, c, buf, window, direction, st, config):
        close, hi, lo = float(c["close"]), float(c["high"]), float(c["low"])
        last = len(df) - 1
        swings = self._swings(df, direction)

        if st["stage"] == 0:
            for i, lvl in reversed(swings):
                if last - i > 96:
                    continue
                if (close > lvl + buf) if direction == 1 else (close < lvl - buf):
                    st["level1"] = lvl
                    a, b = max(0, i - 6), i + 1
                    st["cons_extreme"] = (float(df["low"].iloc[a:b].min()) if direction == 1
                                          else float(df["high"].iloc[a:b].max()))
                    st["stage"], st["bars_since_bo"] = 1, 0
                    break
            return None

        st["bars_since_bo"] += 1
        failed = (close < st["level1"] - buf) if direction == 1 else (close > st["level1"] + buf)
        if failed or st["bars_since_bo"] > window:
            self._states[ctx.pair] = self._new_state()
            return None

        if st["stage"] == 1:
            for i, lvl in reversed(swings):
                if last - i > 96:
                    continue
                beyond = (lvl > st["level1"] + 2 * buf) if direction == 1 else (lvl < st["level1"] - 2 * buf)
                brk = (close > lvl + buf) if direction == 1 else (close < lvl - buf)
                if beyond and brk:
                    st["level2"] = lvl
                    st["stage"], st["bars_since_bo"] = 2, 0
                    break
            return None

        tol = max(buf, RETEST_TOL_FRAC * st["level1"])
        touched = (lo <= st["level1"] + tol) if direction == 1 else (hi >= st["level1"] - tol)
        confirmed = (close > st["level1"]) if direction == 1 else (close < st["level1"])
        if touched and confirmed:
            lvl1 = st["level1"]
            
            # Compute target based on continuation_target config
            target_mode = getattr(config.params, "continuation_target", "dol")
            tp = None
            if target_mode == "measured_move" and st["level2"] is not None:
                distance = abs(st["level2"] - lvl1)
                tp = lvl1 + distance * 2 if direction == 1 else lvl1 - distance * 2
            elif target_mode == "next_swing":
                tp = self._find_next_swing(df, st["level2"] if st["level2"] else lvl1, direction)
            
            # Fallback to DOL if mode is "dol" or if next_swing/measured_move failed
            if tp is None:
                tp = float(ctx.dol)
            
            stop_mode = getattr(config.params, "stop_mode", "consolidation")
            if direction == 1:
                sl_val = (lo - buf) if stop_mode == "retest_candle" else (min(st["cons_extreme"], lo) - buf)
                sig = {"direction": "BUY", "ref_price": close,
                       "sl": sl_val, "tp": tp,
                       "reason": f"1BO/2BO double breakout, RBS retest of {lvl1:.2f} ({target_mode})"}
            else:
                sl_val = (hi + buf) if stop_mode == "retest_candle" else (max(st["cons_extreme"], hi) + buf)
                sig = {"direction": "SELL", "ref_price": close,
                       "sl": sl_val, "tp": tp,
                       "reason": f"1BO/2BO double breakout, SBR retest of {lvl1:.2f} ({target_mode})"}
            sig.update({"module": self.name, "entry": "next_open"})
            self._states[ctx.pair] = self._new_state()
            return sig
        return None
