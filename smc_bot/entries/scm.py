from typing import Optional
import pandas as pd
from .base import EntryModule
from ..core.pairs import buffer_price

class SingleCandleMitigation(EntryModule):
    name = "SCM"

    def check(self, df, ctx, config):
        if not ctx.in_kill_zone or ctx.zone == "EQUILIBRIUM":
            return None
        if ctx.bias == "BULLISH" and ctx.zone != "DISCOUNT":
            return None
        if ctx.bias == "BEARISH" and ctx.zone != "PREMIUM":
            return None
        if ctx.dol is None:
            return None
        max_ret = getattr(config.params, "max_retracement", 1.0)
        if max_ret < 1.0 and (ctx.retr is None or ctx.retr < 0.5 or ctx.retr > max_ret):
            return None
        if getattr(config.conditions, "require_unmitigated_zone", False) and not ctx.in_poi:
            return None
        if len(df) < 4:
            return None
        c0, c1 = df.iloc[-1], df.iloc[-2]
        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)
        if ctx.bias == "BULLISH":
            swept_major = c1["low"] < ctx.pdl
            mitigates = c0["open"] <= max(c1["open"], c1["close"]) and c0["close"] > c1["high"]
            if swept_major and mitigates:
                return {"module": self.name, "direction": "BUY", "entry": "next_open",
                        "ref_price": float(c0["close"]), "sl": float(c1["low"] - buf),
                        "tp": float(ctx.dol),
                        "reason": "PDL sweep + single-candle mitigation in discount"}
        else:
            swept_major = c1["high"] > ctx.pdh
            mitigates = c0["open"] >= min(c1["open"], c1["close"]) and c0["close"] < c1["low"]
            if swept_major and mitigates:
                return {"module": self.name, "direction": "SELL", "entry": "next_open",
                        "ref_price": float(c0["close"]), "sl": float(c1["high"] + buf),
                        "tp": float(ctx.dol),
                        "reason": "PDH sweep + single-candle mitigation in premium"}
        return None
