from typing import Optional
import pandas as pd
from .base import EntryModule
from ..core.pairs import buffer_price

class ChoChNoIDM(EntryModule):
    name = "CHoCH_No_IDM"

    def check(self, df, ctx, config):
        if not ctx.in_kill_zone or ctx.zone == "EQUILIBRIUM":
            return None
        if ctx.bias == "BULLISH" and ctx.zone != "DISCOUNT":
            return None
        if ctx.bias == "BEARISH" and ctx.zone != "PREMIUM":
            return None
        max_ret = getattr(config.params, "max_retracement", 1.0)
        if max_ret < 1.0 and (ctx.retr is None or ctx.retr < 0.5 or ctx.retr > max_ret):
            return None
        if getattr(config.conditions, "require_unmitigated_zone", False) and not ctx.in_poi:
            return None
        if len(df) < 4:
            return None
        c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)
        if ctx.bias == "BULLISH":
            swept_low = c0["low"] < c1["low"] and c0["close"] > c1["low"]
            choch_up = c0["close"] > c2["high"]
            if swept_low and choch_up:
                return {"module": self.name, "direction": "BUY", "entry": "next_open",
                        "ref_price": float(c0["close"]), "sl": float(c0["low"] - buf),
                        "tp": float(ctx.dol),
                        "reason": "Prev-low sweep + close-through CHoCH in discount"}
        else:
            swept_high = c0["high"] > c1["high"] and c0["close"] < c1["high"]
            choch_dn = c0["close"] < c2["low"]
            if swept_high and choch_dn:
                return {"module": self.name, "direction": "SELL", "entry": "next_open",
                        "ref_price": float(c0["close"]), "sl": float(c0["high"] + buf),
                        "tp": float(ctx.dol),
                        "reason": "Prev-high sweep + close-through CHoCH in premium"}
        return None
