from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd
import smc

@dataclass
class Context:
    pair: str
    bias: str
    zone: str
    equilibrium: float
    leg_high: float
    leg_low: float
    dol: float
    in_kill_zone: bool
    pdh: float
    pdl: float

class ContextEngine:
    def __init__(self, swing_length: int = 10, kill_zones_utc=((7, 10), (12, 15))):
        self.swing_length = swing_length
        self.kill_zones_utc = kill_zones_utc

    def build(self, df_m15, now_utc: datetime, pair: str = ""):
        if df_m15 is None or len(df_m15) < self.swing_length * 4:
            return None
        swings = smc.swing_highs_lows(df_m15, swing_length=self.swing_length)
        bc = smc.bos_choch(df_m15, swing_highs_lows=swings, close_break=True)
        bos_col = bc["BOS"].dropna()
        if bos_col.empty:
            return None
        bias = "BULLISH" if bos_col.iloc[-1] == 1 else "BEARISH"
        levels = swings["Level"].dropna()
        if len(levels) < 2:
            return None
        a, b = levels.iloc[-2], levels.iloc[-1]
        leg_low, leg_high = (a, b) if a < b else (b, a)
        equilibrium = leg_low + (leg_high - leg_low) * 0.5
        price = df_m15["close"].iloc[-1]
        zone = "PREMIUM" if price > equilibrium else "DISCOUNT" if price < equilibrium else "EQUILIBRIUM"
        days = df_m15.index.date
        uniq = sorted(set(days))
        if len(uniq) >= 2:
            mask = days == uniq[-2]
            pdh = float(df_m15["high"][mask].max())
            pdl = float(df_m15["low"][mask].min())
        else:
            pdh = float(df_m15["high"].max())
            pdl = float(df_m15["low"].min())
        dol = pdl if bias == "BULLISH" else pdh
        hour = now_utc.replace(tzinfo=timezone.utc).hour
        in_kz = any(s <= hour < e for s, e in self.kill_zones_utc)
        return Context(pair=pair, bias=bias, zone=zone, equilibrium=equilibrium,
                       leg_high=leg_high, leg_low=leg_low, dol=dol,
                       in_kill_zone=in_kz, pdh=pdh, pdl=pdl)
