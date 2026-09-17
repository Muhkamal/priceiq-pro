from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import logging
import pandas as pd
from smartmoneyconcepts import smc

logger = logging.getLogger(__name__)

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
    retr: Optional[float] = None
    in_poi: bool = False

class ContextEngine:
    def __init__(self, swing_length: int = 10, kill_zones_utc=((7, 10), (12, 15))):
        self.swing_length = swing_length
        self.kill_zones_utc = kill_zones_utc

    def build(self, df_m15, now_utc: datetime, pair: str = "", profile=None,
              compute_poi: bool = False):
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
        s_prev, s_last = levels.iloc[-2], levels.iloc[-1]
        leg_low, leg_high = (s_prev, s_last) if s_prev < s_last else (s_last, s_prev)
        equilibrium = leg_low + (leg_high - leg_low) * 0.5
        price = float(df_m15["close"].iloc[-1])
        zone = "PREMIUM" if price > equilibrium else "DISCOUNT" if price < equilibrium else "EQUILIBRIUM"

        retr = None
        if bias == "BULLISH" and s_last > s_prev:
            rng = s_last - s_prev
            retr = (s_last - price) / rng if rng > 0 else None
        elif bias == "BEARISH" and s_last < s_prev:
            rng = s_prev - s_last
            retr = (price - s_last) / rng if rng > 0 else None

        dol_mode = getattr(profile, "dol_mode", "calendar")
        if dol_mode == "rolling24h":
            cutoff = df_m15.index[-1] - pd.Timedelta(hours=24)
            recent = df_m15[df_m15.index > cutoff]
            if recent.empty:
                recent = df_m15
            pdh = float(recent["high"].max()); pdl = float(recent["low"].min())
        else:
            days = df_m15.index.date
            uniq = sorted(set(days))
            if len(uniq) >= 2:
                mask = days == uniq[-2]
                pdh = float(df_m15["high"][mask].max()); pdl = float(df_m15["low"][mask].min())
            else:
                pdh = float(df_m15["high"].max()); pdl = float(df_m15["low"].min())
        dol = pdh if bias == "BULLISH" else pdl

        if profile is not None and not profile.kill_zones:
            in_kz = True
        else:
            hour = now_utc.replace(tzinfo=timezone.utc).hour
            in_kz = any(s <= hour < e for s, e in self.kill_zones_utc)

        in_poi = False
        if compute_poi:
            try:
                want = 1 if bias == "BULLISH" else -1
                ob = smc.ob(df_m15, swing_highs_lows=swings, close_mitigation=False)
                fvg = smc.fvg(df_m15)
                if not ob.index.equals(df_m15.index):
                    raise ValueError("OB index misaligned with df_m15")
                start_idx = max(0, len(df_m15) - 500)
                for i in range(start_idx, len(ob)):
                    r = ob.iloc[i]
                    if pd.isna(r["Top"]) or r["OB"] != want:
                        continue
                    zb, zt = r["Bottom"], r["Top"]
                    after = df_m15.iloc[i + 1:]
                    if len(after):
                        if want == 1 and (after["low"] < zb).any():
                            continue
                        if want == -1 and (after["high"] > zt).any():
                            continue
                    h = zt - zb
                    if h <= 0:
                        continue
                    if (zb - 0.15 * h) <= price <= (zt + 0.15 * h):
                        in_poi = True
                        break
                if not in_poi:
                    for i in range(len(fvg)):
                        r = fvg.iloc[i]
                        if pd.isna(r["Top"]) or r["FVG"] != want:
                            continue
                        mit = r["MitigatedIndex"]
                        if not pd.isna(mit) and int(mit) != 0:
                            continue
                        zb, zt = r["Bottom"], r["Top"]
                        h = zt - zb
                        if h <= 0:
                            continue
                        if (zb - 0.15 * h) <= price <= (zt + 0.15 * h):
                            in_poi = True
                            break
            except Exception as e:
                logger.warning(f"POI detection failed for {pair}: {e}")
                in_poi = False

        return Context(pair=pair, bias=bias, zone=zone, equilibrium=equilibrium,
                       leg_high=leg_high, leg_low=leg_low, dol=dol,
                       in_kill_zone=in_kz, pdh=pdh, pdl=pdl, retr=retr, in_poi=in_poi)
