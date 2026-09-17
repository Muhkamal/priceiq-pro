"""Market profiles: tells the scanner how each instrument behaves."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MarketProfile:
    source: str                 # "twelvedata" | "deriv"
    deriv_symbol: str = ""
    kill_zones: bool = True     # synthetics trade 24/7 -> False
    dol_mode: str = "calendar"  # "calendar" (PDH/PDL) | "rolling24h"
    stake_based: bool = False   # Deriv uses stake, not lots
    granularity: int = 300      # seconds per candle (M5)

PROFILES = {
    "XAUUSD": MarketProfile(source="twelvedata"),
    "EURUSD": MarketProfile(source="twelvedata"),
    "V75":  MarketProfile(source="deriv", deriv_symbol="R_75",
                          kill_zones=False, dol_mode="rolling24h", stake_based=True),
    "STEP": MarketProfile(source="deriv", deriv_symbol="STPINDX",
                          kill_zones=False, dol_mode="rolling24h", stake_based=True),
}

def get_profile(pair: str) -> Optional[MarketProfile]:
    return PROFILES.get(pair)
