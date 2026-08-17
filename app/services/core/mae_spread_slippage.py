"""
M.A.E. Spread & Slippage Modeller
Applies realistic worst-case fill prices per pair.
Add this as a post-processing step to ANY signal before it goes out.
"""

from typing import Dict

SPREAD_PIPS: Dict[str, float] = {
    "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.8,
    "AUDUSD": 1.4, "NZDUSD": 1.8, "USDCAD": 2.0, "EURGBP": 1.6,
    "EURJPY": 1.8, "GBPJPY": 2.5, "XAUUSD": 25.0, "BTCUSD": 50.0,
}

SLIPPAGE_PIPS: Dict[str, float] = {
    "EURUSD": 0.5, "GBPUSD": 0.7, "USDJPY": 0.6, "USDCHF": 0.8,
    "AUDUSD": 0.6, "NZDUSD": 0.8, "USDCAD": 0.9, "EURGBP": 0.7,
    "EURJPY": 0.9, "GBPJPY": 1.2, "XAUUSD": 10.0, "BTCUSD": 20.0,
}

def _pip_size(pair: str) -> float:
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair:
        return 0.1
    if "BTC" in pair:
        return 1.0
    return 0.0001

def spread_cost(pair: str) -> float:
    """Spread in price units."""
    return SPREAD_PIPS.get(pair.upper(), 2.0) * _pip_size(pair)

def slippage_cost(pair: str) -> float:
    """Slippage in price units."""
    return SLIPPAGE_PIPS.get(pair.upper(), 1.0) * _pip_size(pair)

def realistic_entry(entry: float, direction: str, pair: str) -> float:
    """
    Adjust entry price for spread + slippage (worst realistic fill).
    direction: 'buy' or 'sell'
    """
    cost = spread_cost(pair) + slippage_cost(pair)
    if direction.lower() == "buy":
        return entry + cost
    else:
        return entry - cost

def realistic_cost_summary(pair: str) -> dict:
    """Return human-readable cost breakdown for logging."""
    sp = spread_cost(pair)
    sl = slippage_cost(pair)
    return {
        "pair": pair,
        "spread_pips": SPREAD_PIPS.get(pair.upper(), 2.0),
        "slippage_pips": SLIPPAGE_PIPS.get(pair.upper(), 1.0),
        "spread_price": round(sp, 6),
        "slippage_price": round(sl, 6),
        "total_cost": round(sp + sl, 6),
    }
