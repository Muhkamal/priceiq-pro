from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PairSpec:
    pip_size: float
    usd_per_pip_per_lot: float

PAIR_SPECS = {
    "EURUSD": PairSpec(0.0001, 10.0),
    "GBPUSD": PairSpec(0.0001, 10.0),
    "AUDUSD": PairSpec(0.0001, 10.0),
    "NZDUSD": PairSpec(0.0001, 10.0),
    "USDCHF": PairSpec(0.0001, 10.9),
    "USDJPY": PairSpec(0.01, 9.0),
    "XAUUSD": PairSpec(0.01, 1.0),
}

SPREADS = {
    "EURUSD": 0.00012,
    "GBPUSD": 0.00015,
    "AUDUSD": 0.00012,
    "NZDUSD": 0.00015,
    "USDCHF": 0.00015,
    "USDJPY": 0.015,
    "XAUUSD": 0.25,
}

def get_spec(pair: str) -> Optional[PairSpec]:
    return PAIR_SPECS.get(pair)

def get_spread(pair: str) -> Optional[float]:
    return SPREADS.get(pair)

def buffer_price(pair: str, pts: float) -> float:
    spec = get_spec(pair)
    return pts * spec.pip_size if spec else 0.0
