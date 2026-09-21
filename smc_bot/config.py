import os
from dataclasses import dataclass, field
from typing import List, Tuple
import yaml

@dataclass
class ConditionsConfig:
    kill_zones: Tuple[Tuple[int, int], ...] = ((7, 10), (12, 15))
    require_displacement: bool = False
    require_unmitigated_zone: bool = False
    news_blackout_min: int = 0

@dataclass
class InvalidationConfig:
    stop_buffer_pts: int = 2
    time_stop_bars: int = 12

@dataclass
class RiskConfig:
    pct_per_trade: float = 0.5
    max_correlated_usd: int = 2
    daily_loss_limit_pct: float = 2.0

@dataclass
class ParamsConfig:
    swing_length: int = 10
    continuation_target: str = "dol"
    stop_mode: str = "consolidation"  # "consolidation" | "retest_candle"  # "dol" | "measured_move" | "next_swing"
    kill_zone_buffer_min: int = 0
    max_retracement: float = 1.0

@dataclass
class SystemConfig:
    markets: List[str] = field(default_factory=lambda: ["XAUUSD"])
    context_tf: str = "M15"
    trigger_tf: str = "M5"
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    entry_modules: List[str] = field(default_factory=lambda: ["choch_no_idm"])
    invalidation: InvalidationConfig = field(default_factory=InvalidationConfig)
    partials: List[dict] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    params: ParamsConfig = field(default_factory=ParamsConfig)

def load_config(path: str = None) -> SystemConfig:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "system.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    kz = raw.get("conditions", {}).get("kill_zones", [[7, 10], [12, 15]])
    return SystemConfig(
        markets=raw.get("markets", ["XAUUSD"]),
        context_tf=raw.get("context_tf", "M15"),
        trigger_tf=raw.get("trigger_tf", "M5"),
        conditions=ConditionsConfig(
            kill_zones=tuple(tuple(z) for z in kz),
            require_displacement=raw.get("conditions", {}).get("require_displacement", False),
            require_unmitigated_zone=raw.get("conditions", {}).get("require_unmitigated_zone", False),
            news_blackout_min=raw.get("conditions", {}).get("news_blackout_min", 0),
        ),
        entry_modules=raw.get("entry_modules", ["choch_no_idm"]),
        invalidation=InvalidationConfig(**(raw.get("invalidation", {}) or {})),
        partials=raw.get("partials", []),
        risk=RiskConfig(**(raw.get("risk", {}) or {})),
        params=ParamsConfig(**(raw.get("params", {}) or {})),
    )
