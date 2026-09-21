"""Pre-gate: reject candle windows that would corrupt swing mapping."""
import pandas as pd


def validate_m5(df, pair: str, max_gap_min: float = 35.0, max_spike_mult: float = 8.0):
    """Returns (ok, reason). Bad ticks -> false BOS/CHoCH -> poisoned journal."""
    if df is None or len(df) < 50:
        return False, "insufficient bars"
    idx = df.index
    if idx.has_duplicates:
        return False, "duplicate timestamps"
    gaps = idx.to_series().diff().dt.total_seconds().dropna() / 60.0
    if gaps.max() > max_gap_min:
        return False, f"gap {gaps.max():.0f}min"
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False, "non-positive price"
    rng = df["high"] - df["low"]
    med = rng.median()
    if med <= 0 or rng.max() > max_spike_mult * med:
        return False, "spike candle"
    return True, "ok"
