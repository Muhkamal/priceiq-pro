"""Pre-gate: reject candle windows that would corrupt swing mapping."""
import pandas as pd

def validate_m5(df, pair: str, max_gap_min: float = 35.0, max_spike_mult: float = 8.0):
    """Returns (ok, reason). Bad ticks -> false BOS/CHoCH -> poisoned journal.

    Gap/spike checks judge only the RECENT tail: one old news candle or the
    Friday-close weekend gap must not latch the gate for days.
    """
    if df is None or len(df) < 50:
        return False, "insufficient bars"
    idx = df.index
    if idx.has_duplicates:
        return False, "duplicate timestamps"
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False, "non-positive price"

    # Gap check: only the last ~100 bars (~8h of M5). Old weekend gaps don't count.
    tail = df.iloc[-101:]
    gaps = tail.index.to_series().diff().dt.total_seconds().dropna() / 60.0
    if len(gaps) and gaps.max() > max_gap_min:
        return False, f"gap {gaps.max():.0f}min"

    # Spike check: only the most recent bar vs the ~100 bars before it.
    # (iloc[-1] may be the just-closed bar; a forming bar's range is small,
    #  so it can never false-trigger this.)
    hist = df.iloc[-101:-1]
    last = df.iloc[-1]
    hist_med = (hist["high"] - hist["low"]).median()
    last_rng = last["high"] - last["low"]
    if hist_med <= 0 or last_rng > max_spike_mult * hist_med:
        return False, "spike candle"
    return True, "ok"
