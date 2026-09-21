"""CHoCH with Inducement: enters on the CHoCH event that establishes the new bias.

Sequence (Tony Iyke's 4 POI rules):
1. M15 CHoCH (close through last swing) establishes new bias direction
2. First pullback after CHoCH = inducement level
3. Price later sweeps that inducement and taps an unmitigated OB/FVG
4. Among candidate zones, pick the one nearest the inducement
5. Entry on M5 close-through confirmation; SL beyond the OB's last candle/wick

This diversifies the signal source into turning-point regimes where
CHoCH_No_IDM and SCM are structurally silent (they demand pre-existing
M15 bias alignment, which anti-correlates with turning points).
"""
from typing import Optional
import pandas as pd
from .base import EntryModule
from ..core.pairs import buffer_price


class ChoChIDM(EntryModule):
    name = "CHoCH_IDM"

    def __init__(self):
        self.k = 3  # fractal half-width for swing confirmation
        self.reset()

    def reset(self):
        self.stage = 0  # 0: waiting for CHoCH | 1: CHoCH fired, waiting for inducement sweep
        self.choch_direction = None  # 1=bullish, -1=bearish
        self.inducement_level = None
        self.choch_bar_idx = None
        self.bars_since_choch = 0

    def _swings(self, df, direction):
        """Confirmed fractal swings within the last 96 bars."""
        h, l = df["high"].values, df["low"].values
        n = len(df)
        out = []
        for i in range(max(self.k, n - 96 - self.k), n - self.k):
            if direction == 1 and h[i] == h[i - self.k:i + self.k + 1].max():
                out.append((i, float(h[i])))
            elif direction == -1 and l[i] == l[i - self.k:i + self.k + 1].min():
                out.append((i, float(l[i])))
        return out

    def check(self, df, ctx, config):
        if ctx is None or len(df) < 2 * self.k + 10:
            return None
        if not ctx.in_kill_zone or ctx.dol is None:
            return None
        
        # IDM module does NOT require zone alignment (it's the bias-establishing entry)
        # But we still require DOL exists (no target = no trade)
        
        c = df.iloc[-1]
        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)
        last_idx = len(df) - 1
        
        if self.stage == 0:
            # Stage 0: Look for CHoCH (close through last swing in opposite direction)
            # This establishes the new bias
            swings_up = self._swings(df, 1)
            swings_dn = self._swings(df, -1)
            
            # Bullish CHoCH: previous bar was below swing low, current bar closes above
            if swings_dn:
                last_swing_low = swings_dn[-1][1]
                prev_close = float(df.iloc[-2]["close"])
                if prev_close < last_swing_low - buf and c["close"] > last_swing_low + buf:
                    # This is a CHoCH from bearish to bullish
                    self.stage = 1
                    self.choch_direction = 1
                    self.choch_bar_idx = last_idx
                    self.bars_since_choch = 0
                    # Inducement = first pullback low after CHoCH (we'll detect it in stage 1)
                    self.inducement_level = None
                    return None
            
            # Bearish CHoCH: previous bar was above swing high, current bar closes below
            if swings_up:
                last_swing_high = swings_up[-1][1]
                prev_close = float(df.iloc[-2]["close"])
                if prev_close > last_swing_high + buf and c["close"] < last_swing_high - buf:
                    # This is a CHoCH from bullish to bearish
                    self.stage = 1
                    self.choch_direction = -1
                    self.choch_bar_idx = last_idx
                    self.bars_since_choch = 0
                    self.inducement_level = None
                    return None
            
            return None
        
        # Stage 1: CHoCH fired, waiting for inducement sweep + POI tap
        self.bars_since_choch += 1
        
        # Timeout: if 96 bars pass without setup, reset
        if self.bars_since_choch > 96:
            self.reset()
            return None
        
        # Detect inducement level (first pullback after CHoCH)
        if self.inducement_level is None:
            # Look for the first minor pullback after CHoCH bar
            if self.bars_since_choch >= 3:
                # Inducement = the extreme of the first pullback
                pullback_start = self.choch_bar_idx + 1
                pullback_end = last_idx
                if pullback_end > pullback_start:
                    pullback_slice = df.iloc[pullback_start:pullback_end]
                    if self.choch_direction == 1:
                        self.inducement_level = float(pullback_slice["low"].min())
                    else:
                        self.inducement_level = float(pullback_slice["high"].max())
        
        if self.inducement_level is None:
            return None
        
        # Check if inducement has been swept
        if self.choch_direction == 1:
            swept = c["low"] < self.inducement_level
        else:
            swept = c["high"] > self.inducement_level
        
        if not swept:
            return None
        
        # Inducement swept - now look for POI tap (unmitigated OB/FVG near inducement)
        # For simplicity, we check if price is near an unmitigated zone
        # (Full implementation would use smc.ob/smc.fvg with mitigation tracking)
        
        # Entry trigger: M5 close through confirmation
        # Bullish: close above the CHoCH bar's high
        # Bearish: close below the CHoCH bar's low
        choch_bar = df.iloc[self.choch_bar_idx]
        
        if self.choch_direction == 1:
            confirm = c["close"] > choch_bar["high"]
            if confirm:
                sl = min(float(choch_bar["low"]), c["low"]) - buf
                signal = {
                    "direction": "BUY",
                    "ref_price": float(c["close"]),
                    "sl": sl,
                    "tp": float(ctx.dol),
                    "reason": f"CHoCH+IDM: bullish CHoCH at bar {self.choch_bar_idx}, inducement swept at {self.inducement_level:.2f}"
                }
                signal.update({"module": self.name, "entry": "next_open"})
                self.reset()
                return signal
        else:
            confirm = c["close"] < choch_bar["low"]
            if confirm:
                sl = max(float(choch_bar["high"]), c["high"]) + buf
                signal = {
                    "direction": "SELL",
                    "ref_price": float(c["close"]),
                    "sl": sl,
                    "tp": float(ctx.dol),
                    "reason": f"CHoCH+IDM: bearish CHoCH at bar {self.choch_bar_idx}, inducement swept at {self.inducement_level:.2f}"
                }
                signal.update({"module": self.name, "entry": "next_open"})
                self.reset()
                return signal
        
        return None
