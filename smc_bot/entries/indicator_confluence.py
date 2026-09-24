"""Indicator-confluence entry module: RSI + Stochastic + Bollinger Bands + MACD.
This replicates the strategy used by actual V75 traders: mean-reversion at extremes
with multi-timeframe confirmation.
"""
from typing import Optional
import pandas as pd
import numpy as np
from .base import EntryModule
from ..core.pairs import buffer_price

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

class IndicatorConfluence(EntryModule):
    name = "Indicator_Confluence"
    
    def check(self, df, ctx, config):
        """Check for indicator confluence signals (mean-reversion at extremes)."""
        if len(df) < 50:
            return None
        
        # Calculate indicators
        rsi = calculate_rsi(df["close"])
        stoch_k, stoch_d = calculate_stochastic(df["high"], df["low"], df["close"])
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df["close"])
        macd_line, signal_line, macd_hist = calculate_macd(df["close"])
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Current indicator values
        curr_rsi = rsi.iloc[-1]
        curr_stoch_k = stoch_k.iloc[-1]
        curr_bb_upper = bb_upper.iloc[-1]
        curr_bb_lower = bb_lower.iloc[-1]
        curr_bb_middle = bb_middle.iloc[-1]
        curr_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2]
        
        buf = buffer_price(ctx.pair, config.invalidation.stop_buffer_pts)
        
        # SELL signal: overbought conditions + rejection
        # RSI >= 70, Stochastic >= 80, price at upper Bollinger, MACD histogram peak, bearish rejection
        sell_rsi = curr_rsi >= 70
        sell_stoch = curr_stoch_k >= 80
        sell_bb = current["high"] >= curr_bb_upper * 0.998  # within 0.2% of upper band
        sell_macd_peak = curr_macd_hist > 0 and curr_macd_hist < prev_macd_hist  # histogram declining from peak
        sell_rejection = (current["high"] - current["close"]) > (current["open"] - current["low"])  # upper wick > lower wick
        
        if sell_rsi and sell_stoch and sell_bb and sell_rejection:
            # Target: Bollinger middle band or fixed 100 pips
            target = min(curr_bb_middle, current["close"] - 1000)  # 1000 points = 100 pips on V75
            sl = current["high"] + buf
            
            # Validate RR
            entry = current["close"]
            risk = sl - entry
            reward = entry - target
            rr = reward / risk if risk > 0 else 0
            
            if rr >= 1.0:  # Lower RR threshold for mean-reversion
                return {
                    "module": self.name,
                    "direction": "SELL",
                    "entry": "next_open",
                    "ref_price": entry,
                    "sl": sl,
                    "tp": target,
                    "reason": f"Indicator confluence SELL: RSI={curr_rsi:.1f}, Stoch={curr_stoch_k:.1f}, BB upper touch"
                }
        
        # BUY signal: oversold conditions + rejection
        # RSI <= 30, Stochastic <= 20, price at lower Bollinger, MACD histogram trough, bullish rejection
        buy_rsi = curr_rsi <= 30
        buy_stoch = curr_stoch_k <= 20
        buy_bb = current["low"] <= curr_bb_lower * 1.002  # within 0.2% of lower band
        buy_macd_trough = curr_macd_hist < 0 and curr_macd_hist > prev_macd_hist  # histogram rising from trough
        buy_rejection = (current["close"] - current["low"]) > (current["high"] - current["open"])  # lower wick > upper wick
        
        if buy_rsi and buy_stoch and buy_bb and buy_rejection:
            # Target: Bollinger middle band or fixed 100 pips
            target = max(curr_bb_middle, current["close"] + 1000)  # 1000 points = 100 pips on V75
            sl = current["low"] - buf
            
            # Validate RR
            entry = current["close"]
            risk = entry - sl
            reward = target - entry
            rr = reward / risk if risk > 0 else 0
            
            if rr >= 1.0:  # Lower RR threshold for mean-reversion
                return {
                    "module": self.name,
                    "direction": "BUY",
                    "entry": "next_open",
                    "ref_price": entry,
                    "sl": sl,
                    "tp": target,
                    "reason": f"Indicator confluence BUY: RSI={curr_rsi:.1f}, Stoch={curr_stoch_k:.1f}, BB lower touch"
                }
        
        return None
