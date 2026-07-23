"""
PriceIQ Pro V5 — Shared Test Fixtures

Provides reusable fixtures for all test modules:
    - trending_candles: 100-bar uptrend
    - ranging_candles:  100-bar sideways
    - volatile_candles: 60-bar high-volatility
    - xauusd_candles:   100-bar XAUUSD-priced candles
    - mock_v5:          minimal V5 orchestrator mock
    - mock_telegram:    Telegram that captures messages
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Add v5 package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Minimal Candle ────────────────────────────────────────────

@dataclass
class Candle:
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float = 1000.0
    timestamp: str = ""
    is_bullish: bool = True
    is_bearish: bool = False
    body:   float = 0.0
    range:  float = 0.0
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0

    def __post_init__(self):
        self.is_bullish = self.close >= self.open
        self.is_bearish = not self.is_bullish
        self.body       = abs(self.close - self.open)
        self.range      = self.high - self.low
        self.upper_shadow = self.high - max(self.open, self.close)
        self.lower_shadow = min(self.open, self.close) - self.low
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _make_candles(
    n: int = 100,
    base: float = 1.0800,
    trend: float = 0.0001,
    noise: float = 0.0003,
    volume_base: float = 1000.0,
) -> List[Candle]:
    """Generate synthetic candles with trend + noise."""
    candles = []
    price   = base
    now     = datetime.now(timezone.utc)
    for i in range(n):
        noise_val = np.random.normal(0, noise)
        open_     = price
        close     = price + trend + noise_val
        high      = max(open_, close) + abs(noise_val) * 0.5
        low       = min(open_, close) - abs(noise_val) * 0.5
        vol       = volume_base + np.random.randint(-200, 500)
        ts        = (now - timedelta(hours=n - i)).isoformat()
        candles.append(Candle(open=open_, high=high, low=low, close=close,
                              volume=float(vol), timestamp=ts))
        price = close
    return candles


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def trending_candles() -> List[Candle]:
    """100-bar uptrend — TrendAgent should fire."""
    np.random.seed(42)
    return _make_candles(100, base=1.08, trend=0.0005, noise=0.0002)


@pytest.fixture
def ranging_candles() -> List[Candle]:
    """100-bar sideways — MeanReversionAgent should fire."""
    np.random.seed(42)
    candles = []
    price   = 1.0800
    now     = datetime.now(timezone.utc)
    for i in range(100):
        # Oscillate around mean
        noise    = np.random.normal(0, 0.0004)
        revert   = (1.0800 - price) * 0.05
        price   += noise + revert
        open_    = price
        close    = price + np.random.normal(0, 0.0002)
        high     = max(open_, close) + 0.0001
        low      = min(open_, close) - 0.0001
        ts       = (now - timedelta(hours=100-i)).isoformat()
        candles.append(Candle(open=open_, high=high, low=low, close=close,
                              volume=1000.0, timestamp=ts))
    return candles


@pytest.fixture
def volatile_candles() -> List[Candle]:
    """60-bar high-volatility (NFP-style)."""
    np.random.seed(42)
    candles = []
    price   = 1920.0
    now     = datetime.now(timezone.utc)
    for i in range(60):
        move  = np.random.normal(0, 5.0)
        open_ = price
        close = price + move
        high  = max(open_, close) + abs(move) * 0.8
        low   = min(open_, close) - abs(move) * 0.8
        ts    = (now - timedelta(hours=60-i)).isoformat()
        candles.append(Candle(open=open_, high=high, low=low, close=close,
                              volume=5000.0, timestamp=ts))
        price = close
    return candles


@pytest.fixture
def xauusd_candles() -> List[Candle]:
    """100-bar XAUUSD-priced candles with slight uptrend."""
    np.random.seed(42)
    return _make_candles(100, base=1920.0, trend=0.5, noise=3.0)


@pytest.fixture
def mock_telegram():
    """Telegram that captures sent messages without actually sending."""
    tg = MagicMock()
    tg.send_message = AsyncMock(return_value=None)
    tg.send_signal_alert = AsyncMock(return_value=None)
    tg.send_no_signal_update = AsyncMock(return_value=None)
    tg.send_startup_message = AsyncMock(return_value=None)
    tg.messages = []

    async def capture(msg):
        tg.messages.append(msg)
    tg.send_message.side_effect = capture
    return tg


@pytest.fixture
def mock_learning_loop():
    """Learning loop that captures updates."""
    ll = MagicMock()
    ll.update = MagicMock(return_value=None)
    ll.get_confidence_threshold = MagicMock(return_value=0.55)
    ll.get_agent_weights = MagicMock(return_value={
        "TrendAgent": 1.0, "MeanReversionAgent": 1.0,
        "BreakoutAgent": 1.0, "LiquidityTrapAgent": 1.0,
    })
    ll.get_full_stats = MagicMock(return_value={"total_trades": 0, "agent_stats": {}})
    ll.store = MagicMock()
    ll.store.all = MagicMock(return_value=[])
    ll.learner = MagicMock()
    ll.learner._agent_weights = {}
    ll.learner.get_pair_stats = MagicMock(return_value={"trades": 0})
    return ll


@pytest.fixture
def tmp_json_path(tmp_path):
    """Temporary path for JSON persistence files."""
    return str(tmp_path / "test_state.json")
