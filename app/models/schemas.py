"""
PriceIQ Pro — Pydantic Models v3.2
Merged: M.A.E. Engine + PriceIQ + 3 Priorities
Updated for v3.2 corrected engine compatibility
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class Trend(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    UNKNOWN = "unknown"

class MarketStage(str, Enum):
    ACCUMULATION = "accumulation"
    ADVANCING = "advancing"
    DISTRIBUTION = "distribution"
    DECLINING = "declining"
    UNKNOWN = "unknown"

class PatternType(str, Enum):
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"

class SignalDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"

class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ELITE = "elite"

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

class MarketStructure(BaseModel):
    trend: Trend
    stage: MarketStage
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    trend_strength: float = Field(0.0, ge=0.0, le=1.0)
    is_healthy_trend: bool = False

class SupportResistance(BaseModel):
    level: float
    touches: int = 1
    strength: float = Field(0.0, ge=0.0, le=1.0)
    is_role_reversal: bool = False
    was_support: bool = False
    was_resistance: bool = False

class PatternValidation(BaseModel):
    is_valid: bool
    pattern_type: PatternType
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rejection_reasons: List[str] = []
    recommendations: List[str] = []

class TradeSignal(BaseModel):
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    risk_reward_1: float
    risk_reward_2: Optional[float] = None
    position_size: Optional[Dict[str, Any]] = None
    pattern: PatternType
    confidence: float
    market_structure: MarketStructure
    support_resistance: List[SupportResistance] = []
    explanation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: str = "1h"
    pair: str = "EURUSD"

class PositionSizeRequest(BaseModel):
    account_balance: float = Field(10000.0, gt=0)
    risk_percent: float = Field(2.0, gt=0, le=5.0)
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float = Field(50.0, gt=0)
    pair: str = "EURUSD"
    # v3.2 additions — optional, engine falls back gracefully
    stop_distance: Optional[float] = Field(None, description="ATR-based stop distance in price terms")
    stop_pips: Optional[float] = Field(None, description="Stop distance in pips (alternative to stop_distance)")

class PositionSizeResult(BaseModel):
    standard_lots: float
    mini_lots: float
    micro_lots: float
    units: int
    risk_amount: float
    risk_per_pip: float
    required_margin: float
    leverage_used: float
    margin_percent: float
    pip_value: float
    if_stop_loss: float
    if_tp1_hit: float
    if_tp2_hit: Optional[float] = None
    warnings: List[str] = []
    is_safe: bool = True

class BacktestRequest(BaseModel):
    pair: str = "EURUSD"
    timeframe: str = "1h"
    bars: int = Field(1000, ge=200, le=5000)
    risk_percent: float = 2.0
    account_balance: float = 10000.0
    use_strict_validation: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class TradeRecord(BaseModel):
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: SignalDirection
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    position_size: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    result: str = "open"
    pattern: PatternType
    market_stage: MarketStage
    risk_reward: float
    bars_held: int = 0

class BacktestMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_percent: float = 0.0
    max_drawdown_amount: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_return: float = 0.0
    total_return_percent: float = 0.0
    expectancy: float = 0.0
    expectancy_per_r: float = 0.0
    win_rate_by_pattern: Dict[str, float] = {}
    win_rate_by_stage: Dict[str, float] = {}
    trades_by_month: Dict[str, int] = {}
    pnl_by_month: Dict[str, float] = {}

class BacktestResult(BaseModel):
    verdict: str
    verdict_color: str
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    equity_curve: List[Dict[str, Any]] = []
    parameters: Dict[str, Any] = {}
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class User(BaseModel):
    id: str
    email: str
    tier: SubscriptionTier = SubscriptionTier.FREE
    daily_signals_used: int = 0
    daily_signals_limit: int = 3
    subscription_expires: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SignalHistory(BaseModel):
    id: str
    user_id: str
    signal: TradeSignal
    is_paper_trade: bool = False
    is_win: Optional[bool] = None
    actual_pnl: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WatchlistItem(BaseModel):
    pair: str
    timeframe: str
    added_at: datetime = Field(default_factory=datetime.utcnow)
