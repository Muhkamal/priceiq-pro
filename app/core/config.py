"""
PriceIQ Pro v3.2 — Complete Configuration
Updated for MAETradingFormula v3.2 corrected engine
"""

from pydantic_settings import BaseSettings
from typing import List
import os
import json


class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "PriceIQ Pro"
    APP_VERSION: str = "3.2.0"
    DEBUG: bool = False

    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change_this_in_production")

    # Database
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # Data Sources
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")  # NEW: 800 req/day
    
    # Brokers
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_PAPER_TRADING: bool = True
    
    # Deriv (Optional)
    DERIV_API_TOKEN: str = os.getenv("DERIV_API_TOKEN", "")
    DERIV_APP_ID: str = os.getenv("DERIV_APP_ID", "1089")

    # Execution
    AUTO_EXECUTE_TRADES: bool = False

    # AI / LLM
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS_SIGNAL: int = 600
    CLAUDE_MAX_TOKENS_BACKTEST: int = 900

    # Alerts
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = 587
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    ALERT_EMAIL: str = os.getenv("ALERT_EMAIL", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Billing
    PAYSTACK_SECRET_KEY: str = os.getenv("PAYSTACK_SECRET_KEY", "")
    PAYSTACK_PUBLIC_KEY: str = os.getenv("PAYSTACK_PUBLIC_KEY", "")
    FREE_DAILY_SIGNALS: int = 3
    STARTER_DAILY_SIGNALS: int = 10
    PRO_DAILY_SIGNALS: int = 50
    ELITE_DAILY_SIGNALS: int = 999

    # Risk Management
    DEFAULT_RISK_PERCENT: float = 2.0
    DEFAULT_ACCOUNT_BALANCE: float = 10_000.0
    DEFAULT_LEVERAGE: float = 50.0
    MAX_POSITION_LOTS: float = 10.0

    # Pattern Validation (v3.2)
    MIN_SHADOW_RATIO: float = 2.0
    MIN_ENGULFING_RATIO: float = 1.5
    DOJI_BODY_PCT: float = 0.05

    # ATR-Based Stops (v3.2)
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_TP_MULTIPLIER: float = 3.0

    # Spread / Entry (v3.2)
    APPLY_SPREAD_TO_ENTRY: bool = True

    # Session Filter (v3.2)
    SESSION_FILTER_ENABLED: bool = True
    MIN_SESSION_QUALITY: float = 0.6

    # Support/Resistance (v3.2)
    SR_CLUSTER_GAP_PCT: float = 0.002
    SR_MAX_LEVELS: int = 10
    SR_LOOKBACK_CANDLES: int = 200

    # Multi-Candle Patterns (v3.2)
    MORNING_STAR_GAP_PCT: float = 0.001

    # Legacy / Unused (kept for backward compatibility)
    MIN_RISK_REWARD: float = 1.5
    MAX_RISK_REWARD: float = 3.0
    STRICT_VALIDATION: bool = True
    MAX_SHADOW_RATIO: float = 3.5
    MIN_TREND_BARS: int = 3
    MIN_LONGER_TREND_BARS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.50
    BAR_CLOSED_REQUIRED: bool = True
    SPREAD_MODEL_ENABLED: bool = True
    DEFAULT_SLIPPAGE_PIPS: float = 0.5
    PSYCHOLOGICAL_LEVELS_ENABLED: bool = True
    ORDER_BLOCKS_ENABLED: bool = True
    VOLUME_PROFILE_ENABLED: bool = True
    SR_CLUSTER_THRESHOLD: float = 0.005
    BACKTEST_MIN_BARS: int = 200
    BACKTEST_DEFAULT_BARS: int = 1_000
    WALK_FORWARD_PERIODS: int = 5
    WALK_FORWARD_IS_RATIO: float = 0.70

    @property
    def DEFAULT_WATCHLIST(self) -> List[str]:
        raw = os.getenv("DEFAULT_WATCHLIST", '["EURUSD","GBPUSD","USDJPY","XAUUSD"]')
        try:
            return json.loads(raw)
        except Exception:
            return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
