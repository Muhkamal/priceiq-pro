"""
PriceIQ Pro — V5 Settings Management v1.0

Single source of truth for all V5 configuration.
Every threshold, multiplier, and path is defined here.
Nothing is hardcoded across 21 files anymore.

All values read from environment variables with sensible defaults.
Validated at startup — contradictory settings fail loudly.

Usage:
    from app.services.core.v5_settings import v5_settings

    # Access any setting:
    v5_settings.ATR_STOP_MULTIPLIER       # 1.5
    v5_settings.MIN_SIGNAL_CONFIDENCE     # 0.55
    v5_settings.MAX_DRAWDOWN_PCT          # 0.10

    # Check if fully configured:
    issues = v5_settings.validate()
    if issues:
        raise SystemExit(f"Config errors: {issues}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _env(key: str, default, cast=str):
    """Read env var with type casting and default."""
    val = os.environ.get(key, "")
    if not val:
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        logger.warning(f"V5Settings: invalid value for {key}='{val}' — using default {default}")
        return default


@dataclass
class V5Settings:
    """
    Complete V5 system configuration.
    All fields have production-safe defaults.
    Override by setting environment variables on Render.
    """

    # ── Account ───────────────────────────────────────────────
    ACCOUNT_BALANCE:     float = field(default_factory=lambda: _env("ACCOUNT_BALANCE",    100.0, float))
    RISK_PERCENT:        float = field(default_factory=lambda: _env("RISK_PERCENT",        2.0,     float))
    TARGET_VOL_PCT:      float = field(default_factory=lambda: _env("TARGET_VOL_PCT",      0.02,    float))
    MAX_DRAWDOWN_PCT:    float = field(default_factory=lambda: _env("MAX_DRAWDOWN",        0.10,    float))
    SOFT_DRAWDOWN_PCT:   float = field(default_factory=lambda: _env("SOFT_DRAWDOWN",       0.06,    float))
    MAX_CONSECUTIVE_LOSSES: int = field(default_factory=lambda: _env("MAX_CONSECUTIVE_LOSSES", 4, int))
    MAX_TOTAL_EXPOSURE:  float = field(default_factory=lambda: _env("MAX_TOTAL_EXPOSURE",  0.08,    float))
    MAX_CORRELATED_POS:  int   = field(default_factory=lambda: _env("MAX_CORRELATED_POS",  2,       int))

    # ── ATR / Signal ──────────────────────────────────────────
    ATR_STOP_MULTIPLIER: float = field(default_factory=lambda: _env("ATR_STOP_MULTIPLIER", 1.5,  float))
    ATR_TP1_MULTIPLIER:  float = field(default_factory=lambda: _env("ATR_TP1_MULTIPLIER",  2.0,  float))
    ATR_TP2_MULTIPLIER:  float = field(default_factory=lambda: _env("ATR_TP2_MULTIPLIER",  3.0,  float))
    ATR_TP3_MULTIPLIER:  float = field(default_factory=lambda: _env("ATR_TP3_MULTIPLIER",  4.5,  float))

    # ── Signal filters ────────────────────────────────────────
    MIN_SIGNAL_CONFIDENCE:  float = field(default_factory=lambda: _env("MIN_SIGNAL_CONFIDENCE", 0.55, float))
    MIN_VOLUME_RATIO:       float = field(default_factory=lambda: _env("MIN_VOLUME_RATIO",       0.70, float))
    RSI_OVERBOUGHT:         int   = field(default_factory=lambda: _env("RSI_OVERBOUGHT",         70,   int))
    RSI_OVERSOLD:           int   = field(default_factory=lambda: _env("RSI_OVERSOLD",           30,   int))
    RSI_PERIOD:             int   = field(default_factory=lambda: _env("RSI_PERIOD",             14,   int))
    MACD_FAST:              int   = field(default_factory=lambda: _env("MACD_FAST",              12,   int))
    MACD_SLOW:              int   = field(default_factory=lambda: _env("MACD_SLOW",              26,   int))
    MACD_SIGNAL:            int   = field(default_factory=lambda: _env("MACD_SIGNAL",            9,    int))
    SIGNAL_COOLDOWN_BARS:   int   = field(default_factory=lambda: _env("SIGNAL_COOLDOWN_BARS",   3,    int))

    # ── Regime classifier ─────────────────────────────────────
    REGIME_MODEL_PATH:      str  = field(default_factory=lambda: _env("REGIME_MODEL_PATH",   "regime_model.pkl"))
    MIN_DRIFT_PSI_WARN:     float = field(default_factory=lambda: _env("MIN_DRIFT_PSI_WARN",  0.10, float))
    MIN_DRIFT_PSI_SEVERE:   float = field(default_factory=lambda: _env("MIN_DRIFT_PSI_SEVERE", 0.25, float))
    REGIME_SMOOTH_ALPHA:    float = field(default_factory=lambda: _env("REGIME_SMOOTH_ALPHA",  0.10, float))

    # ── Regime transition model ───────────────────────────────
    TRANSITION_SHIFT_RISK_REDUCE: float = field(default_factory=lambda: _env("SHIFT_RISK_REDUCE", 0.35, float))
    TRANSITION_SHIFT_RISK_BLOCK:  float = field(default_factory=lambda: _env("SHIFT_RISK_BLOCK",  0.50, float))

    # ── Learning ──────────────────────────────────────────────
    LEARNING_STATE_PATH:    str   = field(default_factory=lambda: _env("LEARNING_STATE_PATH",   "learning_state.json"))
    REGIME_WEIGHTS_PATH:    str   = field(default_factory=lambda: _env("REGIME_WEIGHTS_PATH",   "regime_weights.json"))
    WIN_PROB_PATH:          str   = field(default_factory=lambda: _env("WIN_PROB_PATH",         "win_prob_calibrator.json"))
    TRANSITION_PATH:        str   = field(default_factory=lambda: _env("TRANSITION_PATH",       "regime_transitions.json"))
    CONFIDENCE_FLOOR:       float = field(default_factory=lambda: _env("CONFIDENCE_FLOOR",       0.45, float))
    CONFIDENCE_CEILING:     float = field(default_factory=lambda: _env("CONFIDENCE_CEILING",     0.80, float))
    MIN_TRADES_TO_ADAPT:    int   = field(default_factory=lambda: _env("MIN_TRADES_TO_ADAPT",    10,   int))

    # ── Trade management ──────────────────────────────────────
    TP1_CLOSE_PCT:          float = field(default_factory=lambda: _env("TP1_CLOSE_PCT",    0.40, float))
    TP2_CLOSE_PCT:          float = field(default_factory=lambda: _env("TP2_CLOSE_PCT",    0.40, float))
    TRAIL_ACTIVATION_R:     float = field(default_factory=lambda: _env("TRAIL_ACTIVATION_R", 1.0, float))
    TRAIL_STEP_ATR:         float = field(default_factory=lambda: _env("TRAIL_STEP_ATR",   0.5,  float))
    BREAKEVEN_BUFFER_R:     float = field(default_factory=lambda: _env("BREAKEVEN_BUFFER_R", 0.05, float))
    # Regime-dependent timeouts (hours)
    TIMEOUT_TRENDING_H:     int   = field(default_factory=lambda: _env("TIMEOUT_TRENDING_H",  72, int))
    TIMEOUT_RANGING_H:      int   = field(default_factory=lambda: _env("TIMEOUT_RANGING_H",   24, int))
    TIMEOUT_VOLATILE_H:     int   = field(default_factory=lambda: _env("TIMEOUT_VOLATILE_H",  12, int))

    # ── VaR / Risk ────────────────────────────────────────────
    VAR_MAX_HEAT_PCT:       float = field(default_factory=lambda: _env("VAR_MAX_HEAT_PCT",      0.06, float))
    VAR_CRITICAL_HEAT_PCT:  float = field(default_factory=lambda: _env("VAR_CRITICAL_HEAT_PCT", 0.10, float))
    MAX_OPEN_POSITIONS:     int   = field(default_factory=lambda: _env("MAX_OPEN_POSITIONS",    6,    int))
    VAR_CONFIDENCE:         float = field(default_factory=lambda: _env("VAR_CONFIDENCE",        0.95, float))

    # ── MTF confluence ────────────────────────────────────────
    MTF_FULL_SIZE_THRESHOLD: float = field(default_factory=lambda: _env("MTF_FULL_SIZE_THRESHOLD", 0.70, float))
    MTF_HALF_SIZE_THRESHOLD: float = field(default_factory=lambda: _env("MTF_HALF_SIZE_THRESHOLD", 0.50, float))
    MTF_BLOCK_THRESHOLD:     float = field(default_factory=lambda: _env("MTF_BLOCK_THRESHOLD",     0.25, float))
    MTF_D1_OVERRIDE:         bool  = field(default_factory=lambda: _env("MTF_D1_OVERRIDE", "true").lower() == "true")

    # ── News blackout ─────────────────────────────────────────
    NEWS_BLACKOUT_PRE_HIGH:   int = field(default_factory=lambda: _env("NEWS_BLACKOUT_PRE_HIGH",   20, int))
    NEWS_BLACKOUT_POST_HIGH:  int = field(default_factory=lambda: _env("NEWS_BLACKOUT_POST_HIGH",  15, int))
    NEWS_BLACKOUT_PRE_MED:    int = field(default_factory=lambda: _env("NEWS_BLACKOUT_PRE_MED",    10, int))
    NEWS_BLACKOUT_POST_MED:   int = field(default_factory=lambda: _env("NEWS_BLACKOUT_POST_MED",   10, int))

    # ── Correlation ───────────────────────────────────────────
    CORR_THRESHOLD:         float = field(default_factory=lambda: _env("CORR_THRESHOLD",   0.75, float))
    CORR_WINDOW_NORMAL:     int   = field(default_factory=lambda: _env("CORR_WINDOW",      60,   int))
    CORR_WINDOW_VOLATILE:   int   = field(default_factory=lambda: _env("CORR_WINDOW_VOL",  20,   int))
    CORR_MIN_OBSERVATIONS:  int   = field(default_factory=lambda: _env("CORR_MIN_OBS",     20,   int))

    # ── Kelly / sizing ────────────────────────────────────────
    KELLY_FRACTION:         float = field(default_factory=lambda: _env("KELLY_FRACTION",   0.5,  float))
    MAX_LOTS:               float = field(default_factory=lambda: _env("MAX_LOTS",         10.0, float))
    MIN_LOTS:               float = field(default_factory=lambda: _env("MIN_LOTS",         0.01, float))

    # ── Execution ─────────────────────────────────────────────
    APPLY_SPREAD_TO_ENTRY:  bool  = field(default_factory=lambda: _env("APPLY_SPREAD", "true").lower() == "true")

    # ── Paths ─────────────────────────────────────────────────
    JOURNAL_PATH:           str = field(default_factory=lambda: _env("JOURNAL_PATH",       "trade_journal.json"))
    EQUITY_CURVE_PATH:      str = field(default_factory=lambda: _env("EQUITY_CURVE_PATH",  "equity_curve.json"))
    POSITIONS_SNAPSHOT:     str = field(default_factory=lambda: _env("POSITIONS_SNAPSHOT", "open_positions_snapshot.json"))

    # ── Watchlist ─────────────────────────────────────────────
    WATCHLIST:              List[str] = field(default_factory=lambda: [
        p.strip() for p in _env("WATCHLIST", "XAUUSD,EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD").split(",")
        if p.strip()
    ])

    # ── Telegram ──────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN:     str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    TELEGRAM_CHAT_ID:       str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID",   ""))

    # ── Broker ────────────────────────────────────────────────
    OANDA_API_KEY:          str   = field(default_factory=lambda: _env("OANDA_API_KEY",         ""))
    OANDA_ACCOUNT_ID:       str   = field(default_factory=lambda: _env("OANDA_ACCOUNT_ID",      ""))
    OANDA_PAPER_TRADING:    bool  = field(default_factory=lambda: _env("OANDA_PAPER", "true").lower() == "true")
    BROKER_WEBHOOK_SECRET:  str   = field(default_factory=lambda: _env("BROKER_WEBHOOK_SECRET", ""))

    # ── Feature flags ─────────────────────────────────────────
    H4_GATE_ENABLED:        bool  = field(default_factory=lambda: _env("H4_GATE_ENABLED",  "true").lower() == "true")
    MTF_ENABLED:            bool  = field(default_factory=lambda: _env("MTF_ENABLED",      "true").lower() == "true")
    NEWS_FILTER_ENABLED:    bool  = field(default_factory=lambda: _env("NEWS_FILTER",      "true").lower() == "true")
    DRIFT_MONITOR_ENABLED:  bool  = field(default_factory=lambda: _env("DRIFT_MONITOR",    "true").lower() == "true")
    VAR_GATE_ENABLED:       bool  = field(default_factory=lambda: _env("VAR_GATE",         "true").lower() == "true")
    KELLY_ENABLED:          bool  = field(default_factory=lambda: _env("KELLY_ENABLED",    "true").lower() == "true")
    THOMPSON_SAMPLING:      bool  = field(default_factory=lambda: _env("THOMPSON_SAMPLING","true").lower() == "true")
    AUGMENTATION_ENABLED:   bool  = field(default_factory=lambda: _env("AUGMENTATION",    "true").lower() == "true")

    def validate(self) -> List[str]:
        """
        Validate settings for internal consistency.
        Returns list of error strings. Empty = all OK.
        """
        errors = []

        # Risk coherence: 5 × risk_pct should not exceed max_drawdown
        # Allow exactly equal (e.g. 2% × 5 = 10% DD is acceptable)
        if (self.RISK_PERCENT / 100) * 5 > self.MAX_DRAWDOWN_PCT * 1.05:
            errors.append(
                f"Risk incoherence: 5×RISK_PERCENT ({self.RISK_PERCENT*5:.1f}%) "
                f">= MAX_DRAWDOWN ({self.MAX_DRAWDOWN_PCT*100:.0f}%)"
            )

        # ATR multipliers: TP1 must be > STOP
        if self.ATR_TP1_MULTIPLIER <= self.ATR_STOP_MULTIPLIER:
            errors.append(
                f"ATR_TP1_MULTIPLIER ({self.ATR_TP1_MULTIPLIER}) "
                f"must be > ATR_STOP_MULTIPLIER ({self.ATR_STOP_MULTIPLIER})"
            )

        # TP ordering
        if not (self.ATR_TP1_MULTIPLIER < self.ATR_TP2_MULTIPLIER < self.ATR_TP3_MULTIPLIER):
            errors.append("ATR TP multipliers must be in ascending order: TP1 < TP2 < TP3")

        # Confidence bounds
        if not (0.0 < self.CONFIDENCE_FLOOR < self.MIN_SIGNAL_CONFIDENCE < self.CONFIDENCE_CEILING < 1.0):
            errors.append(
                f"Confidence bounds must satisfy: "
                f"FLOOR({self.CONFIDENCE_FLOOR}) < "
                f"MIN_CONFIDENCE({self.MIN_SIGNAL_CONFIDENCE}) < "
                f"CEILING({self.CONFIDENCE_CEILING})"
            )

        # MTF thresholds
        if not (self.MTF_BLOCK_THRESHOLD < self.MTF_HALF_SIZE_THRESHOLD < self.MTF_FULL_SIZE_THRESHOLD):
            errors.append("MTF thresholds must be: BLOCK < HALF_SIZE < FULL_SIZE")

        # Kelly fraction
        if not (0.0 < self.KELLY_FRACTION <= 1.0):
            errors.append(f"KELLY_FRACTION must be (0, 1], got {self.KELLY_FRACTION}")

        # RSI bounds
        if not (self.RSI_OVERSOLD < 50 < self.RSI_OVERBOUGHT):
            errors.append(f"RSI_OVERSOLD ({self.RSI_OVERSOLD}) must be < 50 < RSI_OVERBOUGHT ({self.RSI_OVERBOUGHT})")

        # Timeout ordering
        if not (self.TIMEOUT_VOLATILE_H <= self.TIMEOUT_RANGING_H <= self.TIMEOUT_TRENDING_H):
            errors.append("Timeouts must satisfy: VOLATILE_H <= RANGING_H <= TRENDING_H")

        return errors

    def regime_timeout(self, regime: str) -> int:
        """Regime-aware position timeout in hours."""
        return {
            "trending": self.TIMEOUT_TRENDING_H,
            "ranging":  self.TIMEOUT_RANGING_H,
            "volatile": self.TIMEOUT_VOLATILE_H,
        }.get(regime, self.TIMEOUT_RANGING_H)

    def to_dict(self) -> Dict:
        """Export all settings as a dict (for dashboard and logging)."""
        import dataclasses
        return {
            k: v for k, v in dataclasses.asdict(self).items()
            if k not in ("TELEGRAM_BOT_TOKEN", "OANDA_API_KEY",
                         "OANDA_ACCOUNT_ID", "BROKER_WEBHOOK_SECRET")
        }

    def log_summary(self):
        """Log key settings at startup."""
        logger.info(
            f"V5 Settings loaded:\n"
            f"  Balance: ${self.ACCOUNT_BALANCE:,.0f} | Risk: {self.RISK_PERCENT}% | "
            f"MaxDD: {self.MAX_DRAWDOWN_PCT:.0%}\n"
            f"  Watchlist: {self.WATCHLIST}\n"
            f"  Gates: MTF={self.MTF_ENABLED} News={self.NEWS_FILTER_ENABLED} "
            f"Drift={self.DRIFT_MONITOR_ENABLED} VaR={self.VAR_GATE_ENABLED}\n"
            f"  Sizing: Kelly={self.KELLY_ENABLED} Thompson={self.THOMPSON_SAMPLING} "
            f"Augment={self.AUGMENTATION_ENABLED}\n"
            f"  Paper trading: {self.OANDA_PAPER_TRADING}"
        )


# ── Singleton ─────────────────────────────────────────────────
v5_settings = V5Settings()
