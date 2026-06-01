"""
PriceIQ Pro — Database Persistence Layer v1.1
Supabase (PostgreSQL) integration.

Tables managed:
  users            — accounts, tiers, usage limits
  signals          — every signal generated (live + backtest)
  trades           — open and closed trade records
  circuit_breaker  — persisted risk state (survives restarts)
  backtest_results — stored backtest runs
  watchlists       — per-user pair watchlists
  api_keys         — hashed API keys for auth

All DB calls are async. Falls back gracefully if Supabase is unavailable.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.core.config import settings

# Optional Supabase import — don't crash if not installed
try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


class Database:
    """
    Supabase wrapper with graceful fallback.
    If Supabase is not configured, all writes are no-ops and reads return empty.
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._available = False

        if _SUPABASE_AVAILABLE and settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                self._available = True
                logger.info("Supabase connected")
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e} — running without persistence")
        else:
            logger.warning("Supabase not configured — running without persistence")

    @property
    def is_available(self) -> bool:
        return self._available

    # ──────────────────────────────────────────────────────────
    # USERS
    # ──────────────────────────────────────────────────────────

    async def create_user(self, email: str, hashed_password: str, tier: str = "free") -> Optional[Dict]:
        if not self._available:
            return None
        try:
            user_id = str(uuid.uuid4())
            data = {
                "id":              user_id,
                "email":           email.lower().strip(),
                "hashed_password": hashed_password,
                "tier":            tier,
                "daily_signals_used": 0,
                "daily_signals_limit": self._tier_limit(tier),
                "created_at":      datetime.now(timezone.utc).isoformat(),
            }
            result = self._client.table("users").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"create_user error: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            result = self._client.table("users").select("*").eq("email", email.lower().strip()).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"get_user_by_email error: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            result = self._client.table("users").select("*").eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"get_user_by_id error: {e}")
            return None

    async def update_user_signal_count(self, user_id: str, count: int) -> bool:
        if not self._available:
            return True
        try:
            self._client.table("users").update({"daily_signals_used": count}).eq("id", user_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_user_signal_count error: {e}")
            return False

    async def reset_daily_signal_counts(self) -> bool:
        """Call this daily at midnight to reset usage counters."""
        if not self._available:
            return True
        try:
            self._client.table("users").update({"daily_signals_used": 0}).neq("id", "none").execute()
            return True
        except Exception as e:
            logger.error(f"reset_daily_signal_counts error: {e}")
            return False

    async def update_user_tier(self, user_id: str, tier: str) -> bool:
        if not self._available:
            return True
        try:
            self._client.table("users").update({
                "tier": tier,
                "daily_signals_limit": self._tier_limit(tier),
            }).eq("id", user_id).execute()
            return True
        except Exception as e:
            logger.error(f"update_user_tier error: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # API KEYS
    # ──────────────────────────────────────────────────────────

    async def create_api_key(self, user_id: str, name: str = "default") -> Optional[Dict]:
        """Generate and store a new API key for a user. Returns the raw key once."""
        if not self._available:
            return {"key": f"priceiq_{uuid.uuid4().hex}", "note": "DB unavailable — key not persisted"}
        try:
            raw_key = f"priceiq_{uuid.uuid4().hex}"
            hashed  = hashlib.sha256(raw_key.encode()).hexdigest()
            data = {
                "id":       str(uuid.uuid4()),
                "user_id":  user_id,
                "name":     name,
                "key_hash": hashed,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used":  None,
                "is_active":  True,
            }
            self._client.table("api_keys").insert(data).execute()
            return {"key": raw_key, "key_id": data["id"]}
        except Exception as e:
            logger.error(f"create_api_key error: {e}")
            return None

    async def validate_api_key(self, raw_key: str) -> Optional[Dict]:
        """Validate an API key and return the associated user. Returns None if invalid."""
        if not self._available:
            return None
        try:
            hashed = hashlib.sha256(raw_key.encode()).hexdigest()
            result = self._client.table("api_keys").select("*, users(*)").eq("key_hash", hashed).eq("is_active", True).execute()
            if not result.data:
                return None
            key_record = result.data[0]
            # Update last_used
            self._client.table("api_keys").update({"last_used": datetime.now(timezone.utc).isoformat()}).eq("key_hash", hashed).execute()
            return key_record.get("users")
        except Exception as e:
            logger.error(f"validate_api_key error: {e}")
            return None

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        if not self._available:
            return True
        try:
            self._client.table("api_keys").update({"is_active": False}).eq("id", key_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            logger.error(f"revoke_api_key error: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # SIGNALS
    # ──────────────────────────────────────────────────────────

    async def save_signal(self, signal: Dict, user_id: Optional[str] = None) -> Optional[str]:
        if not self._available:
            return None
        try:
            record_id = str(uuid.uuid4())
            data = {
                "id":         record_id,
                "user_id":    user_id,
                "pair":       signal.get("pair"),
                "timeframe":  signal.get("timeframe"),
                "direction":  signal.get("direction"),
                "pattern":    signal.get("pattern"),
                "confidence": signal.get("confidence"),
                "entry_price":   signal.get("entry_price"),
                "stop_loss":     signal.get("stop_loss"),
                "take_profit_1": signal.get("take_profit_1"),
                "take_profit_2": signal.get("take_profit_2"),
                "risk_reward":   signal.get("risk_reward_1"),
                "signal_data":   json.dumps(signal),
                "created_at":    datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("signals").insert(data).execute()
            return record_id
        except Exception as e:
            logger.error(f"save_signal error: {e}")
            return None

    async def get_signals(
        self,
        user_id: Optional[str] = None,
        pair: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        if not self._available:
            return []
        try:
            query = self._client.table("signals").select("*").order("created_at", desc=True).limit(limit)
            if user_id:
                query = query.eq("user_id", user_id)
            if pair:
                query = query.eq("pair", pair.upper())
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"get_signals error: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # TRADES
    # ──────────────────────────────────────────────────────────

    async def save_trade(self, trade: Dict, user_id: Optional[str] = None, signal_id: Optional[str] = None) -> Optional[str]:
        if not self._available:
            return None
        try:
            trade_id = str(uuid.uuid4())
            data = {
                "id":          trade_id,
                "user_id":     user_id,
                "signal_id":   signal_id,
                "pair":        trade.get("pair"),
                "direction":   trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "stop_loss":   trade.get("stop_loss"),
                "take_profit": trade.get("take_profit_1"),
                "position_size": trade.get("position_size"),
                "status":      "open",
                "broker_order_id": trade.get("broker_order_id"),
                "opened_at":   datetime.now(timezone.utc).isoformat(),
                "closed_at":   None,
                "exit_price":  None,
                "pnl":         None,
                "result":      None,
            }
            self._client.table("trades").insert(data).execute()
            return trade_id
        except Exception as e:
            logger.error(f"save_trade error: {e}")
            return None

    async def close_trade(self, trade_id: str, exit_price: float, pnl: float, result: str) -> bool:
        if not self._available:
            return True
        try:
            self._client.table("trades").update({
                "status":     "closed",
                "exit_price": exit_price,
                "pnl":        pnl,
                "result":     result,
                "closed_at":  datetime.now(timezone.utc).isoformat(),
            }).eq("id", trade_id).execute()
            return True
        except Exception as e:
            logger.error(f"close_trade error: {e}")
            return False

    async def get_open_trades(self, user_id: Optional[str] = None) -> List[Dict]:
        if not self._available:
            return []
        try:
            query = self._client.table("trades").select("*").eq("status", "open")
            if user_id:
                query = query.eq("user_id", user_id)
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"get_open_trades error: {e}")
            return []

    async def get_trade_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        if not self._available:
            return []
        try:
            result = self._client.table("trades").select("*").eq("user_id", user_id).order("opened_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"get_trade_history error: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # CIRCUIT BREAKER STATE (persisted across restarts)
    # ──────────────────────────────────────────────────────────

    async def save_circuit_breaker_state(self, user_id: str, state: Dict) -> bool:
        if not self._available:
            return True
        try:
            data = {
                "user_id":    user_id,
                "state":      json.dumps(state),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            # Upsert — update if exists, insert if not
            self._client.table("circuit_breaker_state").upsert(data, on_conflict="user_id").execute()
            return True
        except Exception as e:
            logger.error(f"save_circuit_breaker_state error: {e}")
            return False

    async def load_circuit_breaker_state(self, user_id: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            result = self._client.table("circuit_breaker_state").select("*").eq("user_id", user_id).execute()
            if result.data:
                return json.loads(result.data[0]["state"])
            return None
        except Exception as e:
            logger.error(f"load_circuit_breaker_state error: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # BACKTEST RESULTS
    # ──────────────────────────────────────────────────────────

    async def save_backtest(self, result: Dict, user_id: Optional[str] = None) -> Optional[str]:
        if not self._available:
            return None
        try:
            backtest_id = str(uuid.uuid4())
            metrics = result.get("metrics", {})
            data = {
                "id":             backtest_id,
                "user_id":        user_id,
                "pair":           result.get("parameters", {}).get("pair"),
                "timeframe":      result.get("parameters", {}).get("timeframe"),
                "verdict":        result.get("verdict"),
                "verdict_color":  result.get("verdict_color"),
                "total_trades":   metrics.get("total_trades"),
                "win_rate":       metrics.get("win_rate"),
                "profit_factor":  metrics.get("profit_factor"),
                "sharpe_ratio":   metrics.get("sharpe_ratio"),
                "max_drawdown":   metrics.get("max_drawdown_percent"),
                "total_return_pct": metrics.get("total_return_percent"),
                "result_data":    json.dumps(result),
                "created_at":     datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("backtest_results").insert(data).execute()
            return backtest_id
        except Exception as e:
            logger.error(f"save_backtest error: {e}")
            return None

    async def get_backtests(self, user_id: str, limit: int = 20) -> List[Dict]:
        if not self._available:
            return []
        try:
            result = self._client.table("backtest_results").select("id, pair, timeframe, verdict, verdict_color, total_trades, win_rate, profit_factor, total_return_pct, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"get_backtests error: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # WATCHLISTS
    # ──────────────────────────────────────────────────────────

    async def get_watchlist(self, user_id: str) -> List[str]:
        if not self._available:
            return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        try:
            result = self._client.table("watchlists").select("pair").eq("user_id", user_id).execute()
            return [r["pair"] for r in result.data] if result.data else []
        except Exception as e:
            logger.error(f"get_watchlist error: {e}")
            return []

    async def add_to_watchlist(self, user_id: str, pair: str, timeframe: str = "1h") -> bool:
        if not self._available:
            return True
        try:
            self._client.table("watchlists").insert({
                "id": str(uuid.uuid4()), "user_id": user_id,
                "pair": pair.upper(), "timeframe": timeframe,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            return True
        except Exception as e:
            logger.error(f"add_to_watchlist error: {e}")
            return False

    async def remove_from_watchlist(self, user_id: str, pair: str) -> bool:
        if not self._available:
            return True
        try:
            self._client.table("watchlists").delete().eq("user_id", user_id).eq("pair", pair.upper()).execute()
            return True
        except Exception as e:
            logger.error(f"remove_from_watchlist error: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    def _tier_limit(self, tier: str) -> int:
        return {
            "free": settings.FREE_DAILY_SIGNALS,
            "starter": settings.STARTER_DAILY_SIGNALS,
            "pro": settings.PRO_DAILY_SIGNALS,
            "elite": settings.ELITE_DAILY_SIGNALS,
        }.get(tier, settings.FREE_DAILY_SIGNALS)


# Singleton
db = Database()
