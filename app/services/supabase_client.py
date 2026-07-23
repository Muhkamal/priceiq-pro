"""
app/services/supabase_client.py
Supabase client for PriceIQ Pro — handles all database operations
"""

import os
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY", "")


class SupabaseClient:
    """
    Lightweight Supabase REST client — no extra SDK needed.
    Uses the service role key for writes, anon key for reads.
    """

    def __init__(self):
        self.url = SUPABASE_URL.rstrip("/")
        self.anon_key = SUPABASE_ANON_KEY
        self.service_key = SUPABASE_SERVICE_KEY
        self.enabled = bool(self.url and self.anon_key)

        if self.enabled:
            logger.info("Supabase client initialized ✓")
        else:
            logger.warning("Supabase not configured — running without persistence")

    def _headers(self, write: bool = False) -> Dict[str, str]:
        key = self.service_key if (write and self.service_key) else self.anon_key
        return {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def _table_url(self, table: str) -> str:
        return f"{self.url}/rest/v1/{table}"

    # ----------------------------------------------------------
    # SIGNALS
    # ----------------------------------------------------------

    async def save_signal(self, signal_data: Dict[str, Any]) -> Optional[Dict]:
        """Save a new trade signal to the database."""
        if not self.enabled:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self._table_url("signals"),
                    headers=self._headers(write=True),
                    json=signal_data,
                )
                resp.raise_for_status()
                result = resp.json()
                logger.info(f"Signal saved: {signal_data.get('pair')} {signal_data.get('direction')}")
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return None

    async def get_active_signals(self, pair: Optional[str] = None) -> List[Dict]:
        """Fetch all active signals, optionally filtered by pair."""
        if not self.enabled:
            return []
        try:
            params = "status=eq.ACTIVE&order=created_at.desc&limit=50"
            if pair:
                params += f"&pair=eq.{pair}"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._table_url('signals')}?{params}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []

    async def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Fetch most recent signals regardless of status."""
        if not self.enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._table_url('signals')}?order=created_at.desc&limit={limit}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch recent signals: {e}")
            return []

    async def update_signal_status(
        self, signal_id: str, status: str, pnl: Optional[float] = None
    ) -> bool:
        """Update a signal's status (TP_HIT, SL_HIT, EXPIRED, etc.)."""
        if not self.enabled:
            return False
        try:
            payload: Dict[str, Any] = {
                "status": status,
                "resolved_at": datetime.utcnow().isoformat(),
            }
            if pnl is not None:
                payload["pnl"] = pnl

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.patch(
                    f"{self._table_url('signals')}?id=eq.{signal_id}",
                    headers=self._headers(write=True),
                    json=payload,
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to update signal {signal_id}: {e}")
            return False

    # ----------------------------------------------------------
    # TRADES
    # ----------------------------------------------------------

    async def save_trade(self, trade_data: Dict[str, Any]) -> Optional[Dict]:
        """Save an executed trade."""
        if not self.enabled:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self._table_url("trades"),
                    headers=self._headers(write=True),
                    json=trade_data,
                )
                resp.raise_for_status()
                result = resp.json()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return None

    async def close_trade(
        self, trade_id: str, exit_price: float, pnl: float, pnl_pips: float
    ) -> bool:
        """Mark a trade as closed with exit details."""
        if not self.enabled:
            return False
        try:
            payload = {
                "status": "CLOSED",
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pips": pnl_pips,
                "closed_at": datetime.utcnow().isoformat(),
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.patch(
                    f"{self._table_url('trades')}?id=eq.{trade_id}",
                    headers=self._headers(write=True),
                    json=payload,
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False

    async def get_open_trades(self) -> List[Dict]:
        """Fetch all currently open trades."""
        if not self.enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._table_url('trades')}?status=eq.OPEN&order=opened_at.desc",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch open trades: {e}")
            return []

    async def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Fetch closed trade history."""
        if not self.enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._table_url('trades')}?status=eq.CLOSED&order=closed_at.desc&limit={limit}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch trade history: {e}")
            return []

    # ----------------------------------------------------------
    # ACCOUNT SNAPSHOTS
    # ----------------------------------------------------------

    async def save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Save a daily account balance snapshot."""
        if not self.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self._table_url("account_snapshots"),
                    headers=self._headers(write=True),
                    json=snapshot_data,
                )
                resp.raise_for_status()
                logger.info(f"Snapshot saved: balance={snapshot_data.get('balance')}")
                return True
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False

    async def get_snapshots(self, limit: int = 30) -> List[Dict]:
        """Fetch recent account snapshots for performance charts."""
        if not self.enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._table_url('account_snapshots')}?order=snapshot_at.desc&limit={limit}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch snapshots: {e}")
            return []

    # ----------------------------------------------------------
    # PERFORMANCE STATS
    # ----------------------------------------------------------

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate win rate, total PnL, and trade counts from closed trades."""
        if not self.enabled:
            return {}
        try:
            trades = await self.get_trade_history(limit=500)
            if not trades:
                return {
                    "total_trades": 0, "win_trades": 0, "loss_trades": 0,
                    "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                }

            wins   = [t for t in trades if (t.get("pnl") or 0) > 0]
            losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
            total_pnl = sum(t.get("pnl") or 0 for t in trades)

            return {
                "total_trades": len(trades),
                "win_trades":   len(wins),
                "loss_trades":  len(losses),
                "win_rate":     round(len(wins) / len(trades) * 100, 1),
                "total_pnl":    round(total_pnl, 2),
                "avg_pnl":      round(total_pnl / len(trades), 2),
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}


# Singleton instance — import this everywhere
supabase = SupabaseClient()
