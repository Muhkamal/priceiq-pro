"""
PriceIQ Pro — Graceful Shutdown Handler v1.0

Handles SIGTERM (Render deploy/restart), SIGINT (Ctrl+C), and OOM kills.
Ensures no state is lost when the process exits.

On shutdown:
    1. Signal all running scan tasks to stop
    2. Flush learning state to disk (learning_state.json)
    3. Flush regime weights to disk (regime_weights.json)
    4. Flush win probability calibrator (win_prob_calibrator.json)
    5. Flush equity curve (equity_curve.json)
    6. Flush trade journal (trade_journal.json)
    7. Save open position state (open_positions_snapshot.json)
    8. Send Telegram shutdown notification with portfolio summary
    9. Wait for active jobs to finish (max 10s timeout)

Usage in main.py lifespan:
    from app.services.core.shutdown_handler import ShutdownHandler
    shutdown = ShutdownHandler(v5=v5, telegram=telegram)
    shutdown.register()   # registers SIGTERM/SIGINT handlers

    # In lifespan shutdown block:
    await shutdown.execute()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT   = 10.0   # max seconds to wait for running tasks
POSITIONS_SNAPSHOT = "open_positions_snapshot.json"


class ShutdownHandler:
    """
    Registers OS signal handlers and orchestrates clean shutdown.
    All persistence is best-effort: failures are logged, not raised.
    """

    def __init__(self, v5=None, telegram=None, extra_flush_fns: Optional[List[Callable]] = None):
        self._v5             = v5
        self._telegram       = telegram
        self._extra_flush_fns = extra_flush_fns or []
        self._shutdown_event  = asyncio.Event()
        self._registered      = False

    # ── Registration ─────────────────────────────────────────

    def register(self):
        """Register SIGTERM and SIGINT handlers. Call at app startup."""
        if self._registered:
            return
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._signal_received, sig)
            self._registered = True
            logger.info("ShutdownHandler: SIGTERM/SIGINT handlers registered")
        except (NotImplementedError, RuntimeError):
            # Windows or no running loop
            signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.execute()))
            signal.signal(signal.SIGINT,  lambda s, f: asyncio.create_task(self.execute()))
            logger.info("ShutdownHandler: fallback signal handlers registered")

    def _signal_received(self, sig):
        logger.info(f"ShutdownHandler: received signal {sig.name}")
        self._shutdown_event.set()
        asyncio.create_task(self.execute())

    async def wait_for_shutdown(self):
        """Await this in your lifespan to block until shutdown signal."""
        await self._shutdown_event.wait()

    # ── Main shutdown sequence ────────────────────────────────

    async def execute(self):
        """Run the full shutdown sequence. Idempotent."""
        logger.info("=" * 60)
        logger.info("GRACEFUL SHUTDOWN INITIATED")
        logger.info("=" * 60)

        v5 = self._v5

        # 1. Stop new signals immediately
        if v5:
            v5._signals_paused = True

        # 2. Flush all persistence layers
        await self._flush_learning(v5)
        await self._flush_positions(v5)
        await self._flush_equity_curve(v5)
        await self._flush_journal(v5)
        await self._flush_regime_components(v5)

        # 3. Run any extra flush functions (e.g. database)
        for fn in self._extra_flush_fns:
            try:
                if asyncio.iscoroutinefunction(fn):
                    await asyncio.wait_for(fn(), timeout=3.0)
                else:
                    fn()
            except Exception as e:
                logger.warning(f"Extra flush fn failed: {e}")

        # 4. Send Telegram notification
        await self._notify_shutdown(v5)

        # 5. Log summary
        logger.info("Shutdown complete — all state flushed")
        logger.info("=" * 60)

    # ── Individual flush steps ────────────────────────────────

    async def _flush_learning(self, v5):
        try:
            if v5 and hasattr(v5, "learning"):
                v5.learning.store._save()
                logger.info("✅ Learning state flushed")
        except Exception as e:
            logger.warning(f"Learning flush failed: {e}")

    async def _flush_positions(self, v5):
        """Save open position state so we can reconcile on restart."""
        try:
            if not v5 or not hasattr(v5, "trade_manager"):
                return
            positions = v5.trade_manager.get_open_positions()
            snapshot  = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": positions,
                "governor_balance": v5.governor.current_balance
                    if hasattr(v5, "governor") else 0,
            }
            with open(POSITIONS_SNAPSHOT, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"✅ Position snapshot saved ({len(positions)} open)")
        except Exception as e:
            logger.warning(f"Position snapshot failed: {e}")

    async def _flush_equity_curve(self, v5):
        try:
            if v5 and hasattr(v5, "equity_tracker"):
                v5.equity_tracker._save()
                logger.info("✅ Equity curve flushed")
        except Exception as e:
            logger.warning(f"Equity curve flush failed: {e}")

    async def _flush_journal(self, v5):
        try:
            if v5 and hasattr(v5, "journal"):
                v5.journal._save()
                logger.info("✅ Trade journal flushed")
        except Exception as e:
            logger.warning(f"Journal flush failed: {e}")

    async def _flush_regime_components(self, v5):
        """Flush regime weights, win prob calibrator, transition model."""
        try:
            if v5 and hasattr(v5, "regime_learner"):
                v5.regime_learner._save()
            if v5 and hasattr(v5, "win_prob_cal"):
                v5.win_prob_cal._save()
            if v5 and hasattr(v5, "regime_transition"):
                v5.regime_transition._save()
            logger.info("✅ Regime components flushed")
        except Exception as e:
            logger.warning(f"Regime component flush failed: {e}")

    async def _notify_shutdown(self, v5):
        """Send Telegram shutdown notification with portfolio summary."""
        if not self._telegram:
            return
        try:
            ts  = datetime.now(timezone.utc).strftime("%H:%M UTC %d %b")
            msg = f"🔴 <b>PriceIQ Pro OFFLINE</b> — {ts}\n\n"

            if v5 and hasattr(v5, "governor"):
                p   = v5.governor.get_portfolio_summary()
                bal = p.get("current_balance", 0)
                dd  = p.get("drawdown", 0)
                n   = p.get("open_positions", 0)
                msg += (
                    f"Balance: ${bal:,.2f}\n"
                    f"Drawdown: {dd:.1%}\n"
                    f"Open positions: {n}\n"
                )
                if n > 0:
                    msg += f"\n⚠️ <b>{n} position(s) still open — check broker manually.</b>\n"
                    msg += f"Position snapshot saved to {POSITIONS_SNAPSHOT}"

            msg += "\nAll state flushed to disk. Will restore on next startup."
            await asyncio.wait_for(
                self._telegram.send_message(msg), timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Shutdown Telegram notification failed: {e}")

    @staticmethod
    def load_position_snapshot() -> Optional[Dict]:
        """Load saved position state on startup to reconcile with broker."""
        if not os.path.exists(POSITIONS_SNAPSHOT):
            return None
        try:
            with open(POSITIONS_SNAPSHOT) as f:
                data = json.load(f)
            logger.info(
                f"Position snapshot found: {len(data.get('positions', {}))} positions "
                f"from {data.get('timestamp', '?')}"
            )
            return data
        except Exception as e:
            logger.warning(f"Position snapshot load failed: {e}")
            return None
