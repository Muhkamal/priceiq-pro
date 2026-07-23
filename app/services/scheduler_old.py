"""
PriceIQ Pro — Task Scheduler v1.2

Runs recurring jobs automatically:
  - Hourly scan (candle close): run_cycle() across all watchlist pairs
  - Daily midnight: reset signal usage counters
  - Every 5 min: sync account balance from broker
  - Daily 8AM UTC: send daily digest to subscribed users
  - 4H candle close: higher timeframe confirmation scan

Uses APScheduler with AsyncIOScheduler backend.
All jobs are async-safe and crash-isolated (one job failure doesn't stop others).
Telegram alerts fire on: startup, every hourly/4H scan result, and all errors.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List
from app.services.telegram_bot import telegram

logger = logging.getLogger(__name__)

# Optional APScheduler import
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed — scheduler disabled. pip install apscheduler")

from app.services.trade_engine import LiveTradeEngine, EngineConfig
from app.services.database import db
from app.services.alerts import AlertManager
from app.core.config import settings


class TradingScheduler:
    """
    Wraps APScheduler to manage all recurring trading jobs.
    Instantiate once at app startup and call start().
    Telegram alerts are sent for every scan result, error, and daily digest.
    """

    WATCHLIST = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]

    def __init__(
        self,
        engine: LiveTradeEngine,
        alert_recipients: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.alert_recipients = alert_recipients or []
        self.alert_manager = AlertManager()
        self._scheduler = None
        self._running = False
        self._scan_count = 0
        self._last_scan: Optional[datetime] = None
        self._last_signals: List = []
        self._lock = asyncio.Lock()

    def start(self):
        """Start all scheduled jobs. Call from FastAPI lifespan startup."""
        if not _SCHEDULER_AVAILABLE:
            logger.warning("Scheduler not started — APScheduler not installed")
            return

        self._scheduler = AsyncIOScheduler(timezone="UTC")

        # Job 1: Hourly candle-close scan
        self._scheduler.add_job(
            self._run_hourly_scan,
            CronTrigger(minute=1),
            id="hourly_scan",
            name="Hourly Signal Scan",
            max_instances=1,
            misfire_grace_time=300,
            coalesce=True,
        )

        # Job 2: Daily midnight — reset signal counts
        self._scheduler.add_job(
            self._reset_daily_counts,
            CronTrigger(hour=0, minute=0),
            id="daily_reset",
            name="Reset Daily Signal Counts",
        )

        # Job 3: Every 5 min — sync broker balance
        self._scheduler.add_job(
            self._sync_broker_balance,
            IntervalTrigger(minutes=5),
            id="balance_sync",
            name="Broker Balance Sync",
        )

        # Job 4: Daily 8AM UTC — send digest
        self._scheduler.add_job(
            self._send_daily_digest,
            CronTrigger(hour=8, minute=0),
            id="daily_digest",
            name="Daily Signal Digest",
        )

        # Job 5: 4H candle close — 4H timeframe scan
        self._scheduler.add_job(
            self._run_4h_scan,
            CronTrigger(hour="0,4,8,12,16,20", minute=1),
            id="4h_scan",
            name="4H Signal Scan",
            max_instances=1,
            coalesce=True,
        )

        self._scheduler.start()
        self._running = True
        logger.info("Scheduler started — hourly scan, 4H scan, daily reset, balance sync, digest")

    def stop(self):
        """Stop all jobs. Call from FastAPI lifespan shutdown."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    def get_status(self) -> dict:
        """Return scheduler status for dashboard."""
        jobs = []
        if self._scheduler:
            for job in self._scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                })
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "jobs": jobs,
            "apscheduler": _SCHEDULER_AVAILABLE,
        }

    # ──────────────────────────────────────────────────────────
    # JOB IMPLEMENTATIONS
    # ──────────────────────────────────────────────────────────

    async def _run_hourly_scan(self):
        """
        Hourly signal scan across all watchlist pairs.
        Sends Telegram alert for each signal found, or a no-signal update.
        """
        async with self._lock:
            try:
                logger.info(f"Running hourly scan #{self._scan_count + 1}")
                decisions = await self.engine.run_cycle(alert_recipients=self.alert_recipients)
                self._scan_count += 1
                self._last_scan = datetime.now(timezone.utc)

                signals_found = [d for d in decisions if d.passed]
                self._last_signals = signals_found

                if signals_found:
                    logger.info(f"Hourly scan complete — {len(signals_found)} signal(s) found")
                    for d in signals_found:
                        if d.signal:
                            await db.save_signal(d.signal.dict())
                            # ── Telegram: fire alert for each signal ──
                            try:
                                await telegram.send_signal_alert(d.signal)
                            except Exception as tg_err:
                                logger.warning(f"Telegram signal alert failed: {tg_err}")
                else:
                    logger.info(f"Hourly scan complete — no signals (checked {len(decisions)} pairs)")
                    # ── Telegram: send no-signal heartbeat ──
                    try:
                        await telegram.send_no_signal_update(
                            pairs_checked=len(decisions) or len(self.WATCHLIST)
                        )
                    except Exception as tg_err:
                        logger.warning(f"Telegram no-signal update failed: {tg_err}")

            except Exception as e:
                logger.error(f"Hourly scan failed: {e}")
                # ── Telegram: report scan error ──
                try:
                    await telegram.send_message(
                        f"⚠️ <b>Hourly Scan Error</b>\n"
                        f"Scan #{self._scan_count + 1} crashed:\n"
                        f"<code>{e}</code>"
                    )
                except Exception:
                    pass

    async def _run_4h_scan(self):
        """
        4H timeframe scan — higher timeframe confirmation.
        Sends Telegram alert for each 4H signal found.
        """
        async with self._lock:
            try:
                logger.info("Running 4H scan")

                config_4h = EngineConfig(
                    timeframe="4h",
                    watchlist=self.engine.config.watchlist,
                    account_balance=self.engine.config.account_balance,
                    risk_percent=self.engine.config.risk_percent,
                    use_multi_timeframe=False,
                )
                engine_4h = LiveTradeEngine(config=config_4h)
                decisions = await engine_4h.run_cycle(alert_recipients=self.alert_recipients)

                signals = [d for d in decisions if d.passed]
                if signals:
                    logger.info(f"4H scan — {len(signals)} signal(s)")
                    for d in signals:
                        if d.signal:
                            await db.save_signal(d.signal.dict())
                            # ── Telegram: fire alert for each 4H signal ──
                            try:
                                await telegram.send_signal_alert(d.signal)
                            except Exception as tg_err:
                                logger.warning(f"Telegram 4H signal alert failed: {tg_err}")
                else:
                    logger.info("4H scan complete — no signals")

            except Exception as e:
                logger.error(f"4H scan failed: {e}")
                # ── Telegram: report 4H scan error ──
                try:
                    await telegram.send_message(
                        f"⚠️ <b>4H Scan Error</b>\n<code>{e}</code>"
                    )
                except Exception:
                    pass

    async def _reset_daily_counts(self):
        """Reset all users' daily signal counts at midnight UTC."""
        try:
            await db.reset_daily_signal_counts()
            logger.info("Daily signal counts reset")
        except Exception as e:
            logger.error(f"Daily reset failed: {e}")

    async def _sync_broker_balance(self):
        """Sync live account balance from broker into circuit breaker."""
        try:
            if self.engine.broker:
                account = await self.engine.broker.get_account()
                self.engine.circuit_breaker.update_balance(account.balance)
        except Exception as e:
            logger.debug(f"Balance sync skipped: {e}")

    async def _send_daily_digest(self):
        """
        Send daily signal digest to subscribed recipients.
        Also sends a Telegram summary of yesterday's signals.
        """
        try:
            signals = [d.signal for d in self._last_signals if d.signal]

            # ── Email digest (existing) ──
            if self.alert_recipients and signals:
                await asyncio.to_thread(
                    self.alert_manager.send_daily_digest,
                    signals,
                    self.alert_recipients,
                )
                logger.info(f"Daily digest sent to {len(self.alert_recipients)} recipients")

            # ── Telegram daily summary ──
            try:
                today = datetime.now(timezone.utc).strftime("%d %b %Y")
                if signals:
                    lines = [f"📊 <b>Daily Digest — {today}</b>\n"]
                    for s in signals:
                        pair = getattr(s, "pair", "?")
                        direction = getattr(s, "direction", "?")
                        entry = getattr(s, "entry", "?")
                        lines.append(f"• {pair} {direction} @ {entry}")
                    lines.append(f"\nTotal signals: <b>{len(signals)}</b>")
                    await telegram.send_message("\n".join(lines))
                else:
                    await telegram.send_message(
                        f"📊 <b>Daily Digest — {today}</b>\n"
                        f"No signals fired in the last 24 hours."
                    )
            except Exception as tg_err:
                logger.warning(f"Telegram daily digest failed: {tg_err}")

        except Exception as e:
            logger.error(f"Daily digest failed: {e}")

    async def trigger_scan_now(self) -> List:
        """Manually trigger an immediate scan (for testing or on-demand)."""
        async with self._lock:
            decisions = await self.engine.run_cycle(alert_recipients=self.alert_recipients)
            self._scan_count += 1
            self._last_scan = datetime.now(timezone.utc)

            signals_found = [d for d in decisions if d.passed]
            self._last_signals = signals_found

            # ── Telegram: manual scan result ──
            try:
                if signals_found:
                    for d in signals_found:
                        if d.signal:
                            await telegram.send_signal_alert(d.signal)
                else:
                    await telegram.send_no_signal_update(
                        pairs_checked=len(decisions) or len(self.WATCHLIST)
                    )
            except Exception as tg_err:
                logger.warning(f"Telegram manual scan alert failed: {tg_err}")

            return decisions


# Singleton — initialized in main.py lifespan
_scheduler_instance: Optional[TradingScheduler] = None


def get_scheduler() -> Optional[TradingScheduler]:
    return _scheduler_instance


def init_scheduler(engine: LiveTradeEngine, alert_recipients: Optional[List[str]] = None) -> TradingScheduler:
    global _scheduler_instance
    _scheduler_instance = TradingScheduler(engine=engine, alert_recipients=alert_recipients)
    return _scheduler_instance
