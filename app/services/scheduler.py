"""
PriceIQ Pro — Final Scheduler v5.3

Every component fully wired:
    ✅ Parallel gate execution (asyncio.gather for gates 0-3)
    ✅ Gap manager: Friday close + Sunday open hooks
    ✅ VaR end_of_day: daily return snapshot at midnight
    ✅ Performance attribution: rebuilt from journal daily
    ✅ Augmentation validator: runs after weekly retrain
    ✅ WebSocket broadcaster: pushes updates to dashboard
    ✅ Paper trading: dry-run mode support
    ✅ Equity curve: records daily snapshot
    ✅ Background task cache: heavy computations pre-warmed
    ✅ MTF confluence: H4/D1 candles fetched and updated
    ✅ Regime transition: updated every bar
    ✅ Hybrid correlation: fed prices every tick
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False

from app.services.trade_engine import LiveTradeEngine, EngineConfig
from app.services.database import db
from app.services.alerts import AlertManager
from app.services.telegram_bot import telegram
from app.services.core.v5_settings import v5_settings

WATCHLIST = v5_settings.WATCHLIST
DRY_RUN   = os.environ.get("DRY_RUN_MODE", "false").lower() == "true"


class TradingScheduler:

    def __init__(self, engine: LiveTradeEngine, alert_recipients: Optional[List[str]] = None):
        self.engine           = engine
        self.alert_recipients = alert_recipients or []
        self.alert_manager    = AlertManager()
        self._scheduler       = None
        self._running         = False
        self._scan_count      = 0
        self._last_scan       = None
        self._lock            = asyncio.Lock()
        self._v5              = None

    def start(self):
        if not _SCHEDULER_AVAILABLE:
            logger.warning("APScheduler not installed — scheduler disabled")
            return

        # Import v5 here (initialised before scheduler starts)
        from app.services.v5_orchestrator_final import get_v5
        self._v5 = get_v5()
        if not self._v5:
            logger.error("V5 orchestrator not initialised — scheduler cannot start")
            return

        # Pre-warm background task cache
        self._setup_background_cache()

        self._scheduler = AsyncIOScheduler(timezone="UTC")

        # ── Every bar (1h): main scan ──────────────────────────
        self._scheduler.add_job(
            self._run_v5_hourly_scan,
            CronTrigger(minute=1),
            id="v5_hourly_scan", name="V5 Hourly Scan",
            max_instances=1, misfire_grace_time=300, coalesce=True,
        )

        # ── 4H scan ───────────────────────────────────────────
        self._scheduler.add_job(
            self._run_v5_4h_scan,
            CronTrigger(hour="0,4,8,12,16,20", minute=1),
            id="v5_4h_scan", name="V5 4H Scan",
            max_instances=1, coalesce=True,
        )

        # ── Daily midnight: reset + VaR daily snapshot ────────
        self._scheduler.add_job(
            self._daily_midnight,
            CronTrigger(hour=0, minute=0),
            id="daily_midnight", name="Daily Midnight Tasks",
        )

        # ── Every 5 min: broker balance + correlation + trade mgr tick ──
        self._scheduler.add_job(
            self._five_minute_tick,
            IntervalTrigger(minutes=5),
            id="five_min_tick", name="5-Min Tick",
        )

        # ── Daily 8AM: digest + attribution report ─────────────
        self._scheduler.add_job(
            self._send_daily_digest,
            CronTrigger(hour=8, minute=0),
            id="daily_digest", name="Daily Digest",
        )

        # ── Friday 17:00 UTC: gap risk check ──────────────────
        self._scheduler.add_job(
            self._friday_close,
            CronTrigger(day_of_week="fri", hour=17, minute=0),
            id="friday_close", name="Friday Gap Risk Check",
        )

        # ── Sunday 21:00 UTC: gap detection on open ───────────
        self._scheduler.add_job(
            self._sunday_open,
            CronTrigger(day_of_week="sun", hour=21, minute=0),
            id="sunday_open", name="Sunday Gap Detection",
        )

        # ── Sunday 2AM UTC: weekly retrain ────────────────────
        self._scheduler.add_job(
            self._weekly_retrain,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_retrain", name="Weekly Regime Retrain",
        )

        # ── Hourly heartbeat ──────────────────────────────────
        self._scheduler.add_job(
            self._heartbeat,
            IntervalTrigger(hours=1),
            id="heartbeat", name="System Heartbeat",
        )

        # ── Paper trading daily summary (if dry-run) ──────────
        if DRY_RUN:
            self._scheduler.add_job(
                self._paper_daily_summary,
                CronTrigger(hour=22, minute=0),
                id="paper_summary", name="Paper Trading Summary",
            )

        self._scheduler.start()
        self._running = True
        logger.info(f"Scheduler started — {'DRY RUN MODE' if DRY_RUN else 'LIVE MODE'}")
        logger.info(f"Watchlist: {WATCHLIST}")

    def stop(self):
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    def get_status(self) -> dict:
        jobs = []
        if self._scheduler:
            for job in self._scheduler.get_jobs():
                jobs.append({
                    "id": job.id, "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                })
        v5_status = self._v5.get_system_status() if self._v5 else {}
        return {
            "running":    self._running,
            "scan_count": self._scan_count,
            "last_scan":  self._last_scan.isoformat() if self._last_scan else None,
            "dry_run":    DRY_RUN,
            "jobs":       jobs,
            "v5":         v5_status,
        }

    # ════════════════════════════════════════════════════════
    # HOURLY SCAN — full parallel-gate signal cycle
    # ════════════════════════════════════════════════════════

    async def _run_v5_hourly_scan(self):
        async with self._lock:
            try:
                v5 = self._v5
                self._scan_count += 1
                self._last_scan   = datetime.now(timezone.utc)
                logger.info(f"V5 hourly scan #{self._scan_count}")

                signals_found = []

                # Fetch all H1 candles in parallel (independent fetches)
                candles_map = await self._fetch_candles_parallel(WATCHLIST, "1h", 300)

                for pair in WATCHLIST:
                    try:
                        candles = candles_map.get(pair, [])
                        if not candles:
                            continue

                        # ── Update MTF confluence with H4/D1 ────────────
                        if v5_settings.MTF_ENABLED and hasattr(v5, "mtf_confluence"):
                            await self._update_mtf(v5, pair, candles)

                        # ── Run full signal cycle ────────────────────────
                        result = await v5.run_signal_cycle(
                            candles=candles, pair=pair, timeframe="1h",
                            account_balance=self.engine.config.account_balance
                                if hasattr(self.engine, "config") else None,
                            signal_bar_index=self._scan_count,
                        )

                        if result.signal_fired:
                            signals_found.append(result)
                            await db.save_signal(result.to_dict())

                            if DRY_RUN and hasattr(v5, "paper_trader"):
                                await v5.paper_trader.open_paper_trade(result)
                            else:
                                await self._send_v5_signal_alert(result)

                            # Push to WebSocket dashboard
                            await self._ws_broadcast_signal(result)

                        else:
                            logger.info(f"{pair}: {result.gate_summary()}")
                            await self._ws_broadcast_block(result)

                    except Exception as e:
                        logger.error(f"V5 scan error ({pair}): {e}", exc_info=True)

                # No-signal heartbeat
                if not signals_found:
                    try:
                        await telegram.send_no_signal_update(pairs_checked=len(WATCHLIST))
                    except Exception:
                        pass

                # Update background cache with fresh status
                if hasattr(self, "_bg_cache") and self._bg_cache:
                    self._bg_cache.trigger("full_status")

            except Exception as e:
                logger.error(f"V5 hourly scan failed: {e}", exc_info=True)
                try:
                    await telegram.send_message(f"⚠️ <b>Scan Error</b>\n<code>{e}</code>")
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════
    # 4H SCAN
    # ════════════════════════════════════════════════════════

    async def _run_v5_4h_scan(self):
        async with self._lock:
            try:
                v5 = self._v5
                logger.info("V5 4H scan")
                candles_map = await self._fetch_candles_parallel(WATCHLIST, "4h", 200)

                for pair in WATCHLIST:
                    candles = candles_map.get(pair, [])
                    if not candles:
                        continue
                    try:
                        result = await v5.run_signal_cycle(
                            candles=candles, pair=pair, timeframe="4h",
                            signal_bar_index=self._scan_count * 10,  # distinct index
                        )
                        if result.signal_fired:
                            await db.save_signal(result.to_dict())
                            if not DRY_RUN:
                                await self._send_v5_signal_alert(result, prefix="[4H] ")
                    except Exception as e:
                        logger.error(f"4H scan error ({pair}): {e}")
            except Exception as e:
                logger.error(f"V5 4H scan failed: {e}")

    # ════════════════════════════════════════════════════════
    # 5-MINUTE TICK — balance, correlations, trade manager
    # ════════════════════════════════════════════════════════

    async def _five_minute_tick(self):
        try:
            v5 = self._v5

            # 1. Sync broker balance
            if self.engine.broker:
                try:
                    account = await self.engine.broker.get_account()
                    self.engine.circuit_breaker.update_balance(account.balance)
                    v5.governor.current_balance = account.balance
                    v5.vol_sizer.update_balance(account.balance)
                    v5.var_engine.update_balance(account.balance)
                    if hasattr(v5.var_engine, "aggregator"):
                        v5.var_engine.aggregator._current_balance = account.balance
                    if account.balance > v5.governor.peak_balance:
                        v5.governor.peak_balance = account.balance
                except Exception as e:
                    logger.debug(f"Balance sync skipped: {e}")

            # 2. Get current prices and run trade manager tick
            current_prices = await self._get_current_prices()
            if current_prices:
                # Trade manager: update TP/SL/trail on all positions
                events = await v5.trade_manager.update_all(current_prices)

                # Feed prices to hybrid correlation estimator
                for pair, price in current_prices.items():
                    v5.corr_estimator.update(pair, price)

                # Paper trading: update simulated positions
                if DRY_RUN and hasattr(v5, "paper_trader"):
                    await v5.paper_trader.update_paper_trades(current_prices)

                # Push position updates to WebSocket dashboard
                if events:
                    await self._ws_push_positions()

        except Exception as e:
            logger.warning(f"5-min tick error: {e}")

    # ════════════════════════════════════════════════════════
    # DAILY MIDNIGHT
    # ════════════════════════════════════════════════════════

    async def _daily_midnight(self):
        try:
            v5 = self._v5

            # 1. Reset signal counts in DB
            await db.reset_daily_signal_counts()

            # 2. VaR daily return snapshot ← KEY FIX
            balance = v5.governor.current_balance
            if hasattr(v5.var_engine, "end_of_day"):
                v5.var_engine.end_of_day(balance)
                logger.info(f"VaR daily snapshot: balance={balance:.2f}")

            # 3. Equity curve daily mark
            if hasattr(v5, "equity_tracker"):
                v5.equity_tracker.record(balance, event="day_end",
                    note=f"Daily close: DD={v5.governor.drawdown:.1%}")

            # 4. Rebuild performance attribution from journal
            if hasattr(v5, "attribution") and hasattr(v5, "journal"):
                n = v5.attribution.load_from_journal(v5.journal)
                logger.info(f"Attribution rebuilt from {n} trades")
                if hasattr(self, "_bg_cache") and self._bg_cache:
                    self._bg_cache.trigger("attribution")

            logger.info("Daily midnight tasks complete")
        except Exception as e:
            logger.error(f"Daily midnight tasks failed: {e}")

    # ════════════════════════════════════════════════════════
    # DAILY 8AM DIGEST
    # ════════════════════════════════════════════════════════

    async def _send_daily_digest(self):
        try:
            v5 = self._v5
            await v5.send_daily_summary()

            # Also send attribution summary if enough trades
            if hasattr(v5, "attribution") and v5.attribution._trades:
                summary = v5.attribution.summary_table()
                if len(v5.attribution._trades) >= 10:
                    await telegram.send_message(summary[:3000])  # Telegram limit

        except Exception as e:
            logger.error(f"Daily digest failed: {e}")

    # ════════════════════════════════════════════════════════
    # FRIDAY CLOSE — gap risk
    # ════════════════════════════════════════════════════════

    async def _friday_close(self):
        try:
            v5 = self._v5
            if not hasattr(v5, "gap_manager"):
                return
            current_prices = await self._get_current_prices()
            if current_prices:
                actions = await v5.gap_manager.on_friday_close(
                    open_positions=v5.governor._open_positions,
                    current_prices=current_prices,
                )
                logger.info(f"Friday gap risk check: {actions}")
        except Exception as e:
            logger.error(f"Friday close task failed: {e}")

    # ════════════════════════════════════════════════════════
    # SUNDAY OPEN — gap detection
    # ════════════════════════════════════════════════════════

    async def _sunday_open(self):
        try:
            v5 = self._v5
            if not hasattr(v5, "gap_manager"):
                return
            # Fetch fresh candles after weekend
            candles_map = await self._fetch_candles_parallel(WATCHLIST, "1h", 10)
            reports = await v5.gap_manager.on_sunday_open(
                pair_candles=candles_map,
                open_positions=v5.governor._open_positions,
            )
            logger.info(f"Sunday gap detection: {len(reports)} pair(s) checked")
        except Exception as e:
            logger.error(f"Sunday open task failed: {e}")

    # ════════════════════════════════════════════════════════
    # WEEKLY RETRAIN — regime classifier + augmentation validation
    # ════════════════════════════════════════════════════════

    async def _weekly_retrain(self):
        try:
            v5 = self._v5
            logger.info("Weekly regime classifier retrain starting...")

            # 1. Collect candles from all pairs
            all_candles = []
            for pair in WATCHLIST[:3]:  # use first 3 pairs
                candles = await self._get_candles(pair, "1h", 1000)
                if candles:
                    all_candles.extend(candles)

            if len(all_candles) < 200:
                logger.warning(f"Weekly retrain: insufficient candles ({len(all_candles)})")
                return

            # 2. Run augmentation A/B validation first
            if v5_settings.AUGMENTATION_ENABLED and hasattr(v5, "augmenter"):
                try:
                    from app.services.ml.augmentation_validator import AugmentationValidator
                    av      = AugmentationValidator()
                    ab_result = await asyncio.to_thread(av.run, all_candles, v5.augmenter)
                    logger.info(f"Augmentation A/B: {ab_result.recommendation}")
                    use_aug = ab_result.use_augmentation
                except Exception as e:
                    logger.warning(f"Augmentation validator failed: {e} — using default True")
                    use_aug = v5_settings.AUGMENTATION_ENABLED
            else:
                use_aug = False

            # 3. Retrain with validated augmentation decision
            result = await v5.train_regime_classifier(
                all_candles,
                save_path=v5_settings.REGIME_MODEL_PATH,
                use_augmentation=use_aug,
            )

            # 4. Notify
            await telegram.send_message(
                f"🧠 <b>Weekly Retrain Complete</b>\n"
                f"Samples: {result.get('n_samples', 0)}\n"
                f"CV Accuracy: {result.get('cv_accuracy', 0):.1%} "
                f"± {result.get('cv_std', 0):.1%}\n"
                f"Augmentation: {'✅ enabled' if use_aug else '❌ skipped'}\n"
                f"Classes: {result.get('class_dist', {})}"
            )
            logger.info(f"Weekly retrain: {result}")

        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}", exc_info=True)
            try:
                await telegram.send_message(f"❌ <b>Weekly Retrain Failed</b>\n<code>{e}</code>")
            except Exception:
                pass

    # ════════════════════════════════════════════════════════
    # HEARTBEAT
    # ════════════════════════════════════════════════════════

    async def _heartbeat(self):
        try:
            v5 = self._v5
            await v5.send_heartbeat()

            # Push heartbeat to WebSocket clients
            try:
                from app.services.api.websocket_dashboard import ws_broadcaster
                portfolio = v5.governor.get_portfolio_summary()
                await ws_broadcaster.heartbeat({
                    "balance":  portfolio.get("current_balance"),
                    "drawdown": portfolio.get("drawdown"),
                    "regime":   v5.regime_transition.regime_streak()[0] if hasattr(v5, "regime_transition") else "?",
                    "signals":  not getattr(v5, "_signals_paused", False),
                })
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    # ════════════════════════════════════════════════════════
    # PAPER TRADING DAILY SUMMARY
    # ════════════════════════════════════════════════════════

    async def _paper_daily_summary(self):
        try:
            v5 = self._v5
            if hasattr(v5, "paper_trader"):
                await v5.paper_trader.send_daily_paper_summary()
        except Exception as e:
            logger.error(f"Paper daily summary failed: {e}")

    # ════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════

    async def _fetch_candles_parallel(
        self, pairs: List[str], timeframe: str, limit: int
    ) -> dict:
        """Fetch candles for multiple pairs in parallel."""
        tasks = {pair: self._get_candles(pair, timeframe, limit) for pair in pairs}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            pair: (candles if not isinstance(candles, Exception) else [])
            for pair, candles in zip(tasks.keys(), results)
        }

    async def _get_candles(self, pair: str, timeframe: str, limit: int = 300):
        """Get candles from engine data fetcher or broker."""
        v5 = self._v5

        # Check cache first
        if hasattr(v5, "candle_cache"):
            cached = v5.candle_cache.get(pair, timeframe)
            if cached:
                return cached

        candles = []
        try:
            if hasattr(self.engine, "data_fetcher") and self.engine.data_fetcher:
                candles = await self.engine.data_fetcher.get_candles(pair, timeframe, limit=limit)
            elif self.engine.broker:
                candles = await self.engine.broker.get_candles(pair, timeframe, limit=limit)
        except Exception as e:
            logger.warning(f"Could not fetch {pair}/{timeframe}: {e}")

        if candles and hasattr(v5, "candle_cache"):
            v5.candle_cache.set(pair, timeframe, candles)

        return candles

    async def _get_current_prices(self) -> dict:
        """Get latest price for all watchlist pairs."""
        prices = {}
        try:
            if self.engine.broker:
                for pair in WATCHLIST:
                    try:
                        # Try get_price method first (not all brokers have it)
                        if hasattr(self.engine.broker, "get_price"):
                            price = await self.engine.broker.get_price(pair)
                        elif hasattr(self.engine.broker, "get_candles"):
                            candles = await self.engine.broker.get_candles(pair, "1m", limit=1)
                            price   = candles[-1].close if candles else None
                        else:
                            price = None
                        if price:
                            prices[pair] = price
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Price fetch error: {e}")
        return prices

    async def _update_mtf(self, v5, pair: str, h1_candles: list):
        """Update MTF confluence with H4 and D1 candles."""
        try:
            h4_candles = await self._get_candles(pair, "4h", 100)
            d1_candles = await self._get_candles(pair, "1d", 100)
            if h4_candles or d1_candles:
                v5.mtf_confluence.update(
                    pair,
                    h1_candles=h1_candles,
                    h4_candles=h4_candles or h1_candles,
                    d1_candles=d1_candles or h1_candles,
                )
        except Exception as e:
            logger.debug(f"MTF update error ({pair}): {e}")

    async def _send_v5_signal_alert(self, result, prefix: str = ""):
        """Format and send signal alert via Telegram."""
        try:
            dir_emoji    = "🟢" if result.direction == "buy" else "🔴"
            regime_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "⚡"}.get(result.regime, "❓")
            drift_warn   = " ⚠️DRIFT" if getattr(result, "drift_warn", False) else ""

            msg = (
                f"{dir_emoji} <b>{prefix}{result.pair} {result.direction.upper()}</b>{drift_warn}\n\n"
                f"Agent: <b>{result.agent_used}</b>\n"
                f"Regime: {regime_emoji} {result.regime.title()} "
                f"(T:{result.regime_probs.get('trending',0):.0%} "
                f"R:{result.regime_probs.get('ranging',0):.0%} "
                f"V:{result.regime_probs.get('volatile',0):.0%})\n"
                f"Shift risk: {result.regime_shift_risk:.0%}\n"
                f"Conflict: {result.conflict_score:.2f}\n"
                f"Confidence: <b>{result.confidence:.0%}</b> "
                f"(WP: {result.win_probability:.0%} EV: {result.expected_value:+.2f}R)\n"
                f"Session: {result.session}\n\n"
                f"Entry: <code>{result.fill_price:.5f}</code>\n"
                f"Stop:  <code>{result.stop_loss:.5f}</code>\n"
                f"TP1:   <code>{result.take_profit_1:.5f}</code>\n"
                f"TP2:   <code>{result.take_profit_2:.5f}</code>\n"
                f"Lots: {result.adjusted_lots} ({result.sizing_method})\n\n"
                f"Heat: {result.portfolio_heat:.1%} | Risk: {result.risk_score:.2f}\n"
                f"TF: {result.timeframe}"
            )
            await telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Signal alert failed: {e}")

    async def _ws_broadcast_signal(self, result):
        """Push signal to WebSocket dashboard."""
        try:
            from app.services.api.websocket_dashboard import ws_broadcaster
            if ws_broadcaster.n_connections > 0:
                await ws_broadcaster.signal_fired(result)
        except Exception:
            pass

    async def _ws_broadcast_block(self, result):
        """Push block event to WebSocket dashboard."""
        try:
            from app.services.api.websocket_dashboard import ws_broadcaster
            if ws_broadcaster.n_connections > 0:
                await ws_broadcaster.signal_blocked(
                    result.pair, result.reasoning[:100],
                    result.gate_summary()
                )
        except Exception:
            pass

    async def _ws_push_positions(self):
        """Push current position state to WebSocket dashboard."""
        try:
            from app.services.api.websocket_dashboard import ws_broadcaster
            if ws_broadcaster.n_connections > 0 and self._v5:
                positions = self._v5.trade_manager.get_open_positions()
                await ws_broadcaster.position_update(positions)
        except Exception:
            pass

    def _setup_background_cache(self):
        """Register heavy tasks with the background cache."""
        try:
            from app.services.core.background_task_cache import BackgroundTaskCache
            v5 = self._v5
            cache = BackgroundTaskCache()

            cache.register("full_status",  lambda: v5.get_system_status(), ttl=60)
            cache.register("montecarlo",   lambda: v5.run_montecarlo(n_paths=1000), ttl=3600)
            cache.register("attribution",
                           lambda: {"summary": v5.attribution.summary_table(),
                                    "report": v5.attribution.full_report()},
                           ttl=1800)

            asyncio.create_task(cache.background_worker())
            self._bg_cache = cache

            # Pre-warm on startup
            cache.trigger("full_status")
            logger.info("Background task cache configured and warming up")
        except Exception as e:
            logger.warning(f"Background cache setup failed: {e}")
            self._bg_cache = None

    async def trigger_scan_now(self) -> List:
        """Manual trigger for on-demand scan."""
        await self._run_v5_hourly_scan()
        return []


# ── Singleton ─────────────────────────────────────────────────
_scheduler_instance: Optional[TradingScheduler] = None


def get_scheduler() -> Optional[TradingScheduler]:
    return _scheduler_instance


def init_scheduler(
    engine: LiveTradeEngine,
    alert_recipients: Optional[List[str]] = None,
) -> TradingScheduler:
    global _scheduler_instance
    _scheduler_instance = TradingScheduler(engine=engine, alert_recipients=alert_recipients)
    return _scheduler_instance
