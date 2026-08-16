"""PriceIQ Pro V5 -- Main Entry Point (SAFE v6 -- circular import fix)"""
import sys, os, asyncio, logging, httpx
from app.services.monitoring.signal_bot import signal_bot
from app.services.learning.signal_performance_tracker import tracker
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_signal_cooldown: dict = {}
SIGNAL_COOLDOWN_MIN = 60

async def trading_loop():
    # -- Import here to avoid circular import at module level --
    from app.services.risk.overnight_gap_manager import OvernightGapManager, HIGH_GAP_PAIRS

    scan_count = 0
    await asyncio.sleep(10)
    try:
        from app.services.data_fetcher import DataFetcher
        fetcher = DataFetcher()
    except Exception as e:
        logger.error(f"DataFetcher init failed: {e}")
        return

    gap_mgr = OvernightGapManager(telegram=None)
    _thursday_closes: dict = {}

    watchlist = ["XAUUSD", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "BTCUSD"]

    while True:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            monday = now + timedelta(days=days_until_monday)
            monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_seconds = (monday - now).total_seconds()
            logger.info(f"Weekend shutdown: {now.strftime('%A %H:%M UTC')} -- sleeping {int(sleep_seconds/3600)}h until Monday 00:00 UTC")
            await asyncio.sleep(sleep_seconds)
            continue

        try:
            from app.services.v5_orchestrator_final import get_v5
            v5 = get_v5()
            if v5 and hasattr(v5, "run_signal_cycle"):
                scan_count += 1
                logger.info(f"Scan #{scan_count} starting...")
                for pair in watchlist:
                    try:
                        candles = await fetcher.get_candles(pair, "1h", limit=300)
                        if candles:
                            try:
                                resolved = await tracker.update_with_candles(pair, candles)
                                for sig in resolved:
                                    try:
                                        await signal_bot.broadcast_outcome(
                                            sig["pair"], sig["direction"], sig["outcome"], sig["pnl_r"]
                                        )
                                    except Exception as be:
                                        logger.warning(f"Outcome broadcast error: {be}")
                            except Exception as te:
                                logger.warning(f"Tracker resolve error: {te}")
                        if candles and len(candles) >= 55:
                            candles = [c for c in candles
                                       if getattr(c, "high", 1) != getattr(c, "low", 0)]
                            if len(candles) < 55:
                                logger.warning(f"Too few candles after sanitization: {pair}")
                                await asyncio.sleep(3)
                                continue

                            now = datetime.now(timezone.utc)
                            if gap_mgr.is_friday_close_window(now):
                                if pair.upper() in HIGH_GAP_PAIRS:
                                    logger.warning(f"FRIDAY GAP RISK: {pair} is high-gap pair. Consider closing before weekend.")
                                if candles:
                                    last_close = getattr(candles[-1], "close", None)
                                    if last_close:
                                        _thursday_closes[pair.upper()] = float(last_close)

                            if gap_mgr.is_sunday_open_window(now):
                                thu_close = _thursday_closes.get(pair.upper())
                                if thu_close and candles:
                                    sun_open = getattr(candles[0], "open", None)
                                    if sun_open:
                                        gap = abs(float(sun_open) - thu_close)
                                        atr = gap_mgr._atr(candles)
                                        if atr > 0 and gap > atr * 1.5:
                                            logger.warning(f"SUNDAY GAP: {pair} gap={gap:.5f} ({gap/atr:.1f}x ATR) vs Thu close {thu_close:.5f}")

                            now = datetime.now(timezone.utc)
                            last_time = _signal_cooldown.get(pair)
                            if last_time and (now - last_time) < timedelta(minutes=SIGNAL_COOLDOWN_MIN):
                                mins_ago = int((now - last_time).total_seconds() / 60)
                                logger.info(f"Signal cooldown: {pair} skipped ({mins_ago}m ago)")
                                await asyncio.sleep(3)
                                continue

                            result = await v5.run_signal_cycle(
                                candles=candles,
                                pair=pair,
                                timeframe="1h",
                                signal_bar_index=scan_count,
                            )
                            if result and result.signal_fired:
                                _signal_cooldown[pair] = datetime.now(timezone.utc)
                                logger.info(f"SIGNAL: {pair} {result.direction} conf={result.confidence:.0%}")

                                try:
                                    await tracker.record(result)
                                except Exception as te:
                                    logger.warning(f"Tracker record error: {te}")

                                # --- PUBLIC BROADCAST (channel + subscribers + admin) ---
                                try:
                                    await signal_bot.broadcast_signal(result)
                                    logger.info("Public signal broadcast sent")
                                except Exception as be:
                                    logger.warning(f"Public broadcast error: {be}")
                            else:
                                logger.debug(f"No signal: {pair}")
                        else:
                            logger.warning(f"Insufficient candles: {pair} got {len(candles) if candles else 0}")
                        await asyncio.sleep(3)
                    except Exception as e:
                        logger.warning(f"Pair error ({pair}): {e}")

                now = datetime.now(timezone.utc)
                if gap_mgr.is_friday_close_window(now):
                    try:
                        positions = getattr(v5, "trade_manager", None)
                        if positions and hasattr(positions, "_positions"):
                            open_pos = positions._positions
                            current_prices = {}
                            for p in watchlist:
                                try:
                                    c = await fetcher.get_candles(p, "1h", limit=1)
                                    if c:
                                        current_prices[p.upper()] = getattr(c[-1], "close", 0)
                                except Exception:
                                    pass
                            await gap_mgr.on_friday_close(open_pos, current_prices)
                    except Exception as e:
                        logger.warning(f"Friday gap check failed: {e}")

                # --- Bar time stops ---
                try:
                    if hasattr(v5, "trade_manager"):
                        if not hasattr(v5, '_scan_count'):
                            v5._scan_count = 0
                        v5._scan_count = scan_count
                        current_prices = {}
                        for p in watchlist:
                            try:
                                c = await fetcher.get_candles(p, "1h", limit=1)
                                if c:
                                    current_prices[p.upper()] = getattr(c[-1], "close", 0)
                            except Exception:
                                pass
                        await v5.trade_manager.check_bar_stops(v5._scan_count, current_prices, max_bars=8)
                except Exception as e:
                    logger.warning(f"Bar stop check: {e}")

                logger.info(f"Scan #{scan_count} complete")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PriceIQ Pro V5 Starting...")
    try:
        from app.services.telegram_bot import telegram as telegram_bot
        from app.services.v5_orchestrator_final import init_v5
        from app.services.core.v5_settings import v5_settings
        v5 = init_v5(
            starting_balance=v5_settings.ACCOUNT_BALANCE,
            risk_pct=v5_settings.RISK_PERCENT,
            target_vol_pct=v5_settings.TARGET_VOL_PCT,
            max_drawdown=v5_settings.MAX_DRAWDOWN_PCT,
            regime_model_path=v5_settings.REGIME_MODEL_PATH,
            learning_state_path=v5_settings.LEARNING_STATE_PATH,
            regime_weights_path=v5_settings.REGIME_WEIGHTS_PATH,
            win_prob_path=v5_settings.WIN_PROB_PATH,
            transition_path=v5_settings.TRANSITION_PATH,
            journal_path=v5_settings.JOURNAL_PATH,
            telegram=telegram_bot,
        )
        logger.info("V5 Orchestrator initialized OK")
    except Exception as e:
        import traceback
        logger.error(f"V5 init failed: {e}")
        traceback.print_exc()

    asyncio.create_task(trading_loop())
    asyncio.create_task(signal_bot.start_polling())
    logger.info("Trading loop started")
    logger.info("Signal subscription bot started")

    try:
        from app.services.api.v5_dashboard_api import v5_router
        app.include_router(v5_router)
    except Exception as e:
        logger.warning(f"V5 dashboard router: {e}")

    try:
        from app.services.api.websocket_dashboard import ws_router
        app.include_router(ws_router)
    except Exception as e:
        logger.warning(f"WebSocket router: {e}")

    try:
        from app.services.api.broker_webhook_v5 import broker_webhook_router
        app.include_router(broker_webhook_router)
    except Exception as e:
        logger.warning(f"Broker webhook V5: {e}")

    logger.info("PriceIQ Pro V5 -- Fully operational")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="PriceIQ Pro V5", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    from app.routers import signals, signals_v32, system
    app.include_router(signals.router)
    app.include_router(signals_v32.router)
    app.include_router(system.router)
except Exception as e:
    logger.error(f"Core routers: {e}")

try:
    from app.routers import session, circuit_breaker, correlation
    from app.routers import price_validator, multi_timeframe, patterns
    app.include_router(session.router)
    app.include_router(circuit_breaker.router)
    app.include_router(correlation.router)
    app.include_router(price_validator.router)
    app.include_router(multi_timeframe.router)
    app.include_router(patterns.router)
except Exception as e:
    logger.error(f"Feature routers: {e}")

try:
    from app.routers import auth, database, monte_carlo, scheduler
    app.include_router(auth.router)
    app.include_router(database.router)
    app.include_router(monte_carlo.router)
    app.include_router(scheduler.router)
except Exception as e:
    logger.error(f"Admin routers: {e}")

try:
    from app.routers import trade_engine
    app.include_router(trade_engine.router)
except Exception as e:
    logger.warning(f"Trade engine: {e}")

try:
    from app.routers import broker_webhook
    app.include_router(broker_webhook.router)
except Exception as e:
    logger.warning(f"Broker webhook: {e}")

try:
    from app.routers.v5 import router as v5_fallback
    app.include_router(v5_fallback)
except Exception as e:
    logger.warning(f"V5 fallback: {e}")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat()}

@app.head("/health")
async def health_head():
    return {"status": "healthy"}

@app.get("/")
async def root():
    try:
        from app.services.v5_orchestrator_final import get_v5
        v5_status = "running" if get_v5() else "not initialised"
    except Exception:
        v5_status = "error"
    return {"app": "PriceIQ Pro V5", "version": "5.0.0",
            "status": "running", "v5": v5_status, "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
