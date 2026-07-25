"""
PriceIQ Pro V5 — Main Entry Point
"""
import sys
import os
import logging
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# SAFE OPTIONAL IMPORTS (🔥 prevents crashes)
# ============================================================

# sklearn-safe (prevents crash if not installed)
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
except Exception as e:
    logger.warning(f"⚠️ sklearn not installed: {e}")

# advanced analyzer safe import
try:
    from app.services.market_analyzer_advanced import MAETradingFormula
except Exception as e:
    logger.warning(f"⚠️ Advanced analyzer missing: {e}")
    MAETradingFormula = None


# ============ BACKGROUND TRADING LOOP ============
async def trading_loop():
    try:
        from app.services.v5_orchestrator_final import get_v5
    except Exception as e:
        logger.error(f"❌ Cannot import V5: {e}")
        return

    while True:
        try:
            v5 = get_v5()

            if v5:
                result = None

                
                try:
                    if hasattr(v5, "run_signal_cycle"):
                        # Fetch candles and run signal cycle for each pair
                        try:
                            from app.services.data_fetcher import DataFetcher
                            fetcher  = DataFetcher()
                            watchlist = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
                            for pair in watchlist:
                                try:
                                    candles = await fetcher.get_candles(pair, "1h", limit=300)
                                    if candles and len(candles) >= 55:
                                        result = await v5.run_signal_cycle(
                                            candles=candles,
                                            pair=pair,
                                            timeframe="1h",
                                            signal_bar_index=_scan_count,
                                        )
                                        if result and result.signal_fired:
                                            logger.info(f"🎯 SIGNAL: {pair} {result.direction} conf={result.confidence:.0%}")
                                        else:
                                            logger.debug(f"No signal: {pair}")
                                    else:
                                        logger.warning(f"Insufficient candles for {pair}: {len(candles) if candles else 0}")
                                except Exception as pair_e:
                                    logger.warning(f"Signal cycle error ({pair}): {pair_e}")
                        except Exception as fetch_e:
                            logger.warning(f"Data fetch error: {fetch_e}")
                        result = None
                    else:
                        logger.error("❌ No valid execution method in V5")
                        result = None

                except Exception as e:
                    logger.error(f"❌ execution failed: {e}")
                    result = None
                

                if asyncio.iscoroutine(result):
                    await result

                logger.info("🔥 LOOP RUNNING — market scan executed")

        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")

        await asyncio.sleep(60)
# ============ LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("PriceIQ Pro V5 Starting...")
    logger.info("=" * 60)

    try:
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
            telegram=None,
        )
        logger.info("V5 Orchestrator initialized OK")
    except Exception as e:
        import traceback
        logger.error(f"V5 init failed: {e}")
        traceback.print_exc()

    # Mount routers
    try:
        from app.services.api.v5_dashboard_api import v5_router
        app.include_router(v5_router)
        logger.info("V5 dashboard router mounted")
    except Exception as e:
        logger.warning(f"V5 dashboard router: {e}")

    try:
        from app.services.api.websocket_dashboard import ws_router
        app.include_router(ws_router)
        logger.info("WebSocket router mounted")
    except Exception as e:
        logger.warning(f"WebSocket router: {e}")

    try:
        from app.services.api.broker_webhook_v5 import broker_webhook_router
        app.include_router(broker_webhook_router)
        logger.info("Broker webhook V5 mounted")
    except Exception as e:
        logger.warning(f"Broker webhook V5: {e}")

    # 🔥 START BACKGROUND LOOP (RENDER-SAFE)
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(trading_loop())
        logger.info("🔥 Trading loop started")
    except Exception as e:
        logger.error(f"❌ Failed to start trading loop: {e}")

    logger.info("PriceIQ Pro V5 — Fully operational")
    yield
    logger.info("Shutting down PriceIQ Pro V5...")


# ============ CREATE APP ============
app = FastAPI(
    title="PriceIQ Pro V5",
    description="Autonomous AI Forex Trading System",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ ROUTERS ============
try:
    from app.routers import signals, signals_v32, system
    app.include_router(signals.router)
    app.include_router(signals_v32.router)
    app.include_router(system.router)
    logger.info("Core routers mounted")
except Exception as e:
    logger.error(f"Core routers failed: {e}")

try:
    from app.routers import session, circuit_breaker, correlation
    from app.routers import price_validator, multi_timeframe, patterns
    app.include_router(session.router)
    app.include_router(circuit_breaker.router)
    app.include_router(correlation.router)
    app.include_router(price_validator.router)
    app.include_router(multi_timeframe.router)
    app.include_router(patterns.router)
    logger.info("Feature routers mounted")
except Exception as e:
    logger.error(f"Feature routers failed: {e}")

try:
    from app.routers import auth, database, monte_carlo, scheduler
    app.include_router(auth.router)
    app.include_router(database.router)
    app.include_router(monte_carlo.router)
    app.include_router(scheduler.router)
    logger.info("Admin routers mounted")
except Exception as e:
    logger.error(f"Admin routers failed: {e}")

try:
    from app.routers import trade_engine
    app.include_router(trade_engine.router)
except Exception as e:
    logger.warning(f"Trade engine router: {e}")

try:
    from app.routers import broker_webhook
    app.include_router(broker_webhook.router)
except Exception as e:
    logger.warning(f"Broker webhook: {e}")

try:
    from app.routers.v5 import router as v5_fallback
    app.include_router(v5_fallback)
    logger.info("V5 fallback router mounted")
except Exception as e:
    logger.warning(f"V5 fallback: {e}")


# ============ HEALTH ============
from datetime import datetime, timezone

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/")
async def root():
    try:
        from app.services.v5_orchestrator_final import get_v5
        v5 = get_v5()
        v5_status = "running" if v5 else "not initialised"
    except Exception:
        v5_status = "error"

    return {
        "app": "PriceIQ Pro V5",
        "version": "5.0.0",
        "status": "running",
        "v5": v5_status,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
