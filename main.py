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


# ============ BACKGROUND TRADING LOOP ============
async def trading_loop():
    from app.services.v5_orchestrator_final import get_v5

    while True:
        try:
            v5 = get_v5()

            if v5:
                try:
                    result = v5.run_cycle()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(f"🔥 LOOP RESULT: {result}")
                except Exception as e:
                    logger.error(f"❌ Trading loop error: {e}")

        except Exception as e:
            logger.error(f"❌ Loop outer error: {e}")

        await asyncio.sleep(60)


# ============ LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("PriceIQ Pro V5 Starting...")
    logger.info("=" * 60)

    # 🔥 DEBUGGED V5 INIT BLOCK (THIS IS THE FIX)
    try:
        from app.services.v5_orchestrator_final import init_v5
        from app.services.core.v5_settings import v5_settings

        try:
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
            logger.info("✅ V5 Orchestrator initialized OK")

        except Exception:
            import traceback
            logger.error("🚨 V5 INIT CRASHED INSIDE init_v5()")
            traceback.print_exc()

    except Exception:
        import traceback
        logger.error("🚨 V5 IMPORT FAILED")
        traceback.print_exc()

    # ============ ROUTERS ============
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

    # 🔥 START LOOP
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

# ============ FALLBACK ROUTERS ============
try:
    from app.routers.v5 import router as v5_fallback
    app.include_router(v5_fallback)
    logger.warning("⚠️ Using fallback V5 router (mock data)")
except Exception as e:
    logger.warning(f"Fallback router failed: {e}")


# ============ HEALTH ============
@app.get("/health")
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


# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
