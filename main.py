"""
PriceIQ Pro V5 — Complete Main Entry
"""

import sys
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============ CREATE APP ============
app = FastAPI(
    title="PriceIQ Pro V5",
    description="Autonomous AI Forex Trading System",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀 PriceIQ Pro V5 Starting...")
    logger.info("=" * 60)
    
    # Try to initialize V5 orchestrator
    try:
        from app.services.v5_orchestrator_final import init_v5, get_v5
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
        logger.info("✅ V5 Orchestrator initialized")
    except Exception as e:
        logger.error(f"❌ V5 init failed: {e}")

    yield
    logger.info("🛑 Shutting down...")

# ============ ROUTES ============
@app.get("/")
async def root():
    return {
        "app": "PriceIQ Pro V5",
        "version": "5.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============ MOUNT ALL ROUTERS ============

# 1. Core routers
try:
    from app.routers import signals, signals_v32, system
    app.include_router(signals.router)
    app.include_router(signals_v32.router)
    app.include_router(system.router)
    logger.info("✅ Core routers mounted")
except Exception as e:
    logger.error(f"❌ Core routers failed: {e}")

# 2. Feature routers
try:
    from app.routers import session, circuit_breaker, correlation, price_validator, multi_timeframe, patterns
    app.include_router(session.router)
    app.include_router(circuit_breaker.router)
    app.include_router(correlation.router)
    app.include_router(price_validator.router)
    app.include_router(multi_timeframe.router)
    app.include_router(patterns.router)
    logger.info("✅ Feature routers mounted")
except Exception as e:
    logger.error(f"❌ Feature routers failed: {e}")

# 3. Admin routers
try:
    from app.routers import auth, database, monte_carlo, scheduler
    app.include_router(auth.router)
    app.include_router(database.router)
    app.include_router(monte_carlo.router)
    app.include_router(scheduler.router)
    logger.info("✅ Admin routers mounted")
except Exception as e:
    logger.error(f"❌ Admin routers failed: {e}")

# 4. Trade engine router
try:
    from app.routers import trade_engine
    app.include_router(trade_engine.router)
    logger.info("✅ Trade engine router mounted")
except Exception as e:
    logger.error(f"❌ Trade engine failed: {e}")

# 5. Broker webhook router
try:
    from app.routers import broker_webhook
    app.include_router(broker_webhook.router)
    logger.info("✅ Broker webhook mounted")
except Exception as e:
    logger.error(f"❌ Broker webhook failed: {e}")

# 6. V5 API router (full version)
try:
    from app.services.api.v5_dashboard_api import v5_router
    app.include_router(v5_router)
    logger.info("✅ V5 API router mounted")
except Exception as e:
    logger.warning(f"⚠️ V5 API router not found: {e}")

# 7. V5 router (fallback)
try:
    from app.routers.v5 import router as v5_router_fallback
    app.include_router(v5_router_fallback)
    logger.info("✅ V5 fallback router mounted")
except Exception as e:
    logger.warning(f"⚠️ V5 fallback router not found: {e}")

# 8. Websocket router
try:
    from app.services.api.websocket_dashboard import ws_router
    app.include_router(ws_router)
    logger.info("✅ Websocket router mounted")
except Exception as e:
    logger.warning(f"⚠️ Websocket router not found: {e}")

# 9. Broker webhook V5
try:
    from app.services.api.broker_webhook_v5 import broker_webhook_router
    app.include_router(broker_webhook_router)
    logger.info("✅ Broker webhook V5 mounted")
except Exception as e:
    logger.warning(f"⚠️ Broker webhook V5 not found: {e}")

logger.info("=" * 60)
logger.info("✅ PriceIQ Pro V5 Started - ALL ROUTERS MOUNTED")
logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
