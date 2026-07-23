"""
PriceIQ Pro V5 — Main Entry Point
"""

import sys
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============ V5 IMPORTS ============
try:
    from app.services.v5_orchestrator_final import init_v5, get_v5
    logger.info("✅ V5 orchestrator imported")
except ImportError as e:
    logger.warning(f"⚠️ V5 orchestrator import error: {e}")
    init_v5 = None
    get_v5 = None

try:
    from app.services.advanced_integration import attach_advanced_capabilities
    logger.info("✅ Advanced integration imported")
except ImportError as e:
    logger.warning(f"⚠️ Advanced integration error: {e}")
    attach_advanced_capabilities = None

try:
    from app.services.core.v5_settings import v5_settings
    logger.info("✅ V5 settings imported")
except ImportError as e:
    logger.warning(f"⚠️ V5 settings error: {e}")
    v5_settings = None

try:
    from app.services.core.preflight import PreflightValidator
    logger.info("✅ Preflight imported")
except ImportError as e:
    logger.warning(f"⚠️ Preflight error: {e}")
    PreflightValidator = None

# ============ EXISTING IMPORTS ============
try:
    from app.routers import signals, signals_v32, system, session, circuit_breaker, auth, database
    ROUTERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Some routers not available: {e}")
    ROUTERS_AVAILABLE = False

# ============ LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀 Starting PriceIQ Pro V5")
    logger.info("=" * 60)

    # Initialize V5 if available
    if init_v5 and v5_settings:
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
            logger.info("✅ V5 orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ V5 init failed: {e}")

    logger.info("=" * 60)
    logger.info("✅ PriceIQ Pro V5 is running!")
    logger.info("=" * 60)

    yield

    logger.info("🛑 Shutting down PriceIQ Pro V5...")

# ============ CREATE APP ============
app = FastAPI(
    title="PriceIQ Pro V5",
    description="Autonomous AI Forex Trading System",
    version="5.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ============ MOUNT ROUTERS ============
if ROUTERS_AVAILABLE:
    try:
        from app.routers import signals, signals_v32, system
        app.include_router(signals.router)
        app.include_router(signals_v32.router)
        app.include_router(system.router)
        logger.info("✅ Core routers mounted")
    except Exception as e:
        logger.warning(f"⚠️ Core routers failed: {e}")

    try:
        from app.routers import session, circuit_breaker, auth, database
        app.include_router(session.router)
        app.include_router(circuit_breaker.router)
        app.include_router(auth.router)
        app.include_router(database.router)
        logger.info("✅ Feature routers mounted")
    except Exception as e:
        logger.warning(f"⚠️ Feature routers failed: {e}")

# ============ V5 ROUTERS ============
try:
    from app.services.api.v5_dashboard_api import v5_router
    app.include_router(v5_router)
    logger.info("✅ V5 dashboard mounted")
except ImportError as e:
    logger.warning(f"⚠️ V5 dashboard not available: {e}")

try:
    from app.services.api.broker_webhook_v5 import broker_webhook_router
    app.include_router(broker_webhook_router)
    logger.info("✅ V5 webhook mounted")
except ImportError as e:
    logger.warning(f"⚠️ V5 webhook not available: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
