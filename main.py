"""
PriceIQ Pro V5 — Professional Entry Point
Runs the REAL V5 orchestrator, not dummy data.
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

# ============ REAL V5 IMPORTS ============
try:
    from app.services.v5_orchestrator_final import init_v5, get_v5
    from app.services.core.v5_settings import v5_settings
    from app.services.advanced_integration import attach_advanced_capabilities
    logger.info("✅ V5 modules loaded")
    V5_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ V5 import failed: {e}")
    V5_AVAILABLE = False

# ============ EXISTING ROUTERS ============
from app.routers import signals, signals_v32, system, session, circuit_breaker, auth, database
from app.routers import price_validator, multi_timeframe, correlation, trade_engine as trade_engine_router
from app.routers import patterns, monte_carlo, scheduler as scheduler_router

# ============ LIFESPAN - REAL V5 STARTS HERE ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀 Starting PriceIQ Pro V5 - PROFESSIONAL MODE")
    logger.info("=" * 60)

    v5 = None
    
    if V5_AVAILABLE:
        try:
            logger.info("Initializing REAL V5 orchestrator...")
            
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
                fred_api_key=os.environ.get("FRED_API_KEY", ""),
                alpha_vantage_key=os.environ.get("ALPHA_VANTAGE_KEY", ""),
                finnhub_key=os.environ.get("FINNHUB_KEY", ""),
            )
            
            logger.info("✅ REAL V5 orchestrator initialized successfully")
            
            # Attach advanced capabilities
            if attach_advanced_capabilities:
                try:
                    attach_advanced_capabilities(v5, watchlist=v5_settings.WATCHLIST)
                    logger.info("✅ Advanced capabilities attached")
                except Exception as e:
                    logger.warning(f"⚠️ Advanced capabilities: {e}")
                    
        except Exception as e:
            logger.error(f"❌ V5 init FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("❌ V5 not available - check imports")

    logger.info("=" * 60)
    logger.info(f"✅ PriceIQ Pro V5 - {'V5 ACTIVE' if v5 else 'MINIMAL MODE'}")
    logger.info("=" * 60)

    yield

    logger.info("🛑 Shutting down...")
    if v5 and hasattr(v5, 'shutdown'):
        await v5.shutdown()

# ============ CREATE APP ============
app = FastAPI(title="PriceIQ Pro V5", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============ ROUTES ============
@app.get("/")
async def root():
    v5 = get_v5() if get_v5 else None
    return {"app": "PriceIQ Pro V5", "version": "5.0.0", "status": "running", "v5_active": v5 is not None}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============ MOUNT ROUTERS ============
app.include_router(signals.router)
app.include_router(signals_v32.router)
app.include_router(system.router)
app.include_router(session.router)
app.include_router(price_validator.router)
app.include_router(multi_timeframe.router)
app.include_router(circuit_breaker.router)
app.include_router(correlation.router)
app.include_router(trade_engine_router.router)
app.include_router(patterns.router)
app.include_router(auth.router)
app.include_router(database.router)
app.include_router(monte_carlo.router)
app.include_router(scheduler_router.router)

# V5 Router - REAL (not dummy)
try:
    from app.services.api.v5_dashboard_api import v5_router
    app.include_router(v5_router)
    logger.info("✅ V5 API router mounted")
except ImportError as e:
    logger.warning(f"⚠️ V5 API router not found: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
