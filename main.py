"""
PriceIQ Pro v3.3 — Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.services.database import db
from app.services.scheduler import init_scheduler, get_scheduler
from app.services.trade_engine import LiveTradeEngine, EngineConfig
from app.services.broker_connector import create_broker
from app.services.auth import validate_auth_config
from app.services.twelve_data_client import init_twelve_data, close_twelve_data

# Import all routers
from app.routers import signals, signals_v32, broker_webhook, system
from app.routers import session, price_validator, multi_timeframe
from app.routers import circuit_breaker, correlation, trade_engine as trade_engine_router
from app.routers import auth, database, monte_carlo, patterns, scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown."""

    # ========== STARTUP ==========
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # 1. Validate auth configuration
    try:
        validate_auth_config()
        logger.info("Auth configuration validated")
    except RuntimeError as e:
        logger.warning(f"Auth not configured: {e}")

    # 2. Initialize Twelve Data client (optional)
    init_twelve_data()

    # 3. Initialize broker (OANDA)
    broker = None
    if settings.OANDA_API_KEY and settings.OANDA_ACCOUNT_ID:
        try:
            broker = create_broker(
                "oanda",
                paper=settings.OANDA_PAPER_TRADING,
                api_key=settings.OANDA_API_KEY,
                account_id=settings.OANDA_ACCOUNT_ID
            )
            logger.info(f"OANDA broker initialized (paper={settings.OANDA_PAPER_TRADING})")
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")

    # 4. Initialize trade engine
    config = EngineConfig.from_settings()
    engine = LiveTradeEngine(config, broker)

    # 5. Initialize scheduler with alert recipients
    alert_recipients = [settings.ALERT_EMAIL] if settings.ALERT_EMAIL else []
    init_scheduler(engine, alert_recipients)

    scheduler_instance = get_scheduler()
    if scheduler_instance:
        scheduler_instance.start()
        logger.info("Scheduler started (hourly scans, daily reset, balance sync)")

    # 6. Set global instances for routers
    from app.routers.circuit_breaker import set_circuit_breaker
    from app.routers.correlation import set_correlation_filter
    from app.routers.trade_engine import set_trade_engine

    set_circuit_breaker(engine.circuit_breaker)
    set_correlation_filter(engine.correlation_filter)
    set_trade_engine(engine)

    logger.info("=" * 60)
    logger.info("PriceIQ Pro is fully operational!")
    logger.info(f"Dashboard: http://localhost:8000/docs")
    logger.info(f"Email alerts: {'ENABLED' if settings.ALERT_EMAIL else 'DISABLED'}")
    logger.info(f"Auto-execute: {'ENABLED' if settings.AUTO_EXECUTE_TRADES else 'DISABLED'}")
    logger.info("=" * 60)

    yield

    # ========== SHUTDOWN ==========
    logger.info("Shutting down PriceIQ Pro...")

    if scheduler_instance:
        scheduler_instance.stop()

    if engine:
        await engine.close()

    await close_twelve_data()

    logger.info("Goodbye!")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Automated Forex Trading System with Multi-Layer Pattern Validation",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def health():
    from datetime import datetime
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ========== REGISTER ROUTERS ==========

# Core routers
app.include_router(signals.router)
app.include_router(signals_v32.router)
app.include_router(broker_webhook.router)
app.include_router(system.router)

# Feature routers
app.include_router(session.router)
app.include_router(price_validator.router)
app.include_router(multi_timeframe.router)
app.include_router(circuit_breaker.router)
app.include_router(correlation.router)
app.include_router(trade_engine_router.router)
app.include_router(patterns.router)

# Admin routers
app.include_router(auth.router)
app.include_router(database.router)
app.include_router(monte_carlo.router)
app.include_router(scheduler.router)

# ========== ROOT ENDPOINT ==========
@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "status": "/api/status",
            "signals": "/api/signals",
            "signals_v32": "/api/signals/generate-v32",
            "session": "/api/session/status",
            "circuit_breaker": "/api/circuit-breaker/status",
            "correlation": "/api/correlation/status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
