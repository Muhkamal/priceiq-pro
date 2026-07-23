"""
PriceIQ Pro V5 — Main Entry (Robust)
"""

import sys
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Ensure we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# ============ MOUNT ROUTERS (Safe) ============

# Try to import and mount each router individually
routers_to_try = [
    ("signals", "signals"),
    ("system", "system"),
    ("session", "session"),
    ("circuit_breaker", "circuit_breaker"),
    ("auth", "auth"),
    ("database", "database"),
]

for router_name, module_name in routers_to_try:
    try:
        module = __import__(f"app.routers.{module_name}", fromlist=["router"])
        if hasattr(module, "router"):
            app.include_router(module.router)
            logger.info(f"✅ Router {router_name} mounted")
        else:
            logger.warning(f"⚠️ Router {router_name} has no 'router' attribute")
    except ImportError as e:
        logger.warning(f"⚠️ Router {router_name} not found: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Router {router_name} error: {e}")

# Try to mount V5 router separately
try:
    from app.routers.v5 import router as v5_router
    app.include_router(v5_router)
    logger.info("✅ V5 router mounted")
except ImportError:
    logger.warning("⚠️ V5 router not found - creating basic one")
    # Create a basic V5 router inline
    from fastapi import APIRouter
    v5_basic = APIRouter(prefix="/api/v5", tags=["v5"])
    @v5_basic.get("/status")
    async def v5_status():
        return {"status": "running", "version": "5.0.0"}
    app.include_router(v5_basic)
except Exception as e:
    logger.error(f"❌ V5 router error: {e}")

logger.info("=" * 60)
logger.info("✅ PriceIQ Pro V5 Started")
logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
