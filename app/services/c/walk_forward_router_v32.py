"""
PriceIQ Pro — Walk-Forward Router v3.2
API endpoint for running walk-forward validation.
"""

from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timezone
import logging

from app.models.schemas import BacktestRequest
from app.services.walk_forward import WalkForwardValidator
from app.services.data_fetcher import data_fetcher

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/walk-forward", tags=["walk-forward"])


@router.post("/run")
async def run_walk_forward(
    request: BacktestRequest,
    n_periods: int = Query(5, ge=2, le=20, description="Number of walk-forward periods"),
    is_ratio: float = Query(0.70, ge=0.50, le=0.90, description="In-sample ratio per period"),
    simulation: bool = Query(False, description="Use simulation mode if backtest not implemented"),
):
    """
    Run walk-forward validation on historical data.

    Returns robustness score, per-period breakdown, and trading verdict.
    """
    try:
        # Fetch candles
        candles = await data_fetcher.get_candles(
            request.pair, 
            request.timeframe, 
            limit=request.bars
        )

        if len(candles) < 500:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 500 bars, got {len(candles)}"
            )

        # Run validation
        validator = WalkForwardValidator(simulation_mode=simulation)
        results = validator.run(
            candles=candles,
            request=request,
            n_periods=n_periods,
            is_ratio=is_ratio,
        )

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Walk-forward error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def walk_forward_health():
    """Check walk-forward system status."""
    return {
        "status": "healthy",
        "version": "3.2",
        "features": {
            "walk_forward": True,
            "simulation_mode": True,
            "async_support": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
