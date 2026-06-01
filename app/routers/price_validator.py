"""
PriceIQ Pro — Price Validator Router v3.3
"""

from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timezone
import logging

from app.services.data_fetcher import data_fetcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/price-validator", tags=["price-validator"])


@router.get("/price")
async def get_live_price(
    pair: str = Query("EURUSD", description="Currency pair"),
):
    """Get current live price for a pair."""
    try:
        # Fetch recent candles to get latest price
        candles = await data_fetcher.get_candles(pair, "1m", limit=1)
        
        if candles and len(candles) > 0:
            latest = candles[-1]
            return {
                "pair": pair,
                "price": latest.close,
                "bid": latest.close - 0.0001,  # Approximate
                "ask": latest.close + 0.0001,  # Approximate
                "timestamp": latest.timestamp.isoformat(),
                "source": "data_fetcher"
            }
        else:
            return {
                "pair": pair,
                "price": None,
                "error": "Could not fetch price",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
        return {
            "pair": pair,
            "price": None,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
