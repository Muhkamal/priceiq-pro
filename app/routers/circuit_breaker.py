"""
PriceIQ Pro — Circuit Breaker Router v3.3
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/circuit-breaker", tags=["circuit-breaker"])

# Global circuit breaker instance
_circuit_breaker = None

def set_circuit_breaker(cb):
    global _circuit_breaker
    _circuit_breaker = cb

def get_circuit_breaker():
    return _circuit_breaker


@router.get("/status")
async def get_breaker_status():
    if _circuit_breaker is None:
        return {"status": "not_initialized"}
    result = _circuit_breaker.check()
    return {
        "is_allowed": result.is_allowed,
        "state": result.state.value,
        "reason": result.reason,
        "stats": result.stats
    }


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
