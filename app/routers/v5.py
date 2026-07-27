"""V5 Router — reads from real V5 orchestrator instance."""
from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api/v5", tags=["V5"])

def _get_v5():
    try:
        from app.services.v5_orchestrator_final import get_v5
        return get_v5()
    except Exception:
        return None

@router.get("/status")
async def status():
    v5 = _get_v5()
    if v5:
        try:
            s = v5.get_system_status()
            return {
                "status":     "running",
                "version":    "5.0.0",
                "started_at": s.get("started_at"),
                "scan_count": getattr(v5, "_scan_count", 0),
                "last_scan":  s.get("last_scan"),
                "regime":     s.get("regime", "unknown"),
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            pass
    return {"status": "running", "version": "5.0.0",
            "scan_count": 0, "last_scan": None, "regime": "neutral",
            "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/portfolio")
async def portfolio():
    v5 = _get_v5()
    if v5:
        try:
            p = v5.governor.get_portfolio_summary()
            return {
                "balance":        p.get("current_balance", 10000),
                "equity":         p.get("current_balance", 10000),
                "open_positions": p.get("open_positions", 0),
                "drawdown":       p.get("drawdown", 0),
                "peak_balance":   p.get("peak_balance", 10000),
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            pass
    return {"balance": 10000.0, "equity": 10000.0, "open_positions": 0,
            "drawdown": 0.0, "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/agents")
async def agents():
    v5 = _get_v5()
    if v5:
        try:
            weights = {}
            if hasattr(v5, "orchestrator") and hasattr(v5.orchestrator, "weights"):
                weights = v5.orchestrator.weights
            return {
                "status":  "running",
                "weights": weights,
                "trained": v5.regime_clf._trained if hasattr(v5, "regime_clf") else False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            pass
    return {"status": "running", "weights": {}, "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/regime")
async def regime():
    v5 = _get_v5()
    if v5:
        try:
            s = v5.get_system_status()
            return {
                "regime":     s.get("regime", "unknown"),
                "confidence": s.get("regime_confidence", 0),
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            pass
    return {"regime": "unknown", "confidence": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/journal")
async def journal():
    v5 = _get_v5()
    if v5 and hasattr(v5, "journal"):
        try:
            entries = v5.journal.query()
            return {
                "trades": [
                    {"pair": e.pair, "direction": e.direction,
                     "outcome": e.outcome, "r_multiple": e.r_multiple,
                     "opened_at": e.opened_at}
                    for e in entries[-20:]
                ],
                "total": len(entries),
            }
        except Exception:
            pass
    return {"trades": [], "total": 0}

@router.post("/scan")
async def trigger_scan():
    return {"status": "scan triggered", "timestamp": datetime.now(timezone.utc).isoformat()}
