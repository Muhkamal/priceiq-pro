"""
PriceIQ Pro — V5 Live Dashboard API v1.0

FastAPI router that exposes the entire V5 system state as REST endpoints.
Mount this in main.py and your React dashboard can poll it.

Endpoints:
    GET  /api/v5/status          — full system health snapshot
    GET  /api/v5/portfolio        — risk governor + VaR report
    GET  /api/v5/positions        — all open managed positions
    GET  /api/v5/agents           — agent weights + regime table
    GET  /api/v5/learning         — win rates, expectancy, thresholds
    GET  /api/v5/journal          — trade journal stats + recent trades
    GET  /api/v5/journal/{id}     — single trade detail
    GET  /api/v5/regime           — current regime + transition probs
    GET  /api/v5/drift            — feature drift status
    GET  /api/v5/calendar         — upcoming economic events
    GET  /api/v5/correlation      — live correlation matrix
    GET  /api/v5/montecarlo       — run Monte Carlo on journal trades
    GET  /api/v5/sensitivity      — last sensitivity test results
    POST /api/v5/scan             — trigger immediate scan
    POST /api/v5/close/{pair}     — force close a position
    POST /api/v5/calendar/refresh — refresh economic calendar
    POST /api/v5/annotate/{id}    — annotate a journal entry

Usage in main.py:
    from app.services.v5_dashboard_api import v5_router
    app.include_router(v5_router)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

logger = logging.getLogger(__name__)

v5_router = APIRouter(prefix="/api/v5", tags=["V5 System"])


# ── Request/Response models ──────────────────────────────────

class AnnotateRequest(BaseModel):
    notes: str = ""
    tags:  List[str] = []


class ForceCloseRequest(BaseModel):
    current_price: float
    reason: str = "Manual close via dashboard"


class ScanRequest(BaseModel):
    pairs: Optional[List[str]] = None
    timeframe: str = "1h"


# ── Helper: get v5 instance ──────────────────────────────────

def _get_v5():
    try:
        from app.services.v5_orchestrator_v2 import get_v5
        v5 = get_v5()
        if not v5:
            raise HTTPException(status_code=503, detail="V5 system not initialised")
        return v5
    except ImportError:
        raise HTTPException(status_code=503, detail="V5 module not found")


def _get_scheduler():
    try:
        from app.services.scheduler import get_scheduler
        return get_scheduler()
    except Exception:
        return None


# ── Endpoints ────────────────────────────────────────────────

@v5_router.get("/status")
async def get_status() -> Dict:
    """Full V5 system health snapshot."""
    v5 = _get_v5()
    try:
        status = v5.get_system_status()
        status["timestamp"] = datetime.now(timezone.utc).isoformat()
        status["scheduler"] = _get_scheduler().get_status() if _get_scheduler() else None
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/portfolio")
async def get_portfolio() -> Dict:
    """Risk governor portfolio state + VaR report."""
    v5 = _get_v5()
    try:
        portfolio = v5.governor.get_portfolio_summary()
        # VaR report
        if hasattr(v5, "var_engine"):
            var_report = v5.var_engine.compute(v5.governor._open_positions)
            portfolio["var"] = v5.var_engine.to_dict(var_report)
            portfolio["stress_test"] = v5.var_engine.stress_test(v5.governor._open_positions)
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/positions")
async def get_positions() -> Dict:
    """All open managed positions with management state."""
    v5 = _get_v5()
    try:
        if hasattr(v5, "trade_manager"):
            return {
                "positions": v5.trade_manager.get_open_positions(),
                "stats":     v5.trade_manager.get_stats(),
            }
        return {"positions": {}, "stats": {}, "note": "TradeManager not attached"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/agents")
async def get_agents() -> Dict:
    """Agent weights, regime-conditional table, and best agent per regime."""
    v5 = _get_v5()
    try:
        return {
            "global_weights":      v5.learning.get_agent_weights(),
            "regime_table":        v5.regime_learner.get_full_table(),
            "best_per_regime":     v5.regime_learner.get_best_agent_per_regime(),
            "regime_summary":      v5.regime_learner.regime_fit_summary(),
            "win_prob_samples":    v5.win_prob_cal.sample_sizes(),
            "win_prob_calibration": v5.win_prob_cal.calibration_table(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/learning")
async def get_learning() -> Dict:
    """Learning loop stats: per-pair win rates, thresholds, expectancy."""
    v5 = _get_v5()
    try:
        return v5.learning.get_full_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/journal")
async def get_journal(last_n: int = 50) -> Dict:
    """Trade journal summary + recent entries."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "journal"):
            return {"note": "TradeJournal not attached", "stats": {}}
        stats   = v5.journal.full_stats()
        recent  = v5.journal.query(last_n=last_n)
        return {
            "stats":   stats,
            "recent":  [
                {
                    "trade_id":   e.trade_id,
                    "pair":       e.pair,
                    "direction":  e.direction,
                    "agent":      e.agent,
                    "regime":     e.regime,
                    "outcome":    e.outcome,
                    "r_multiple": e.r_multiple,
                    "pnl_usd":    e.pnl_usd,
                    "confidence": e.confidence,
                    "session":    e.session,
                    "opened_at":  e.opened_at,
                    "closed_at":  e.closed_at,
                    "notes":      e.notes,
                    "tags":       e.tags,
                }
                for e in reversed(recent)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/journal/{trade_id}")
async def get_journal_entry(trade_id: str) -> Dict:
    """Single trade journal entry with full detail."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "journal"):
            raise HTTPException(status_code=404, detail="TradeJournal not attached")
        entries = v5.journal.query()
        entry   = next((e for e in entries if e.trade_id == trade_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        from dataclasses import asdict
        return asdict(entry)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/regime")
async def get_regime() -> Dict:
    """Current regime classification + transition probabilities."""
    v5 = _get_v5()
    try:
        result = {"drift_status": v5.drift_monitor.status_dict()}
        if hasattr(v5, "regime_transition"):
            streak_r, streak_n = v5.regime_transition.regime_streak()
            result.update({
                "transition_matrix": v5.regime_transition.get_transition_matrix(),
                "current_streak":    {"regime": streak_r, "bars": streak_n},
                "shift_risk":        v5.regime_transition.shift_risk(streak_r),
                "volatile_surge_prob": v5.regime_transition.volatility_surge_prob(streak_r),
                "size_multiplier":   v5.regime_transition.size_multiplier(streak_r),
                "summary":           v5.regime_transition.summary(),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/drift")
async def get_drift() -> Dict:
    """Feature drift monitor status."""
    v5 = _get_v5()
    try:
        report = v5.drift_monitor.get_last_report()
        return {
            "status": v5.drift_monitor.status_dict(),
            "last_report": {
                "max_psi":        report.max_psi,
                "severe":         report.severe,
                "warn":           report.warn,
                "drifted":        report.drifted_features,
                "recommendation": report.recommendation,
                "timestamp":      report.timestamp,
            } if report else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/calendar")
async def get_calendar(hours_ahead: int = 24) -> Dict:
    """Upcoming economic events and current blackout status."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "calendar"):
            return {"note": "EconomicCalendar not attached"}
        events  = v5.calendar.upcoming_events(hours_ahead=hours_ahead)
        status  = v5.calendar.status()
        return {
            "status": status,
            "upcoming": [
                {
                    "title":    e.title,
                    "currency": e.currency,
                    "impact":   e.impact,
                    "dt_utc":   e.dt_utc.isoformat(),
                    "forecast": e.forecast,
                    "actual":   e.actual,
                }
                for e in events
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/correlation")
async def get_correlation() -> Dict:
    """Live dynamic correlation matrix."""
    v5 = _get_v5()
    try:
        watchlist = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
        return {
            "matrix":  v5.corr_estimator.get_correlation_matrix(watchlist),
            "alerts":  v5.corr_estimator.get_high_correlation_alerts(watchlist),
            "status":  v5.corr_estimator.status_dict(watchlist),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.get("/montecarlo")
async def run_montecarlo(n_paths: int = 1000) -> Dict:
    """Run Monte Carlo simulation on all journal trades."""
    v5 = _get_v5()
    try:
        from app.services.research.sensitivity_and_montecarlo import MonteCarloEquityCurve
        all_records = v5.learning.store.all()
        r_mults     = [r.r_multiple for r in all_records]
        if len(r_mults) < 5:
            return {"error": f"Need ≥ 5 trades. Have {len(r_mults)}."}
        mc      = MonteCarloEquityCurve(r_mults)
        report  = mc.run(n_paths=n_paths, starting_balance=v5.governor.starting_balance)
        summary = mc.summary(report)
        return {
            "n_paths":          report.n_paths,
            "n_trades":         report.n_trades,
            "final_equity":     report.final_equity,
            "max_drawdown_dist": report.max_drawdown,
            "prob_profit":      report.prob_profit,
            "prob_ruin":        report.prob_ruin,
            "consecutive_loss_dist": report.consecutive_loss_dist,
            "recommendation":   report.recommendation,
            "summary":          summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Action endpoints ─────────────────────────────────────────

@v5_router.post("/scan")
async def trigger_scan(request: ScanRequest = Body(default=ScanRequest())) -> Dict:
    """Trigger an immediate signal scan."""
    scheduler = _get_scheduler()
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    try:
        await scheduler._run_v5_hourly_scan()
        return {"status": "scan triggered", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.post("/close/{pair}")
async def force_close_position(pair: str, request: ForceCloseRequest) -> Dict:
    """Force close an open position."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "trade_manager"):
            raise HTTPException(status_code=404, detail="TradeManager not attached")
        await v5.trade_manager.force_close(pair, request.current_price, request.reason)
        return {"status": "closed", "pair": pair, "price": request.current_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.post("/calendar/refresh")
async def refresh_calendar() -> Dict:
    """Force-refresh the economic calendar."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "calendar"):
            raise HTTPException(status_code=404, detail="EconomicCalendar not attached")
        await v5.calendar.refresh()
        return {"status": "refreshed", "calendar": v5.calendar.status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v5_router.post("/annotate/{trade_id}")
async def annotate_trade(trade_id: str, request: AnnotateRequest) -> Dict:
    """Add notes and tags to a journal entry."""
    v5 = _get_v5()
    try:
        if not hasattr(v5, "journal"):
            raise HTTPException(status_code=404, detail="TradeJournal not attached")
        v5.journal.annotate(trade_id, request.notes, request.tags)
        return {"status": "annotated", "trade_id": trade_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
