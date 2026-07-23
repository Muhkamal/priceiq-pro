"""
PriceIQ Pro V5.4 — main_advanced.py

Complete integration of all four advanced capabilities:
    ✅ Tick-level execution (bar close detection, sub-second order routing)
    ✅ Cross-asset macro signals (TIPS yields, DXY, VIX → 5th agent)
    ✅ Walk-forward optimization (rolling parameter re-optimization)
    ✅ Extended orchestrator (macro agreement/disagreement adjustments)

New environment variables:
    TICK_ENGINE_ENABLED=true    → enable tick-level execution
    SIMULATE_TICKS=true         → use simulated ticks (no OANDA needed)
    MACRO_AGENT_ENABLED=true    → enable macro signal agent (default: true)
    WFO_ENABLED=true            → enable walk-forward optimization
    FRED_API_KEY=...            → free at fred.stlouisfed.org (for real rate data)

Usage:
    Replace main_final.py with this file.
    All previous V5.3 capabilities are preserved + new ones added.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import os
from datetime import datetime, timezone

from app.services.core.v5_settings import v5_settings
from app.services.v5_orchestrator_final        import init_v5, get_v5
from app.services.core.preflight               import PreflightValidator
from app.services.core.shutdown_handler        import ShutdownHandler
from app.services.core.data_quality_validator  import DataQualityValidator
from app.services.core.mtf_confluence          import MultiTimeframeConfluence
from app.services.core.paper_trading           import PaperTradingEngine
from app.services.core.background_task_cache   import BackgroundTaskCache, make_cached_v5_router
from app.services.risk.overnight_gap_manager   import OvernightGapManager
from app.services.risk.hybrid_correlation      import HybridCorrelationEstimator
from app.services.risk.var_engine_v2           import VaREngine
from app.services.ml.emergency_retrain         import EmergencyRetrainTrigger
from app.services.monitoring.equity_curve_tracker   import EquityCurveTracker
from app.services.monitoring.telegram_commander     import TelegramCommander
from app.services.research.performance_attribution  import PerformanceAttribution

# ── NEW: Advanced capabilities ────────────────────────────────
from app.services.advanced_integration import (
    attach_advanced_capabilities,
    AdvancedSchedulerJobs,
    make_advanced_router,
)

from app.services.scheduler_final         import init_scheduler, get_scheduler
from app.services.trade_engine            import LiveTradeEngine, EngineConfig
from app.services.broker_connector        import create_broker
from app.services.auth                    import validate_auth_config
from app.services.twelve_data_client      import init_twelve_data, close_twelve_data
from app.services.telegram_bot            import telegram

from app.routers import (signals, signals_v32, broker_webhook, system,
                         session, price_validator, multi_timeframe,
                         circuit_breaker, correlation,
                         trade_engine as trade_engine_router,
                         auth, database, monte_carlo, patterns)
from app.routers import scheduler as scheduler_router
from app.services.api.v5_dashboard_api    import v5_router
from app.services.api.broker_webhook_v5   import broker_webhook_router
from app.services.api.websocket_dashboard import ws_router, ws_broadcaster

DRY_RUN            = os.environ.get("DRY_RUN_MODE",          "false").lower() == "true"
TICK_ENGINE_ENABLED = os.environ.get("TICK_ENGINE_ENABLED",  "false").lower() == "true"
MACRO_ENABLED       = os.environ.get("MACRO_AGENT_ENABLED",  "true").lower()  == "true"
WFO_ENABLED         = os.environ.get("WFO_ENABLED",          "false").lower() == "true"
FRED_API_KEY        = os.environ.get("FRED_API_KEY",          "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # ═══════════════════════════════════════════════════════
    # STARTUP
    # ═══════════════════════════════════════════════════════

    mode_tags = []
    if DRY_RUN:            mode_tags.append("DRY_RUN")
    if TICK_ENGINE_ENABLED: mode_tags.append("TICK_ENGINE")
    if MACRO_ENABLED:       mode_tags.append("MACRO_AGENT")
    if WFO_ENABLED:         mode_tags.append("WFO")
    if FRED_API_KEY:        mode_tags.append("FRED_API")

    logger.info("=" * 60)
    logger.info(f"PriceIQ Pro V5.4 — {' + '.join(mode_tags) or 'LIVE'}")
    logger.info("=" * 60)

    # 1. Settings
    v5_settings.log_summary()
    config_errors = v5_settings.validate()
    if config_errors:
        raise SystemExit(f"Settings invalid: {config_errors}")

    # 2. Preflight
    report = await PreflightValidator().run()
    if not report.passed:
        raise SystemExit("Preflight failed")

    # 3. Clients
    try:
        validate_auth_config()
    except RuntimeError as e:
        logger.warning(f"Auth: {e}")
    init_twelve_data()

    # 4. Broker
    broker = None
    if v5_settings.OANDA_API_KEY and v5_settings.OANDA_ACCOUNT_ID:
        try:
            broker = create_broker(
                "oanda", paper=v5_settings.OANDA_PAPER_TRADING,
                api_key=v5_settings.OANDA_API_KEY,
                account_id=v5_settings.OANDA_ACCOUNT_ID,
            )
            logger.info(f"Broker: OANDA (paper={v5_settings.OANDA_PAPER_TRADING})")
        except Exception as e:
            logger.error(f"Broker init failed: {e}")

    # 5. Trade engine
    config = EngineConfig.from_settings()
    engine = LiveTradeEngine(config, broker)

    # 6. V5 orchestrator (base system)
    v5 = init_v5(
        starting_balance=v5_settings.ACCOUNT_BALANCE,
        risk_pct=v5_settings.RISK_PERCENT,
        target_vol_pct=v5_settings.TARGET_VOL_PCT,
        max_drawdown=v5_settings.MAX_DRAWDOWN_PCT,
        telegram=telegram,
        regime_model_path=v5_settings.REGIME_MODEL_PATH,
        learning_state_path=v5_settings.LEARNING_STATE_PATH,
        regime_weights_path=v5_settings.REGIME_WEIGHTS_PATH,
        win_prob_path=v5_settings.WIN_PROB_PATH,
        transition_path=v5_settings.TRANSITION_PATH,
        journal_path=v5_settings.JOURNAL_PATH,
    )

    # 7. Attach V5.3 supplementary components
    v5.mtf_confluence    = MultiTimeframeConfluence()
    v5.gap_manager       = OvernightGapManager(telegram=telegram)
    v5.data_validator    = DataQualityValidator()
    v5.equity_tracker    = EquityCurveTracker(
        starting_balance=v5_settings.ACCOUNT_BALANCE,
        path=v5_settings.EQUITY_CURVE_PATH,
    )
    v5.attribution       = PerformanceAttribution()
    v5.emergency_retrain = EmergencyRetrainTrigger(
        classifier=v5.regime_clf, augmenter=v5.augmenter,
        drift_monitor=v5.drift_monitor, telegram=telegram,
        model_path=v5_settings.REGIME_MODEL_PATH,
    )
    v5.corr_estimator    = HybridCorrelationEstimator(
        window=v5_settings.CORR_WINDOW_NORMAL,
        threshold=v5_settings.CORR_THRESHOLD,
    )
    v5.var_engine        = VaREngine(
        account_balance=v5_settings.ACCOUNT_BALANCE,
        max_heat_pct=v5_settings.VAR_MAX_HEAT_PCT,
        critical_heat_pct=v5_settings.VAR_CRITICAL_HEAT_PCT,
        max_open_positions=v5_settings.MAX_OPEN_POSITIONS,
    )
    v5.paper_trader      = PaperTradingEngine(
        starting_virtual_balance=v5_settings.ACCOUNT_BALANCE,
        risk_pct=v5_settings.RISK_PERCENT,
        telegram=telegram, learning_loop=v5.learning, journal=v5.journal,
    )
    if hasattr(v5, "trade_manager"):
        v5.trade_manager.journal = v5.journal
    v5._signals_paused   = False
    v5._last_wfo_report  = None

    # 8. ── NEW: Attach advanced capabilities ─────────────────
    attach_advanced_capabilities(
        v5=v5, broker=broker, telegram=telegram,
        watchlist=v5_settings.WATCHLIST,
        timeframes=["1h"],
    )

    # 9. Macro initial data fetch
    if MACRO_ENABLED and hasattr(v5, "macro_agent"):
        try:
            snap = await v5.macro_agent.refresh_data()
            if snap:
                logger.info(
                    f"Macro snapshot loaded: gold_bias={snap.gold_macro_bias} "
                    f"real_rate={snap.us_real_rate}% DXY={snap.dxy_level}"
                )
            elif not FRED_API_KEY:
                logger.warning(
                    "Macro agent: no FRED_API_KEY set. "
                    "DXY/VIX data available via Yahoo Finance. "
                    "Set FRED_API_KEY=<free key> at fred.stlouisfed.org for real rate data."
                )
        except Exception as e:
            logger.warning(f"Macro initial fetch failed: {e}")

    # 10. Position snapshot (post-restart reconciliation)
    snapshot = ShutdownHandler.load_position_snapshot()
    if snapshot and snapshot.get("positions"):
        n = len(snapshot["positions"])
        logger.warning(f"Found {n} open position(s) from last session")
        await telegram.send_message(
            f"⚠️ <b>Restart detected</b>\n"
            f"{n} position(s) open at last shutdown.\n"
            f"Verify against broker before trading."
        )

    # 11. Shutdown handler
    shutdown          = ShutdownHandler(v5=v5, telegram=telegram)
    shutdown.register()
    app.state.shutdown = shutdown

    # 12. Economic calendar
    try:
        await v5.calendar.refresh()
    except Exception as e:
        logger.warning(f"Calendar refresh failed: {e}")

    # 13. Router dependencies
    from app.routers.circuit_breaker import set_circuit_breaker
    from app.routers.correlation     import set_correlation_filter
    from app.routers.trade_engine    import set_trade_engine
    set_circuit_breaker(engine.circuit_breaker)
    set_correlation_filter(engine.correlation_filter)
    set_trade_engine(engine)

    # 14. Scheduler (fully wired including advanced jobs)
    alert_recipients = [e for e in [os.environ.get("ALERT_EMAIL", "")] if e]
    init_scheduler(engine, alert_recipients)
    scheduler = get_scheduler()
    if scheduler:
        scheduler.start()

        # ── NEW: Register advanced scheduler jobs ─────────────
        adv_jobs = AdvancedSchedulerJobs(v5, scheduler)
        adv_jobs.register()
        logger.info("Advanced scheduler jobs registered")

    # 15. Telegram commander
    commander = TelegramCommander()
    asyncio.create_task(commander.start_polling())
    app.state.commander = commander

    # 16. Startup notification
    try:
        caps_str = ", ".join(filter(None, [
            "TICK" if TICK_ENGINE_ENABLED else "",
            "MACRO" if MACRO_ENABLED else "",
            "WFO" if WFO_ENABLED else "",
            "FRED" if FRED_API_KEY else "",
            "DRY_RUN" if DRY_RUN else "",
        ]))

        snap    = getattr(v5, "macro_agent", None)
        snap    = snap.get_snapshot() if snap else None
        macro_str = (f"\nMacro: gold_bias={snap.gold_macro_bias} "
                     f"real_rate={snap.us_real_rate}% DXY={snap.dxy_level}") if snap else ""

        await telegram.send_startup_message()
        await telegram.send_message(
            f"🟢 <b>PriceIQ Pro V5.4</b>\n"
            f"Mode: {caps_str or 'STANDARD'}\n"
            f"Balance: ${v5.governor.current_balance:,.2f}\n"
            f"Watchlist: {', '.join(v5_settings.WATCHLIST)}\n"
            f"Model: {'trained ✅' if v5.regime_clf._trained else 'heuristic'}\n"
            f"Agents: 4 + {'macro ✅' if MACRO_ENABLED else 'macro ❌'}"
            f"{macro_str}"
        )
        await ws_broadcaster.heartbeat(v5.governor.get_portfolio_summary())
    except Exception as e:
        logger.warning(f"Startup notifications failed: {e}")

    logger.info("=" * 60)
    logger.info("PriceIQ Pro V5.4 — Fully operational")
    logger.info("=" * 60)

    yield

    # ═══════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════

    logger.info("Shutting down PriceIQ Pro V5.4...")

    # Stop tick engine if running
    if hasattr(v5, "tick_engine"):
        v5.tick_engine.stop()

    if hasattr(app.state, "commander"):
        app.state.commander.stop()
    if scheduler:
        scheduler.stop()
    if engine:
        await engine.close()
    await close_twelve_data()
    if hasattr(app.state, "shutdown"):
        await app.state.shutdown.execute()

    logger.info("Goodbye.")


# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title="PriceIQ Pro",
    description="Autonomous AI Forex Trading System V5.4",
    version="5.4",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "version": "5.4",
            "timestamp": datetime.utcnow().isoformat()}


# ── Routers ───────────────────────────────────────────────────
app.include_router(v5_router)
app.include_router(broker_webhook_router)
app.include_router(ws_router)

# ── NEW: Advanced capabilities router ────────────────────────
@app.on_event("startup")
async def _mount_advanced_router():
    """Mount advanced router after v5 is initialised."""
    v5 = get_v5()
    if v5:
        adv_router = make_advanced_router(v5)
        app.include_router(adv_router)

# Existing
app.include_router(signals.router)
app.include_router(signals_v32.router)
app.include_router(broker_webhook.router)
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


@app.get("/")
async def root():
    v5 = get_v5()
    s  = v5.get_system_status() if v5 else {}
    p  = s.get("portfolio", {})
    macro_status = {}
    if v5 and hasattr(v5, "macro_agent"):
        macro_status = v5.macro_agent.status_dict()

    return {
        "app":     "PriceIQ Pro",
        "version": "5.4",
        "mode":    "dry_run" if DRY_RUN else "live",
        "status":  "running",
        "docs":    "/docs",
        "ws":      "/api/v5/ws",
        "capabilities": {
            "tick_engine":  TICK_ENGINE_ENABLED,
            "macro_agent":  MACRO_ENABLED,
            "wfo":          WFO_ENABLED,
            "fred_api":     bool(FRED_API_KEY),
        },
        "v5": {
            "balance":    p.get("current_balance"),
            "drawdown":   p.get("drawdown"),
            "trading":    p.get("trading_allowed"),
            "open_pos":   p.get("open_positions"),
        },
        "macro": macro_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
