"""
PriceIQ Pro — Advanced Integration v2.0

All 7 agents + RL Trade Manager wired.
"""

from __future__ import annotations
import asyncio, logging, os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MACRO_AGENT_ENABLED   = os.environ.get("MACRO_AGENT_ENABLED",   "true").lower()  == "true"
NEWS_AGENT_ENABLED    = os.environ.get("NEWS_AGENT_ENABLED",    "true").lower()  == "true"
COT_AGENT_ENABLED     = os.environ.get("COT_AGENT_ENABLED",     "true").lower()  == "true"
RL_MANAGER_ENABLED    = os.environ.get("RL_MANAGER_ENABLED",    "false").lower() == "true"
TICK_ENGINE_ENABLED   = os.environ.get("TICK_ENGINE_ENABLED",   "false").lower() == "true"
WFO_ENABLED           = os.environ.get("WFO_ENABLED",           "false").lower() == "true"


def attach_advanced_capabilities(v5, broker=None, telegram=None,
                                  watchlist=None, timeframes=None):
    watchlist  = watchlist  or ["XAUUSD","EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD"]
    timeframes = timeframes or ["1h"]
    caps       = []

    if MACRO_AGENT_ENABLED:
        try:
            from app.services.agents.extended_orchestrator import ExtendedAgentOrchestrator
            ext = ExtendedAgentOrchestrator()
            if hasattr(v5,"orchestrator") and hasattr(v5.orchestrator,"weights"):
                for k,w in v5.orchestrator.weights.items():
                    if k in ext.weights: ext.weights[k] = w
            v5.orchestrator = ext
            v5.macro_agent  = ext.macro_agent
            caps.append("MacroAgent(5)")
        except Exception as e:
            logger.warning(f"MacroAgent: {e}")

    if NEWS_AGENT_ENABLED:
        try:
            from app.services.agents.news_sentiment_agent import NewsSentimentAgent
            v5.news_agent = NewsSentimentAgent()
            if hasattr(v5,"orchestrator") and hasattr(v5.orchestrator,"weights"):
                v5.orchestrator.weights["NewsSentimentAgent"] = 1.0
            caps.append("NewsSentimentAgent(6)")
        except Exception as e:
            logger.warning(f"NewsSentimentAgent: {e}")

    if COT_AGENT_ENABLED:
        try:
            from app.services.agents.cot_report_agent import COTReportAgent
            v5.cot_agent = COTReportAgent()
            if hasattr(v5,"orchestrator") and hasattr(v5.orchestrator,"weights"):
                v5.orchestrator.weights["COTReportAgent"] = 1.0
            caps.append("COTReportAgent(7)")
        except Exception as e:
            logger.warning(f"COTReportAgent: {e}")

    if RL_MANAGER_ENABLED:
        try:
            from app.services.agents.rl_trade_manager import RLTradeManager
            rl = RLTradeManager(journal=getattr(v5,"journal",None),
                                telegram=telegram,
                                learning_loop=getattr(v5,"learning",None))
            if hasattr(v5,"journal"):
                result = rl.train(v5.journal)
                logger.info(f"RL trained: {result.get('n_trajectories',0)} trajs")
            v5.rl_trade_manager = rl
            v5.use_rl_manager   = True
            caps.append("RLTradeManager")
        except Exception as e:
            logger.warning(f"RLTradeManager: {e}")

    if TICK_ENGINE_ENABLED:
        try:
            from app.services.execution.tick_execution_engine import TickExecutionEngine
            paper    = os.environ.get("OANDA_PAPER","true").lower() == "true"
            simulate = os.environ.get("SIMULATE_TICKS","false").lower() == "true"
            v5.tick_engine = TickExecutionEngine(v5=v5, broker=broker,
                                                  telegram=telegram,
                                                  paper=paper, simulate=simulate)
            caps.append(f"TickEngine({'sim' if simulate else 'live'})")
        except Exception as e:
            logger.warning(f"TickEngine: {e}")

    if WFO_ENABLED:
        try:
            from app.services.research.walk_forward_optimizer import WalkForwardOptimizer
            def _bt(c,p): return v5.backtest_engine.run(candles=c, pair="EURUSD",
                                                         starting_balance=10_000.0, risk_pct=2.0)
            v5.wfo = WalkForwardOptimizer(backtest_fn=_bt, max_combos=30)
            caps.append("WFO")
        except Exception as e:
            logger.warning(f"WFO: {e}")

    logger.info(f"Advanced capabilities: {', '.join(caps) if caps else 'none'}")


class AdvancedSchedulerJobs:
    def __init__(self, v5, scheduler_instance):
        self.v5 = v5
        self.scheduler = scheduler_instance

    def register(self):
        try:
            from apscheduler.triggers.cron     import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
            s = self.scheduler._scheduler

            if MACRO_AGENT_ENABLED and hasattr(self.v5,"macro_agent"):
                s.add_job(self._refresh_macro, IntervalTrigger(hours=1),
                          id="macro_refresh", replace_existing=True)

            if NEWS_AGENT_ENABLED and hasattr(self.v5,"news_agent"):
                s.add_job(self._refresh_news, IntervalTrigger(minutes=30),
                          id="news_refresh", replace_existing=True)

            if COT_AGENT_ENABLED and hasattr(self.v5,"cot_agent"):
                s.add_job(self._refresh_cot,
                          CronTrigger(day_of_week="fri", hour=20, minute=30),
                          id="cot_refresh", replace_existing=True)

            if RL_MANAGER_ENABLED and hasattr(self.v5,"rl_trade_manager"):
                s.add_job(self._retrain_rl,
                          CronTrigger(day_of_week="sun", hour=4, minute=0),
                          id="rl_retrain", replace_existing=True)

            if WFO_ENABLED and hasattr(self.v5,"wfo"):
                s.add_job(self._run_wfo,
                          CronTrigger(day_of_week="sun", hour=3, minute=0),
                          id="wfo_weekly", replace_existing=True)

            if TICK_ENGINE_ENABLED and hasattr(self.v5,"tick_engine"):
                watchlist = os.environ.get("WATCHLIST",
                    "XAUUSD,EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD").split(",")
                asyncio.create_task(
                    self.v5.tick_engine.start(pairs=watchlist, timeframes=["1h"])
                )
        except Exception as e:
            logger.warning(f"Advanced scheduler registration: {e}")

    async def _refresh_macro(self):
        try:
            if hasattr(self.v5,"macro_agent"):
                snap = await self.v5.macro_agent.refresh_data()
                if snap: logger.info(f"Macro: {snap.gold_macro_bias} real_rate={snap.us_real_rate}%")
        except Exception as e: logger.error(f"Macro refresh: {e}")

    async def _refresh_news(self):
        try:
            if hasattr(self.v5,"news_agent"):
                await self.v5.news_agent.refresh()
                n = self.v5.news_agent.status_dict().get("n_articles",0)
                logger.info(f"News: {n} articles")
        except Exception as e: logger.error(f"News refresh: {e}")

    async def _refresh_cot(self):
        try:
            if hasattr(self.v5,"cot_agent"):
                snaps = await self.v5.cot_agent.refresh()
                logger.info(f"COT: {len(snaps)} instruments")
        except Exception as e: logger.error(f"COT refresh: {e}")

    async def _retrain_rl(self):
        try:
            if hasattr(self.v5,"rl_trade_manager"):
                result = await asyncio.to_thread(self.v5.rl_trade_manager.train, self.v5.journal)
                logger.info(f"RL retrained: {result}")
        except Exception as e: logger.error(f"RL retrain: {e}")

    async def _run_wfo(self):
        try:
            if not hasattr(self.v5,"wfo"): return
            for pair in ["EURUSD","XAUUSD"]:
                try:
                    candles = await self.scheduler._get_candles(pair,"1h",1000)
                    if len(candles) < 300: continue
                    report = await asyncio.to_thread(self.v5.wfo.run, candles, pair)
                    self.v5._last_wfo_report = report
                except Exception as e: logger.error(f"WFO {pair}: {e}")
        except Exception as e: logger.error(f"WFO job: {e}")


def make_advanced_router(v5):
    from fastapi import APIRouter, HTTPException
    router = APIRouter(prefix="/api/v5/advanced", tags=["V5 Advanced"])

    @router.get("/agents")
    async def all_agents_status():
        out = {}
        if hasattr(v5,"news_agent"):  out["news"]  = v5.news_agent.status_dict()
        if hasattr(v5,"cot_agent"):   out["cot"]   = v5.cot_agent.status_dict()
        if hasattr(v5,"macro_agent"): out["macro"] = v5.macro_agent.status_dict()
        if hasattr(v5,"rl_trade_manager"): out["rl"] = v5.rl_trade_manager.get_stats()
        return out

    @router.get("/news")
    async def news_status():
        if not hasattr(v5,"news_agent"): return {"status":"disabled"}
        return v5.news_agent.status_dict()

    @router.post("/news/refresh")
    async def news_refresh():
        if not hasattr(v5,"news_agent"): raise HTTPException(404,"Not enabled")
        await v5.news_agent.refresh()
        return {"refreshed": True}

    @router.get("/cot")
    async def cot_status():
        if not hasattr(v5,"cot_agent"): return {"status":"disabled"}
        return v5.cot_agent.status_dict()

    @router.post("/cot/refresh")
    async def cot_refresh():
        if not hasattr(v5,"cot_agent"): raise HTTPException(404,"Not enabled")
        snaps = await v5.cot_agent.refresh()
        return {"refreshed": True, "instruments": list(snaps.keys())}

    @router.get("/macro")
    async def macro_status():
        if not hasattr(v5,"macro_agent"): return {"status":"disabled"}
        snap = v5.macro_agent.get_snapshot()
        return {"status": v5.macro_agent.status_dict(),
                "snapshot": snap.__dict__ if snap else None}

    @router.post("/macro/refresh")
    async def macro_refresh():
        if not hasattr(v5,"macro_agent"): raise HTTPException(404,"Not enabled")
        snap = await v5.macro_agent.refresh_data()
        return {"refreshed": snap is not None, "snapshot": snap.__dict__ if snap else None}

    @router.get("/rl")
    async def rl_status():
        if not hasattr(v5,"rl_trade_manager"): return {"status":"disabled"}
        return v5.rl_trade_manager.get_stats()

    @router.post("/rl/retrain")
    async def rl_retrain():
        if not hasattr(v5,"rl_trade_manager"): raise HTTPException(404,"Not enabled")
        result = await asyncio.to_thread(v5.rl_trade_manager.train, v5.journal)
        return result

    @router.get("/tick-engine")
    async def tick_engine():
        if not hasattr(v5,"tick_engine"): return {"status":"disabled"}
        return v5.tick_engine.get_stats()

    @router.post("/wfo/run")
    async def wfo_run(pair: str = "EURUSD", n_candles: int = 500):
        if not hasattr(v5,"wfo"): raise HTTPException(404,"Not enabled")
        asyncio.create_task(_run_wfo_bg(v5, pair, n_candles))
        return {"status":"queued","pair":pair,"retry_after":60}

    @router.get("/wfo/status")
    async def wfo_status():
        if not hasattr(v5,"wfo"): return {"status":"disabled"}
        r = getattr(v5,"_last_wfo_report",None)
        if not r: return {"status":"no_results_yet"}
        return {"efficiency_ratio":r.efficiency_ratio,"oos_sharpe":r.oos_sharpe,
                "recommendation":r.recommendation,"timestamp":r.timestamp}

    return router


async def _run_wfo_bg(v5, pair, n_candles):
    try:
        from app.services.scheduler_final import get_scheduler
        sched   = get_scheduler()
        candles = await sched._get_candles(pair,"1h",n_candles) if sched else []
        if len(candles) < 300: return
        report  = await asyncio.to_thread(v5.wfo.run, candles, pair)
        v5._last_wfo_report = report
    except Exception as e:
        logger.error(f"WFO bg: {e}")
