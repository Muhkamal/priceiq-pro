"""
PriceIQ Pro — Live Trade Engine v3.3
Updated for v3.2 corrected engine compatibility.

FIXES:
  1. Correct import paths (market_analyzer_advanced)
  2. MultiTimeframeAnalyzer replaced with v3.2 native H4 gate
  3. Confidence threshold configurable, defaults to 0.45
  4. Broker health check before execution
  5. Position size validation
  6. Bounded decision history
  7. Async cleanup on shutdown
  8. Daily counters for O(1) status
  9. Thread-safe decision history and daily counters
  10. Fixed H4 direction mismatch (always checked as "buy")
  11. Fixed balance divide-by-zero in record_trade_outcome
  12. Removed emoji from logs
"""

import asyncio
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from collections import deque
from pydantic import Field
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


class EngineConfig(BaseModel):
    account_balance:        float = 10_000.0
    risk_percent:           float = 2.0
    daily_loss_limit_pct:   float = 5.0
    weekly_loss_limit_pct:  float = 10.0
    max_consecutive_losses: int   = 5
    max_drawdown_pct:       float = 15.0
    max_lots_per_currency:  float = 3.0
    use_news_filter:        bool  = True
    use_session_filter:     bool  = True
    use_circuit_breaker:    bool  = True
    use_correlation_filter: bool  = True
    use_multi_timeframe:    bool  = True
    block_medium_news:      bool  = False
    strict_session:         bool  = True
    auto_execute:           bool  = False
    paper_trading:          bool  = True
    broker_type:            str   = "oanda"
    watchlist:              List[str] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    timeframe:              str   = "1h"
    confidence_threshold:   float = 0.45
    max_decision_history:   int   = 10_000

    @validator("confidence_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    @classmethod
    def from_settings(cls):
        """Create config from application settings."""
        from app.core.config import settings
        return cls(
            account_balance=settings.DEFAULT_ACCOUNT_BALANCE,
            risk_percent=settings.DEFAULT_RISK_PERCENT,
            watchlist=settings.DEFAULT_WATCHLIST,
            paper_trading=settings.OANDA_PAPER_TRADING,
            auto_execute=settings.AUTO_EXECUTE_TRADES,
        )


class SignalDecision(BaseModel):
    pair:           str
    signal:         Optional[Any]  = None
    passed:         bool           = False
    blocked_by:     Optional[str]  = None
    block_reason:   Optional[str]  = None
    filter_results: Dict[str, Any] = Field(default_factory=dict)
    order_result:   Optional[Any]  = None
    timestamp:      Optional[datetime] = None

    def __init__(self, **data):
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class LiveTradeEngine:

    def __init__(self, config: EngineConfig, broker=None):
        self.config = config
        self.broker = broker

        # v3.2 corrected imports
        from app.services.market_analyzer_advanced import MAETradingFormula
        from app.services.news_filter import NewsFilter
        from app.services.session_filter import SessionFilter
        from app.services.circuit_breaker import CircuitBreaker, TradeOutcome
        from app.services.correlation_filter import CorrelationFilter, OpenPosition
        from app.services.ai_explanations import AIExplanationService
        from app.services.alerts import AlertManager

        self.mae_engine = MAETradingFormula()
        self.mae_engine.set_data_fetcher(None)
        self.news_filter = NewsFilter()
        self.session_filter = SessionFilter()
        self.circuit_breaker = CircuitBreaker(
            daily_loss_limit_pct=config.daily_loss_limit_pct,
            weekly_loss_limit_pct=config.weekly_loss_limit_pct,
            max_consecutive_losses=config.max_consecutive_losses,
            max_drawdown_pct=config.max_drawdown_pct,
            starting_balance=config.account_balance,
        )
        self.correlation_filter = CorrelationFilter(
            max_lots_per_currency=config.max_lots_per_currency
        )
        self.ai_service = AIExplanationService()
        self.alert_manager = AlertManager()

        # Thread-safe bounded decision history
        self._decisions: deque = deque(maxlen=config.max_decision_history)
        self._decisions_lock = threading.RLock()

        # Daily counters for O(1) status
        self._daily_decisions = 0
        self._daily_signals = 0
        self._last_reset_date = datetime.now(timezone.utc).date()
        self._counters_lock = threading.RLock()

        self._TradeOutcome = TradeOutcome
        self._OpenPosition = OpenPosition
        self._running = False

    async def close(self):
        """Graceful shutdown — close connections."""
        self._running = False
        if self.broker and hasattr(self.broker, 'close'):
            try:
                await self.broker.close()
            except Exception as e:
                logger.warning(f"Broker close error: {e}")
        logger.info("LiveTradeEngine shut down")

    def _reset_daily_counters(self):
        """Reset daily counters if date changed."""
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_date:
            with self._counters_lock:
                self._daily_decisions = 0
                self._daily_signals = 0
                self._last_reset_date = today

    async def run_cycle(self, alert_recipients=None) -> List[SignalDecision]:
        from app.services.data_fetcher import data_fetcher

        self._running = True
        self._reset_daily_counters()
        now = datetime.now(timezone.utc)
        decisions = []

        # Sync balance from broker
        if self.broker and self.config.use_circuit_breaker:
            try:
                account = await self.broker.get_account()
                self.circuit_breaker.update_balance(account.balance)
            except Exception as e:
                logger.warning(f"Balance sync failed: {e}")

        # Global circuit breaker check
        if self.config.use_circuit_breaker:
            cb = self.circuit_breaker.check(now)
            if not cb.is_allowed:
                for pair in self.config.watchlist:
                    decisions.append(SignalDecision(
                        pair=pair, passed=False,
                        blocked_by="circuit_breaker",
                        block_reason=cb.reason,
                        filter_results={"circuit_breaker": cb.dict()},
                    ))
                with self._counters_lock:
                    self._daily_decisions += len(decisions)
                with self._decisions_lock:
                    self._decisions.extend(decisions)
                return decisions

        tasks = [self._evaluate_pair(pair, now) for pair in self.config.watchlist]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Pair evaluation error: {result}")
                continue
            decisions.append(result)
            with self._counters_lock:
                self._daily_decisions += 1
                if result.passed:
                    self._daily_signals += 1

        # Alerts for passed signals
        passed = [d for d in decisions if d.passed and d.signal]
        if passed and alert_recipients:
            for decision in passed:
                try:
                    await asyncio.to_thread(
                        self.alert_manager.send_signal_alert,
                        decision.signal, alert_recipients, ["email", "telegram"],
                    )
                except Exception as e:
                    logger.warning(f"Alert failed: {e}")

        with self._decisions_lock:
            self._decisions.extend(decisions)
        return decisions

    async def _evaluate_pair(self, pair: str, now: datetime) -> SignalDecision:
        from app.services.data_fetcher import data_fetcher
        from app.services.broker_connector import OrderStatus

        filter_results = {}

        # Gate 1: Session
        if self.config.use_session_filter:
            sr = self.session_filter.check(pair, now, strict=self.config.strict_session)
            filter_results["session"] = sr.dict()
            if not sr.is_allowed:
                return SignalDecision(pair=pair, passed=False, blocked_by="session_filter",
                                      block_reason=sr.reason, filter_results=filter_results)

        # Gate 2: News
        if self.config.use_news_filter:
            nr = await self.news_filter.check(pair, now, block_medium=self.config.block_medium_news)
            filter_results["news"] = nr.dict()
            if nr.is_blocked:
                return SignalDecision(pair=pair, passed=False, blocked_by="news_filter",
                                      block_reason=nr.reason, filter_results=filter_results)

        # Gate 3: Fetch candles
        try:
            candles = await data_fetcher.get_candles(pair, self.config.timeframe, limit=300)

            if self.config.use_multi_timeframe:
                # Determine direction from signal context (not hardcoded "buy")
                # We check H4 alignment after signal generation for the correct direction
                h4_allowed = None
                h4_reason = "Direction not yet determined"
            else:
                h4_allowed = True
                h4_reason = "Multi-timeframe disabled"
        except Exception as e:
            return SignalDecision(pair=pair, passed=False, blocked_by="data_fetch",
                                  block_reason=f"Data fetch failed: {e}", filter_results=filter_results)

        # Gate 4: Signal generation
        try:
            signal = self.mae_engine.generate_signal(
                candles=candles, pair=pair, timeframe=self.config.timeframe,
                account_balance=self.config.account_balance,
                risk_percent=self.config.risk_percent, bar_closed=True,
                mtf_aligned=True,  # H4 check deferred to after direction known
            )
        except Exception as e:
            return SignalDecision(pair=pair, passed=False, blocked_by="signal_engine",
                                  block_reason=f"Signal error: {e}", filter_results=filter_results)

        if not signal:
            return SignalDecision(pair=pair, passed=False, blocked_by="no_signal",
                                  block_reason="No valid pattern", filter_results=filter_results)

        filter_results["signal"] = {
            "pattern": signal.pattern.value,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
        }

        # Gate 4b: H4 alignment check (deferred — now we know the direction)
        if self.config.use_multi_timeframe and h4_allowed is None:
            try:
                h4_allowed, h4_reason = await self.mae_engine.check_h4_alignment(
                    pair, signal.direction.value
                )
                filter_results["h4_alignment"] = {"allowed": h4_allowed, "reason": h4_reason}
                if not h4_allowed:
                    return SignalDecision(
                        pair=pair, signal=signal, passed=False,
                        blocked_by="h4_misalignment",
                        block_reason=h4_reason,
                        filter_results=filter_results,
                    )
            except Exception as e:
                logger.warning(f"H4 alignment check failed for {pair}: {e}")
                # Fail open — don't block on H4 check error
                h4_allowed = True
                h4_reason = f"H4 check error, allowing: {e}"

        # Gate 5: Confidence (configurable threshold)
        if signal.confidence < self.config.confidence_threshold:
            return SignalDecision(
                pair=pair, signal=signal, passed=False, blocked_by="confidence",
                block_reason=f"Confidence {signal.confidence:.0%} below threshold {self.config.confidence_threshold:.0%}",
                filter_results=filter_results,
            )

        # Gate 6: Correlation
        if self.config.use_correlation_filter and signal.position_size:
            lots = signal.position_size.get("standard_lots", 0.01)
            if lots <= 0 or lots > self.config.max_lots_per_currency:
                logger.warning(f"Position size {lots} lots out of range for {pair}")
                lots = max(0.01, min(lots, self.config.max_lots_per_currency))

            cr = self.correlation_filter.check(pair, signal.direction.value, lots)
            filter_results["correlation"] = cr.dict()
            if not cr.is_allowed:
                return SignalDecision(pair=pair, signal=signal, passed=False,
                                      blocked_by="correlation_filter", block_reason=cr.reason,
                                      filter_results=filter_results)

        # AI explanation (optional)
        try:
            signal.explanation = await self.ai_service.explain_signal(signal)
        except Exception as e:
            logger.warning(f"AI explanation failed for {pair}: {e}")

        # Gate 7: Broker health check
        if self.config.auto_execute and self.broker:
            try:
                if hasattr(self.broker, 'is_connected') and not self.broker.is_connected():
                    return SignalDecision(
                        pair=pair, signal=signal, passed=False,
                        blocked_by="broker_disconnected",
                        block_reason="Broker not connected",
                        filter_results=filter_results,
                    )
            except Exception:
                pass

        # Gate 8: Execute
        order_result = None
        if self.config.auto_execute and self.broker:
            if not signal.position_size or "units" not in signal.position_size:
                logger.warning(f"Invalid position size for {pair}, using fallback")
                units = 1000
            else:
                units = signal.position_size["units"]
                if units <= 0:
                    logger.warning(f"Zero or negative units for {pair}, using fallback")
                    units = 1000

            try:
                order_result = await self.broker.execute_signal(signal, units)
            except Exception as e:
                logger.error(f"Broker execution error for {pair}: {e}")
                return SignalDecision(pair=pair, signal=signal, passed=False,
                                      blocked_by="broker_error", block_reason=str(e),
                                      filter_results=filter_results)

            execution_succeeded = (
                order_result.success and
                order_result.status in (OrderStatus.FILLED, OrderStatus.RECONCILED)
            )

            if execution_succeeded:
                lots = signal.position_size.get("standard_lots", 0.01) if signal.position_size else 0.01
                self.correlation_filter.add_position(self._OpenPosition(
                    pair=pair,
                    direction=signal.direction.value,
                    lots=lots,
                    entry_price=order_result.entry_price or signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    risk_amount=signal.position_size.get("risk_amount", 0) if signal.position_size else 0,
                ))

                if order_result.reconciled:
                    logger.info(
                        f"{pair} position registered via reconciliation — "
                        f"order {order_result.order_id} confirmed post-timeout"
                    )
            else:
                return SignalDecision(
                    pair=pair, signal=signal, passed=False,
                    blocked_by="broker_rejection",
                    block_reason=order_result.error or "Order not filled",
                    filter_results=filter_results,
                    order_result=order_result,
                )

        return SignalDecision(
            pair=pair, signal=signal, passed=True,
            filter_results=filter_results, order_result=order_result,
        )

    def record_trade_outcome(self, pair: str, pnl: float, result: str, balance: float):
        """Record trade outcome and update filters."""
        valid_results = {"win", "loss", "breakeven"}
        if result not in valid_results:
            logger.warning(f"Invalid result '{result}' for {pair}, defaulting to 'loss'")
            result = "loss"

        # Guard against divide-by-zero
        base_balance = balance if balance > 0 else self.config.account_balance
        if base_balance <= 0:
            logger.error(f"Balance is zero or negative ({balance}), cannot calculate PnL percentage")
            base_balance = 1.0  # Prevent crash, though data is meaningless

        self.circuit_breaker.record_trade(self._TradeOutcome(
            pair=pair, pnl=pnl,
            pnl_pct=(pnl / base_balance) * 100,
            result=result,
        ))
        self.circuit_breaker.update_balance(balance)
        self.correlation_filter.close_position(pair)

    def kill(self, reason: str = "Manual kill"):
        self.circuit_breaker.kill(reason)

    def resume(self):
        self.circuit_breaker.reset_kill()

    def get_status(self) -> Dict:
        cb = self.circuit_breaker.get_stats()
        open_pos = self.correlation_filter.get_open_positions()
        self._reset_daily_counters()

        return {
            "circuit_breaker": cb,
            "open_positions": len(open_pos),
            "total_risk_amount": round(self.correlation_filter.get_total_risk(), 2),
            "net_exposure": self.correlation_filter.get_net_exposure(),
            "decisions_today": self._daily_decisions,
            "signals_today": self._daily_signals,
            "engine_running": self._running,
            "config": {
                "auto_execute": self.config.auto_execute,
                "paper_trading": self.config.paper_trading,
                "confidence_threshold": self.config.confidence_threshold,
            }
        }
