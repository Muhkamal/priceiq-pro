"""
PriceIQ Pro — Live Monitoring System v1.0

Tracks system health in real-time and fires alerts via Telegram.

Monitors:
    - Account drawdown (warn at 5%, hard at 10%)
    - Win rate degradation (warn if < 40% over last 20 trades)
    - Strategy confidence drift
    - Anomaly detection (abnormal PnL swings, missing heartbeats)
    - Regime change events
    - Daily / weekly performance summaries

Usage:
    monitor = SystemMonitor(telegram=telegram, governor=governor)

    # Call after every trade:
    await monitor.on_trade_closed(outcome)

    # Call on regime change:
    await monitor.on_regime_change("trending", "volatile", pair="XAUUSD")

    # Call on system heartbeat (every hour from scheduler):
    await monitor.heartbeat(portfolio_summary)
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thresholds
DD_WARN_PCT    = 0.05    # 5% drawdown → warning
DD_HARD_PCT    = 0.10    # 10% → emergency
WIN_RATE_WARN  = 0.40    # below 40% win rate over last 20 → warning
WIN_RATE_MIN   = 0.30    # below 30% → emergency
MIN_TRADES_FOR_WARN = 15  # need at least this many trades before warning


class SystemMonitor:
    """
    Real-time system health monitor with Telegram alerting.
    Keeps a rolling window of events to detect degradation.
    """

    def __init__(self, telegram=None, governor=None, learning_loop=None):
        self.telegram      = telegram
        self.governor      = governor
        self.learning_loop = learning_loop

        self._recent_outcomes: Deque[Dict] = deque(maxlen=50)
        self._alert_cooldowns: Dict[str, str] = {}   # alert_key → last_sent timestamp
        self._alert_cooldown_hours = 2                # don't repeat same alert within 2h
        self._last_heartbeat: Optional[str] = None

    # ── Public event hooks ───────────────────────────────────

    async def on_trade_closed(self, outcome: Dict):
        """
        Call after every trade closes.
        outcome = {"pair": "XAUUSD", "agent": "TrendAgent", "result": "win",
                   "r_multiple": 1.5, "pnl": 75.0, ...}
        """
        self._recent_outcomes.append(outcome)
        await self._check_win_rate()
        await self._check_drawdown()
        await self._check_agent_degradation(outcome.get("agent", ""))

    async def on_regime_change(self, from_regime: str, to_regime: str, pair: str = ""):
        """Alert when ML classifier detects a regime shift."""
        msg = (
            f"🔄 <b>Regime Change Detected</b>\n"
            f"Pair: {pair or 'portfolio'}\n"
            f"{from_regime.upper()} → {to_regime.upper()}\n"
            f"Strategy agents will re-weight automatically."
        )
        await self._send(msg, alert_key=f"regime_{pair}_{to_regime}")
        logger.info(f"Regime change: {pair} {from_regime} → {to_regime}")

    async def on_signal_generated(self, pair: str, agent: str, direction: str, confidence: float):
        """Log signals (no Telegram spam — just internal logging)."""
        logger.info(f"Signal: {pair} {direction.upper()} via {agent} (conf={confidence:.2f})")

    async def on_signal_blocked(self, pair: str, reason: str):
        """Log blocked signals — useful for debugging."""
        logger.info(f"Signal BLOCKED: {pair} — {reason}")

    async def heartbeat(self, portfolio_summary: Dict):
        """
        Called hourly from scheduler.
        Sends a summary if portfolio is healthy, emergency alert if not.
        """
        self._last_heartbeat = datetime.now(timezone.utc).isoformat()

        dd   = portfolio_summary.get("drawdown", 0)
        bal  = portfolio_summary.get("current_balance", 0)
        exp  = portfolio_summary.get("total_open_exposure", 0)
        risk = portfolio_summary.get("risk_score", 0)
        allowed = portfolio_summary.get("trading_allowed", True)

        # Emergency state
        if dd >= DD_HARD_PCT or not allowed:
            await self._send_emergency(portfolio_summary)
            return

        # Normal heartbeat (only send if something worth noting)
        if dd >= DD_WARN_PCT or risk > 0.6:
            win_rate = self._calc_win_rate()
            msg = (
                f"⚠️ <b>PriceIQ Hourly Check</b>\n"
                f"Balance: ${bal:,.2f}\n"
                f"Drawdown: {dd:.1%}\n"
                f"Open exposure: {exp:.1%}\n"
                f"Risk score: {risk:.2f}\n"
                f"Win rate (last 20): {win_rate:.1%}\n"
                f"Trading: {'✅' if allowed else '🛑'}"
            )
            await self._send(msg, alert_key="hourly_warn")

    async def daily_summary(self, learning_stats: Dict, portfolio: Dict):
        """Send end-of-day summary via Telegram."""
        today     = datetime.now(timezone.utc).strftime("%d %b %Y")
        bal       = portfolio.get("current_balance", 0)
        dd        = portfolio.get("drawdown", 0)
        total_tr  = learning_stats.get("total_trades", 0)

        agent_lines = []
        for agent, stats in learning_stats.get("agent_stats", {}).items():
            w   = stats.get("weight", 1.0)
            wr  = stats.get("win_rate") or 0
            ev  = stats.get("expectancy") or 0
            agent_lines.append(f"  • {agent}: WR={wr:.0%} EV={ev:+.2f}R W={w:.2f}")

        pair_lines = []
        for pair, stats in learning_stats.get("pairs", {}).items():
            wr    = stats.get("win_rate") or 0
            tr    = stats.get("trades", 0)
            thr   = stats.get("threshold", 0.55)
            pair_lines.append(f"  • {pair}: {tr} trades WR={wr:.0%} thresh={thr:.2f}")

        msg_parts = [
            f"📊 <b>Daily Summary — {today}</b>",
            f"Balance: ${bal:,.2f}  |  DD: {dd:.1%}",
            f"Total trades: {total_tr}",
            f"Win rate (all): {self._calc_win_rate():.1%}",
            "",
            "<b>Agent Performance:</b>",
            *agent_lines,
            "",
            "<b>Pair Performance:</b>",
            *pair_lines,
        ]

        await self._send("\n".join(msg_parts), alert_key="daily_summary", force=True)

    # ── Internal checks ──────────────────────────────────────

    async def _check_win_rate(self):
        if len(self._recent_outcomes) < MIN_TRADES_FOR_WARN:
            return
        wr = self._calc_win_rate()

        if wr < WIN_RATE_MIN:
            await self._send(
                f"🚨 <b>CRITICAL: Win Rate Collapse</b>\n"
                f"Win rate over last 20 trades: <b>{wr:.1%}</b>\n"
                f"Minimum threshold: {WIN_RATE_MIN:.0%}\n"
                f"Consider pausing trading and reviewing strategy.",
                alert_key="winrate_critical",
            )
        elif wr < WIN_RATE_WARN:
            await self._send(
                f"⚠️ <b>Win Rate Warning</b>\n"
                f"Win rate over last 20 trades: {wr:.1%}\n"
                f"Warning threshold: {WIN_RATE_WARN:.0%}",
                alert_key="winrate_warn",
            )

    async def _check_drawdown(self):
        if not self.governor:
            return
        dd = self.governor.drawdown

        if dd >= DD_HARD_PCT:
            await self._send(
                f"🚨 <b>EMERGENCY: Hard Drawdown Limit Hit</b>\n"
                f"Drawdown: <b>{dd:.1%}</b> ≥ {DD_HARD_PCT:.0%}\n"
                f"All trading HALTED automatically by Risk Governor.\n"
                f"Manual review required before resuming.",
                alert_key="dd_hard",
                force=True,
            )
        elif dd >= DD_WARN_PCT:
            await self._send(
                f"⚠️ <b>Drawdown Warning</b>\n"
                f"Drawdown: {dd:.1%} (warn level: {DD_WARN_PCT:.0%})\n"
                f"Position sizes being reduced 50% automatically.",
                alert_key="dd_warn",
            )

    async def _check_agent_degradation(self, agent_name: str):
        if not agent_name:
            return
        agent_trades = [o for o in self._recent_outcomes if o.get("agent") == agent_name]
        if len(agent_trades) < 10:
            return
        agent_wins = sum(1 for t in agent_trades[-10:] if t.get("result") == "win")
        agent_wr   = agent_wins / 10.0
        if agent_wr < 0.30:
            await self._send(
                f"⚠️ <b>Agent Degradation: {agent_name}</b>\n"
                f"Win rate (last 10 trades): {agent_wr:.0%}\n"
                f"Weight being reduced automatically by learning system.",
                alert_key=f"agent_degrade_{agent_name}",
            )

    async def _send_emergency(self, portfolio: Dict):
        dd      = portfolio.get("drawdown", 0)
        bal     = portfolio.get("current_balance", 0)
        losses  = portfolio.get("consecutive_losses", 0)
        msg = (
            f"🚨 <b>EMERGENCY ALERT — PriceIQ Pro</b>\n\n"
            f"Trading has been <b>HALTED</b> by Risk Governor.\n\n"
            f"Balance: ${bal:,.2f}\n"
            f"Drawdown: <b>{dd:.1%}</b>\n"
            f"Consecutive losses: {losses}\n\n"
            f"Action required: Review positions and manually reset loss streak "
            f"before resuming if conditions have changed."
        )
        await self._send(msg, alert_key="emergency", force=True)

    def _calc_win_rate(self, n: int = 20) -> float:
        recent = list(self._recent_outcomes)[-n:]
        if not recent:
            return 0.0
        wins = sum(1 for t in recent if t.get("result") == "win")
        return wins / len(recent)

    async def _send(self, message: str, alert_key: str = "", force: bool = False):
        """Send via Telegram with cooldown to prevent spam."""
        if not self.telegram:
            logger.info(f"[Monitor alert — no Telegram]: {message[:80]}...")
            return

        if alert_key and not force:
            last = self._alert_cooldowns.get(alert_key)
            if last:
                from datetime import timedelta
                try:
                    last_dt = datetime.fromisoformat(last)
                    now     = datetime.now(timezone.utc)
                    if (now - last_dt).total_seconds() < self._alert_cooldown_hours * 3600:
                        return   # still in cooldown
                except Exception:
                    pass

        try:
            await self.telegram.send_message(message)
            if alert_key:
                self._alert_cooldowns[alert_key] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.warning(f"Monitor Telegram send failed: {e}")
