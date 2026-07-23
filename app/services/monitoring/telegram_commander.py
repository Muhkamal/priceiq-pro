"""
PriceIQ Pro — Two-Way Telegram Command Bot v1.0

Enables controlling PriceIQ from your phone via Telegram commands.
No more opening a browser — /pause, /resume, /status, /close from anywhere.

Commands:
    /start          — welcome message + command list
    /status         — full system health snapshot
    /positions      — all open positions with P&L
    /portfolio      — balance, drawdown, risk score
    /agents         — current agent weights + best per regime
    /journal        — last 10 trades summary
    /regime         — current regime + shift risk
    /calendar       — next 3 high-impact events
    /pause          — pause all new signals (emergency)
    /resume         — resume signals
    /close XAUUSD   — force close a position
    /scan           — trigger immediate scan now
    /balance        — current account balance
    /risk           — portfolio heat + VaR
    /help           — command reference

Architecture:
    Long-polling in a background asyncio task.
    Each command dispatches to the V5 orchestrator.
    Flood protection: 1 command per 3 seconds per user.
    Auth: only responds to TELEGRAM_CHAT_ID (your chat).

Usage in main.py lifespan:
    from app.services.monitoring.telegram_commander import TelegramCommander
    commander = TelegramCommander()
    asyncio.create_task(commander.start_polling())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID",   "")
API_BASE   = f"https://api.telegram.org/bot{BOT_TOKEN}"
POLL_TIMEOUT = 30      # long-poll timeout seconds
FLOOD_WINDOW = 3.0     # seconds between commands per user


def _api(method: str, **params) -> Optional[Dict]:
    """Synchronous Telegram API call."""
    try:
        url  = f"{API_BASE}/{method}"
        data = json.dumps(params).encode()
        req  = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=35) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Telegram API {method} failed: {e}")
        return None


async def _api_async(method: str, **params) -> Optional[Dict]:
    return await asyncio.to_thread(_api, method, **params)


class TelegramCommander:
    """
    Long-polling Telegram bot that dispatches commands to the V5 system.
    Runs as a background asyncio task alongside FastAPI.
    """

    COMMANDS = {
        "/start":     "Welcome + command list",
        "/status":    "Full system health",
        "/positions": "Open positions",
        "/portfolio": "Balance + drawdown",
        "/agents":    "Agent weights",
        "/journal":   "Last 10 trades",
        "/regime":    "Current regime",
        "/calendar":  "Upcoming news events",
        "/pause":     "Pause new signals",
        "/resume":    "Resume signals",
        "/close":     "/close PAIR — force close position",
        "/scan":      "Trigger immediate scan",
        "/balance":   "Account balance",
        "/risk":      "VaR + portfolio heat",
        "/help":      "This command list",
    }

    def __init__(self):
        self._offset      = 0
        self._paused      = False
        self._running     = False
        self._last_cmd:   Dict[str, float] = {}   # chat_id → timestamp
        self._v5          = None   # lazy-loaded

    def _get_v5(self):
        if not self._v5:
            try:
                from app.services.v5_orchestrator_final import get_v5
                self._v5 = get_v5()
            except Exception:
                pass
        return self._v5

    async def start_polling(self):
        """Start long-polling loop. Run as asyncio.create_task()."""
        if not BOT_TOKEN or not CHAT_ID:
            logger.warning("TelegramCommander: BOT_TOKEN or CHAT_ID not set — commander disabled")
            return

        self._running = True
        logger.info("TelegramCommander: polling started")
        await self._send("🟢 <b>PriceIQ Pro online.</b> Type /help for commands.")

        while self._running:
            try:
                updates = await asyncio.to_thread(
                    _api, "getUpdates",
                    offset=self._offset,
                    timeout=POLL_TIMEOUT,
                    allowed_updates=["message"],
                )
                if updates and updates.get("ok"):
                    for update in updates.get("result", []):
                        self._offset = update["update_id"] + 1
                        await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"TelegramCommander poll error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False

    # ── Update handler ────────────────────────────────────────

    async def _handle_update(self, update: Dict):
        msg = update.get("message", {})
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        text    = msg.get("text", "").strip()

        # Auth: only respond to configured chat
        if chat_id != str(CHAT_ID):
            logger.debug(f"Commander: ignoring message from unauthorized chat {chat_id}")
            return

        # Flood protection
        now = datetime.now(timezone.utc).timestamp()
        last = self._last_cmd.get(chat_id, 0)
        if now - last < FLOOD_WINDOW:
            return
        self._last_cmd[chat_id] = now

        if not text.startswith("/"):
            return

        parts   = text.split()
        command = parts[0].lower().split("@")[0]   # handle /cmd@botname
        args    = parts[1:]

        logger.info(f"TelegramCommander: received {command} {args}")

        handlers = {
            "/start":     self._cmd_start,
            "/help":      self._cmd_help,
            "/status":    self._cmd_status,
            "/positions": self._cmd_positions,
            "/portfolio": self._cmd_portfolio,
            "/agents":    self._cmd_agents,
            "/journal":   self._cmd_journal,
            "/regime":    self._cmd_regime,
            "/calendar":  self._cmd_calendar,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
            "/close":     lambda: self._cmd_close(args),
            "/scan":      self._cmd_scan,
            "/balance":   self._cmd_balance,
            "/risk":      self._cmd_risk,
        }

        handler = handlers.get(command)
        if handler:
            try:
                await handler()
            except Exception as e:
                await self._send(f"❌ Command error: <code>{e}</code>")
                logger.error(f"Commander handler error: {e}")
        else:
            await self._send(f"Unknown command: {command}\nType /help for the list.")

    # ── Command implementations ───────────────────────────────

    async def _cmd_start(self):
        await self._send(
            "🤖 <b>PriceIQ Pro — Trading Intelligence</b>\n\n"
            "Your AI trading system is online and monitoring markets.\n\n"
            "Type /help for all commands."
        )

    async def _cmd_help(self):
        lines = ["📋 <b>Available Commands</b>\n"]
        for cmd, desc in self.COMMANDS.items():
            lines.append(f"<code>{cmd}</code> — {desc}")
        await self._send("\n".join(lines))

    async def _cmd_status(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 system not initialised yet.")
            return
        s  = v5.get_system_status()
        p  = s.get("portfolio", {})
        rt = s.get("regime_transition", {})
        msg = (
            f"📊 <b>System Status</b>\n\n"
            f"Version: {s.get('version', '?')}\n"
            f"Balance: <b>${p.get('current_balance', 0):,.2f}</b>\n"
            f"Drawdown: {p.get('drawdown', 0):.1%}\n"
            f"Risk score: {p.get('risk_score', 0):.2f}\n"
            f"Open positions: {p.get('open_positions', 0)}\n"
            f"Trading: {'✅' if p.get('trading_allowed') else '🛑 HALTED'}\n\n"
            f"Regime: <b>{rt.get('current', '?')}</b> "
            f"({rt.get('streak_bars', 0)} bars)\n"
            f"Shift risk: {rt.get('shift_risk', 0):.0%}\n\n"
            f"Model trained: {'✅' if s.get('regime_model_trained') else '❌ using heuristic'}\n"
            f"Signals: {'⏸ PAUSED' if self._paused else '▶️ ACTIVE'}"
        )
        await self._send(msg)

    async def _cmd_positions(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        positions = v5.trade_manager.get_open_positions()
        if not positions:
            await self._send("📭 No open positions.")
            return
        lines = ["📈 <b>Open Positions</b>\n"]
        for pair, pos in positions.items():
            emoji = "🟢" if pos["direction"] == "buy" else "🔴"
            lines.append(
                f"{emoji} <b>{pair}</b> {pos['direction'].upper()}\n"
                f"  Entry: {pos['entry']:.5f} | SL: {pos['current_sl']:.5f}\n"
                f"  Lots: {pos['lots_remaining']} | Age: {pos['age_hours']}h\n"
                f"  Status: {pos['status']} | PnL: ${pos['pnl_realised']:+.2f}\n"
                f"  BE: {'✅' if pos['sl_at_be'] else '❌'} Trail: {'✅' if pos['trailing'] else '❌'}\n"
            )
        await self._send("\n".join(lines))

    async def _cmd_portfolio(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        p   = v5.governor.get_portfolio_summary()
        var = v5.var_engine.compute(v5.governor._open_positions)
        msg = (
            f"💼 <b>Portfolio</b>\n\n"
            f"Balance: <b>${p['current_balance']:,.2f}</b>\n"
            f"Peak: ${p['peak_balance']:,.2f}\n"
            f"Drawdown: <b>{p['drawdown']:.1%}</b>\n"
            f"Exposure: {p['total_open_exposure']:.1%}\n"
            f"Consec losses: {p['consecutive_losses']}\n"
            f"Risk score: {p['risk_score']:.2f}\n\n"
            f"Portfolio heat: <b>{var.portfolio_heat_pct:.1%}</b>\n"
            f"VaR (95%): ${var.var_historical_usd or 0:,.0f}\n"
            f"Safe to trade: {'✅' if var.safe_to_trade else '🛑 NO'}"
        )
        await self._send(msg)

    async def _cmd_agents(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        weights = v5.learning.get_agent_weights()
        best    = v5.regime_learner.get_best_agent_per_regime()
        lines   = ["🤖 <b>Agent Weights</b>\n"]
        for agent, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 5)
            lines.append(f"  {agent}: {w:.3f} {bar}")
        lines.append("\n<b>Best agent per regime:</b>")
        for regime, (agent, w) in best.items():
            lines.append(f"  {regime}: {agent} ({w:.2f})")
        await self._send("\n".join(lines))

    async def _cmd_journal(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        stats   = v5.journal.full_stats()
        recent  = v5.journal.query(last_n=5)
        lines   = [
            f"📒 <b>Trade Journal</b>\n",
            f"Total trades: {stats.get('total_trades', 0)}",
            f"Win rate: {stats.get('win_rate', 0):.1%}",
            f"Expectancy: {stats.get('expectancy', 0):+.2f}R",
            f"Total PnL: ${stats.get('total_pnl', 0):+,.2f}\n",
            "<b>Recent trades:</b>",
        ]
        for e in reversed(recent):
            emoji = "✅" if e.outcome == "win" else "❌"
            lines.append(
                f"{emoji} {e.pair} {e.direction} | {e.outcome} "
                f"{e.r_multiple:+.1f}R ${e.pnl_usd:+.0f} | {e.agent}"
            )
        await self._send("\n".join(lines))

    async def _cmd_regime(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        streak_r, streak_n = v5.regime_transition.regime_streak()
        matrix  = v5.regime_transition.get_transition_matrix()
        row     = matrix.get(streak_r, {})
        msg = (
            f"🔄 <b>Regime Intelligence</b>\n\n"
            f"Current: <b>{streak_r.upper()}</b> ({streak_n} bars)\n"
            f"Shift risk: {v5.regime_transition.shift_risk(streak_r):.0%}\n"
            f"Volatile surge: {v5.regime_transition.volatility_surge_prob(streak_r):.0%}\n"
            f"Size multiplier: {v5.regime_transition.size_multiplier(streak_r):.0%}\n\n"
            f"<b>Transition probs from {streak_r}:</b>\n"
            f"  → trending:  {row.get('trending', 0):.0%}\n"
            f"  → ranging:   {row.get('ranging', 0):.0%}\n"
            f"  → volatile:  {row.get('volatile', 0):.0%}\n\n"
            f"Drift: {v5.drift_monitor.status_dict().get('max_psi', 'N/A')}"
        )
        await self._send(msg)

    async def _cmd_calendar(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        events = v5.calendar.upcoming_events(hours_ahead=24, impact_filter="HIGH")[:5]
        if not events:
            await self._send("📅 No HIGH-impact events in next 24h.")
            return
        lines = ["📅 <b>Upcoming HIGH-Impact Events (24h)</b>\n"]
        now   = datetime.now(timezone.utc)
        for e in events:
            mins = (e.dt_utc - now).total_seconds() / 60
            lines.append(
                f"⚡ <b>{e.title}</b> ({e.currency})\n"
                f"   {e.dt_utc.strftime('%H:%M UTC')} — in {mins:.0f} min"
            )
        await self._send("\n".join(lines))

    async def _cmd_pause(self):
        self._paused = True
        # Set flag on orchestrator if available
        v5 = self._get_v5()
        if v5:
            v5._signals_paused = True
        await self._send("⏸ <b>Signals PAUSED.</b> New signals will be blocked.\nType /resume to restart.")

    async def _cmd_resume(self):
        self._paused = False
        v5 = self._get_v5()
        if v5:
            v5._signals_paused = False
        await self._send("▶️ <b>Signals RESUMED.</b> System is active.")

    async def _cmd_close(self, args):
        if not args:
            await self._send("Usage: /close PAIR (e.g. /close XAUUSD)")
            return
        pair = args[0].upper()
        v5   = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        positions = v5.trade_manager.get_open_positions()
        if pair not in positions:
            await self._send(f"No open position for {pair}.")
            return
        # We don't have live price here — send instruction
        await self._send(
            f"⚠️ To force-close {pair}, use the dashboard:\n"
            f"POST /api/v5/close/{pair}\n\n"
            f"Or call your broker directly to close the position."
        )

    async def _cmd_scan(self):
        await self._send("🔍 Triggering immediate scan...")
        try:
            from app.services.scheduler import get_scheduler
            scheduler = get_scheduler()
            if scheduler:
                asyncio.create_task(scheduler._run_v5_hourly_scan())
                await self._send("✅ Scan triggered. Results incoming via alerts.")
            else:
                await self._send("⚠️ Scheduler not available.")
        except Exception as e:
            await self._send(f"❌ Scan failed: {e}")

    async def _cmd_balance(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        p = v5.governor.get_portfolio_summary()
        await self._send(
            f"💰 Balance: <b>${p['current_balance']:,.2f}</b>\n"
            f"Peak: ${p['peak_balance']:,.2f}\n"
            f"Drawdown: {p['drawdown']:.1%}"
        )

    async def _cmd_risk(self):
        v5 = self._get_v5()
        if not v5:
            await self._send("⚠️ V5 not initialised.")
            return
        var = v5.var_engine.compute(v5.governor._open_positions)
        stress = v5.var_engine.stress_test(v5.governor._open_positions)
        lines = [
            "⚖️ <b>Risk Dashboard</b>\n",
            f"Portfolio heat: <b>{var.portfolio_heat_pct:.1%}</b>",
            f"VaR 95% (hist): ${var.var_historical_usd or 0:,.0f}",
            f"VaR 95% (MC):   ${var.var_montecarlo_usd or 0:,.0f}",
            f"Safe to trade: {'✅' if var.safe_to_trade else '🛑 NO'}",
            "",
            "<b>Stress tests:</b>",
        ]
        for s in stress[:3]:
            surv = "✅" if s["survivable"] else "💀"
            lines.append(f"  {surv} {s['scenario']}: ${s['estimated_loss']:,.0f} ({s['loss_pct']:.1%})")
        await self._send("\n".join(lines))

    # ── Sender ────────────────────────────────────────────────

    async def _send(self, text: str):
        await _api_async(
            "sendMessage",
            chat_id=CHAT_ID,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
