from __future__ import annotations
import asyncio, json, logging, os
from typing import Dict, List, Set
import httpx

logger = logging.getLogger(__name__)

BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CHANNEL_ID    = os.environ.get("TELEGRAM_CHANNEL_ID", "")
SUBS_FILE     = "subscribers.json"

class SubscriberStore:
    def __init__(self):
        self._subs: Set[str] = set()
        self._load()
        if ADMIN_CHAT_ID:
            self._subs.add(str(ADMIN_CHAT_ID))

    def add(self, chat_id: str) -> bool:
        chat_id = str(chat_id)
        if chat_id in self._subs:
            return False
        self._subs.add(chat_id)
        self._save()
        return True

    def remove(self, chat_id: str) -> bool:
        chat_id = str(chat_id)
        if chat_id not in self._subs:
            return False
        self._subs.discard(chat_id)
        self._save()
        return True

    def all(self) -> List[str]:
        return list(self._subs)

    def count(self) -> int:
        return len(self._subs)

    def is_subscribed(self, chat_id: str) -> bool:
        return str(chat_id) in self._subs

    def _save(self):
        try:
            with open(SUBS_FILE, "w") as f:
                json.dump(list(self._subs), f)
        except Exception:
            pass

    def _load(self):
        try:
            with open(SUBS_FILE) as f:
                self._subs = set(str(s) for s in json.load(f))
        except Exception:
            self._subs = set()


class TelegramSignalBot:
    def __init__(self):
        self._store        = SubscriberStore()
        self._offset       = 0
        self._running      = False
        self._signal_count = 0
        self._win_count    = 0
        self._loss_count   = 0

    async def send(self, chat_id: str, text: str) -> bool:
        if not BOT_TOKEN:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={"chat_id": str(chat_id), "text": text, "parse_mode": "HTML"}
                )
                return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Send failed to {chat_id}: {e}")
            return False

    async def broadcast(self, text: str) -> Dict:
        targets = list(set(self._store.all() + ([str(CHANNEL_ID)] if CHANNEL_ID else [])))
        results = await asyncio.gather(*[self.send(t, text) for t in targets], return_exceptions=True)
        sent    = sum(1 for r in results if r is True)
        logger.info(f"Broadcast: {sent}/{len(targets)} delivered")
        return {"sent": sent, "total": len(targets)}

    async def broadcast_signal(self, result) -> Dict:
        self._signal_count += 1
        direction = getattr(result, "direction", "")
        pair      = getattr(result, "pair", "")
        conf      = getattr(result, "confidence", 0)
        agent     = getattr(result, "agent_used", "")
        regime    = getattr(result, "regime", "")
        entry     = getattr(result, "fill_price", 0)
        sl        = getattr(result, "stop_loss", 0)
        tp1       = getattr(result, "take_profit_1", 0)
        tp2       = getattr(result, "take_profit_2", 0)
        lots      = getattr(result, "adjusted_lots", 0)
        ev        = getattr(result, "expected_value", 0)

        emoji     = "🟢" if direction == "buy" else "🔴"
        reg_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "⚡"}.get(regime, "❓")
        decimals  = 2 if "XAU" in pair else 5

        msg = (
            f"{emoji} <b>SIGNAL #{self._signal_count}: {pair}</b>\n\n"
            f"Direction: <b>{direction.upper()}</b>\n"
            f"Confidence: <b>{conf:.0%}</b>\n"
            f"Agent: {agent}\n"
            f"Regime: {reg_emoji} {regime.title()}\n\n"
            f"📊 <b>Trade Levels</b>\n"
            f"Entry:  <code>{entry:.{decimals}f}</code>\n"
            f"SL:     <code>{sl:.{decimals}f}</code>\n"
            f"TP1:    <code>{tp1:.{decimals}f}</code>\n"
            f"TP2:    <code>{tp2:.{decimals}f}</code>\n"
            f"Lots:   {lots}\n"
            f"EV:     {ev:+.2f}R\n\n"
            f"⚠️ <i>Trade at your own risk.</i>\n"
            f"📡 <i>@daethdevilbot — #{self._signal_count}</i>"
        )
        return await self.broadcast(msg)

    async def broadcast_outcome(self, pair: str, direction: str, outcome: str, pnl_r: float):
        emoji = "✅" if outcome in ("win", "tp1", "tp2") else "❌"
        if outcome in ("win", "tp1", "tp2"):
            self._win_count += 1
        elif outcome == "loss":
            self._loss_count += 1
        total = self._win_count + self._loss_count
        wr    = self._win_count / total if total else 0
        msg   = (
            f"{emoji} <b>RESULT: {pair} {direction.upper()}</b>\n\n"
            f"Outcome: <b>{outcome.upper()}</b>\n"
            f"P&L: <b>{pnl_r:+.2f}R</b>\n\n"
            f"📊 Stats: {wr:.0%} WR ({self._win_count}W/{self._loss_count}L)\n"
            f"📡 <i>@daethdevilbot</i>"
        )
        await self.broadcast(msg)

    # --- USER COMMANDS ---

    async def handle_start(self, chat_id: str, username: str = ""):
        name = f"@{username}" if username else "there"
        await self.send(chat_id,
            f"👋 Welcome {name} to <b>Kayman Signals</b>!\n\n"
            f"🤖 AI-powered forex & gold signals\n"
            f"📊 7 agents + macro + COT + news\n\n"
            f"/subscribe — get signals\n"
            f"/stats — performance\n"
            f"/help — all commands"
        )

    async def handle_subscribe(self, chat_id: str, username: str = ""):
        if self._store.add(chat_id):
            await self.send(chat_id,
                f"✅ <b>Subscribed!</b>\n\n"
                f"You'll receive signals for:\n"
                f"XAUUSD • EURUSD • GBPUSD\n"
                f"USDJPY • USDCHF • AUDUSD • BTCUSD\n\n"
                f"Subscribers: {self._store.count()}\n"
                f"Type /unsubscribe to stop."
            )
            if chat_id != ADMIN_CHAT_ID and ADMIN_CHAT_ID:
                await self.send(ADMIN_CHAT_ID,
                    f"📣 New subscriber: @{username or chat_id}\n"
                    f"Total: {self._store.count()}"
                )
        else:
            await self.send(chat_id, "ℹ️ Already subscribed! Type /unsubscribe to stop.")

    async def handle_unsubscribe(self, chat_id: str):
        if self._store.remove(chat_id):
            await self.send(chat_id, "👋 Unsubscribed. Type /subscribe to rejoin anytime.")
        else:
            await self.send(chat_id, "ℹ️ You weren't subscribed. Type /subscribe to join.")

    async def handle_stats(self, chat_id: str):
        total = self._win_count + self._loss_count
        wr    = self._win_count / total if total else 0
        await self.send(chat_id,
            f"📊 <b>Kayman Signals Stats</b>\n\n"
            f"Signals sent: {self._signal_count}\n"
            f"Win rate: <b>{wr:.0%}</b>\n"
            f"Wins: {self._win_count} | Losses: {self._loss_count}\n"
            f"Subscribers: {self._store.count()}"
        )

    # --- PERFORMANCE TRACKER COMMANDS ---

    async def handle_leaderboard(self, chat_id: str):
        from app.services.learning.signal_performance_tracker import tracker
        board = tracker.get_leaderboard(top_n=10)
        if not board:
            await self.send(chat_id, "📊 No tracked signals yet. Need 5+ resolved per combo.")
            return
        lines = ["🏆 <b>Signal Leaderboard</b>\n"]
        for i, row in enumerate(board, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "▫️"
            lines.append(
                f"{emoji} <b>{row['combo']}</b>\n"
                f"   WR: {row['win_rate']:.0%} | Avg: {row['avg_pnl_r']:+.2f}R | N={row['count']}\n"
            )
        await self.send(chat_id, "\n".join(lines))

    async def handle_pairstats(self, chat_id: str, pair: str):
        from app.services.learning.signal_performance_tracker import tracker
        stats = tracker.get_stats(pair=pair)
        if stats["count"] == 0:
            await self.send(chat_id, f"📊 No resolved signals for {pair} yet.")
            return
        msg = (
            f"📊 <b>{pair} Performance</b>\n\n"
            f"Signals: <b>{stats['count']}</b>\n"
            f"Win Rate: <b>{stats['win_rate']:.0%}</b>\n"
            f"Avg R: <b>{stats['avg_pnl_r']:+.2f}R</b>\n"
            f"Total R: <b>{stats['total_r']:+.2f}R</b>\n\n"
            f"TP1: {stats['tp1']} | TP2: {stats['tp2']}\n"
            f"SL: {stats['sl']} | Timeout: {stats['timeout']}"
        )
        await self.send(chat_id, msg)

    async def handle_performance(self, chat_id: str):
        from app.services.learning.signal_performance_tracker import tracker
        pairs = tracker.get_pair_stats()
        if not pairs:
            await self.send(chat_id, "📊 No resolved signals yet. First signals need to hit SL or TP.")
            return
        lines = [
            f"📊 <b>Overall Performance</b>\n",
        ]
        for pair, st in pairs.items():
            lines.append(
                f"<b>{pair}</b>: {st['win_rate']:.0%} WR | {st['avg_pnl_r']:+.2f}R avg | n={st['count']}"
            )
        await self.send(chat_id, "\n".join(lines))

    async def handle_modelstats(self, chat_id: str):
        from app.services.ml.signal_outcome_predictor import outcome_predictor
        stats = outcome_predictor.get_stats()
        if stats["status"] == "cold_start":
            await self.send(chat_id,
                f"🧠 <b>AI Model: Cold Start</b>\n\n"
                f"Samples collected: {stats['total_samples']}/{outcome_predictor.MIN_SAMPLES}\n"
                f"Need {outcome_predictor.MIN_SAMPLES - stats['total_samples']} more resolved signals to train.\n\n"
                f"Keep taking signals — the model is watching."
            )
            return
        status_emoji = "✅" if stats["model_loaded"] else "⏳"
        await self.send(chat_id,
            f"🧠 <b>AI Signal Predictor</b> {status_emoji}\n\n"
            f"Status: <b>{stats['status'].upper()}</b>\n"
            f"Total samples: <b>{stats['total_samples']}</b>\n"
            f"Historical WR: <b>{stats['win_rate']:.0%}</b>\n"
            f"Recent 50 WR: <b>{stats['recent_50_wr']:.0%}</b>\n"
            f"Pending signals: {stats['pending_unlabeled']}\n\n"
            f"<i>Retrains every {outcome_predictor.RETRAIN_EVERY} new outcomes</i>"
        )

    async def handle_help(self, chat_id: str):
        await self.send(chat_id,
            f"📋 <b>Commands</b>\n\n"
            f"/subscribe — receive signals\n"
            f"/unsubscribe — stop signals\n"
            f"/stats — bot stats\n"
            f"/performance — signal performance by pair\n"
            f"/leaderboard — best agent+regime+pair combos\n"
            f"/pairstats PAIR — e.g. /pairstats XAUUSD\n"
            f"/modelstats — AI model training status\n"
            f"/help — this message\n\n"
            f"⚠️ <i>Trading involves risk.</i>"
        )

    # --- POLLING ---

    async def start_polling(self):
        self._running = True
        logger.info("Signal bot polling started")
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=35) as client:
                    resp = await client.get(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                        params={"offset": self._offset, "timeout": 30,
                                "allowed_updates": ["message"]},
                    )
                    if resp.status_code == 200:
                        for update in resp.json().get("result", []):
                            self._offset = update["update_id"] + 1
                            await self._process(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Polling: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False

    async def _process(self, update: dict):
        try:
            msg      = update.get("message", {})
            chat_id  = str(msg.get("chat", {}).get("id", ""))
            text     = msg.get("text", "").strip().lower().split("@")[0]
            username = msg.get("from", {}).get("username", "")
            if not chat_id or not text:
                return
            if text == "/start":
                await self.handle_start(chat_id, username)
            elif text == "/subscribe":
                await self.handle_subscribe(chat_id, username)
            elif text == "/unsubscribe":
                await self.handle_unsubscribe(chat_id)
            elif text in ("/stats", "/performance"):
                await self.handle_stats(chat_id)
            elif text == "/help":
                await self.handle_help(chat_id)
            elif text == "/leaderboard":
                await self.handle_leaderboard(chat_id)
            elif text.startswith("/pairstats"):
                parts = text.split()
                if len(parts) >= 2:
                    await self.handle_pairstats(chat_id, parts[1].upper())
                else:
                    await self.send(chat_id, "Usage: /pairstats XAUUSD")
            elif text == "/performance":
                await self.handle_performance(chat_id)
            elif text == "/modelstats":
                await self.handle_modelstats(chat_id)
            else:
                await self.send(chat_id, "Type /help for commands.")
        except Exception as e:
            logger.debug(f"Process update: {e}")

    def get_stats(self) -> dict:
        total = self._win_count + self._loss_count
        return {
            "subscribers":  self._store.count(),
            "signals_sent": self._signal_count,
            "win_rate":     round(self._win_count / total, 3) if total else None,
        }

signal_bot = TelegramSignalBot()
