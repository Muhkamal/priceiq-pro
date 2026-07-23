"""
app/services/telegram_bot.py
PriceIQ Pro — Telegram Alert Bot
Sends signal alerts, daily digests, and status updates
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    """Sends messages to Telegram via Bot API."""

    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)

        if self.enabled:
            logger.info("Telegram bot initialized ✓")
        else:
            logger.warning("Telegram bot not configured — alerts disabled")

    def _url(self, method: str) -> str:
        return TELEGRAM_API.format(token=self.token, method=method)

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a plain message."""
        if not self.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._url("sendMessage"), json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": parse_mode,
                })
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    # ----------------------------------------------------------
    # SIGNAL ALERT
    # ----------------------------------------------------------

    async def send_signal_alert(self, signal: Dict[str, Any]) -> bool:
        """
        Send a trade signal alert.
        Works with both TradeSignal objects and plain dicts.
        """
        if not self.enabled:
            return False

        try:
            # Support both object attributes and dict keys
            def get(key, default="N/A"):
                if isinstance(signal, dict):
                    return signal.get(key, default)
                return getattr(signal, key, default)

            direction   = str(get("direction", "")).replace("SignalDirection.", "")
            pair        = get("pair", "UNKNOWN")
            entry       = get("entry_price", 0)
            sl          = get("stop_loss", 0)
            tp          = get("take_profit_1", 0)
            rr          = get("risk_reward_1", 0)
            confidence  = get("confidence", 0)
            pattern     = str(get("pattern", "")).replace("PatternType.", "").replace("_", " ").title()
            timeframe   = get("timeframe", "1h")
            explanation = get("explanation", "")

            # Direction emoji
            emoji     = "🟢" if "BUY" in direction.upper() else "🔴"
            dir_label = "BUY  📈" if "BUY" in direction.upper() else "SELL 📉"

            # Confidence bar
            conf_pct  = int(float(confidence) * 100)
            filled    = int(conf_pct / 10)
            conf_bar  = "█" * filled + "░" * (10 - filled)

            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            message = (
                f"{emoji} <b>SIGNAL ALERT — {pair}</b>\n"
                f"{'─' * 30}\n"
                f"📊 <b>Direction:</b>  {dir_label}\n"
                f"🕯 <b>Pattern:</b>   {pattern}  ({timeframe})\n"
                f"\n"
                f"💰 <b>Entry:</b>      <code>{float(entry):.5f}</code>\n"
                f"🛑 <b>Stop Loss:</b>  <code>{float(sl):.5f}</code>\n"
                f"🎯 <b>Take Profit:</b> <code>{float(tp):.5f}</code>\n"
                f"⚖️ <b>R:R Ratio:</b>  {float(rr):.1f}:1\n"
                f"\n"
                f"🧠 <b>Confidence:</b> {conf_pct}%\n"
                f"    [{conf_bar}]\n"
                f"\n"
                f"📝 {explanation}\n"
                f"{'─' * 30}\n"
                f"🕐 {now}\n"
                f"⚠️ <i>Paper trading — not financial advice</i>"
            )

            return await self.send_message(message)

        except Exception as e:
            logger.error(f"Failed to format signal alert: {e}")
            return False

    # ----------------------------------------------------------
    # DAILY DIGEST
    # ----------------------------------------------------------

    async def send_daily_digest(
        self,
        total_signals: int,
        buy_signals: int,
        sell_signals: int,
        top_pairs: list,
        avg_confidence: float,
        account_balance: float = 0,
    ) -> bool:
        """Send end-of-day summary."""
        if not self.enabled:
            return False

        pairs_str = ", ".join(top_pairs) if top_pairs else "None"
        date_str  = datetime.utcnow().strftime("%A %d %B %Y")

        message = (
            f"📊 <b>DAILY DIGEST — {date_str}</b>\n"
            f"{'─' * 30}\n"
            f"📈 Buy signals:    <b>{buy_signals}</b>\n"
            f"📉 Sell signals:   <b>{sell_signals}</b>\n"
            f"🔢 Total signals:  <b>{total_signals}</b>\n"
            f"\n"
            f"🏆 Top pairs:  {pairs_str}\n"
            f"🧠 Avg confidence: {int(avg_confidence * 100)}%\n"
        )

        if account_balance:
            message += f"💼 Account balance: <b>${account_balance:,.2f}</b>\n"

        message += (
            f"{'─' * 30}\n"
            f"🤖 PriceIQ Pro v3.2 | priceiq-pro.onrender.com"
        )

        return await self.send_message(message)

    # ----------------------------------------------------------
    # SYSTEM ALERTS
    # ----------------------------------------------------------

    async def send_startup_message(self) -> bool:
        """Notify when server starts."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return await self.send_message(
            f"✅ <b>PriceIQ Pro is Online</b>\n"
            f"Server started at {now}\n"
            f"Scanning: EURUSD, GBPUSD, USDJPY, XAUUSD, AUDUSD, GBPJPY\n"
            f"Next hourly scan at :01 past the hour"
        )

    async def send_error_alert(self, error: str) -> bool:
        """Notify on critical errors."""
        return await self.send_message(
            f"🚨 <b>ERROR ALERT</b>\n"
            f"<code>{error[:500]}</code>\n"
            f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
        )

    async def send_no_signal_update(self, pairs_checked: int) -> bool:
        """Quiet hourly update — no signal found."""
        now = datetime.utcnow().strftime("%H:%M UTC")
        return await self.send_message(
            f"🔍 <b>Scan Complete</b> — {now}\n"
            f"No signals on {pairs_checked} pairs this hour.\n"
            f"<i>Markets quiet — waiting for setup...</i>"
        )


# Singleton instance
telegram = TelegramBot()
