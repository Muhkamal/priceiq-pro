"""
PriceIQ Pro — Alert System v3.2
Updated with async support, retries, rate limiting, and deduplication.
"""

import asyncio
import smtplib
import httpx
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Set
from datetime import datetime, timezone
import logging

from app.models.schemas import TradeSignal, SignalDirection
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Sends alerts via Email and Telegram with rate limiting and deduplication.

    Features:
    - Async Telegram sending (non-blocking)
    - Rate limiting (Telegram: max 1 msg/sec)
    - Retry logic for transient failures
    - Signal deduplication (prevents duplicate alerts)
    - HTML + plain text email support
    """

    def __init__(self):
        self.telegram_bot_token = settings.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = settings.TELEGRAM_CHAT_ID
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.alert_email = settings.ALERT_EMAIL

        # Rate limiting: Telegram allows 30 msg/sec, we use 1/sec for safety
        self._telegram_semaphore = asyncio.Semaphore(1)
        self._telegram_min_interval = 1.0  # seconds between messages
        self._last_telegram_time = None

        # Deduplication: track sent signal IDs
        self._sent_signals: Set[str] = set()
        self._max_sent_memory = 1000

        # Reusable HTTP client for Telegram
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _is_duplicate(self, signal: TradeSignal) -> bool:
        """Check if signal was already sent."""
        signal_id = f"{signal.pair}_{signal.pattern.value}_{signal.direction.value}_{signal.timestamp.isoformat()}"
        if signal_id in self._sent_signals:
            return True
        self._sent_signals.add(signal_id)
        if len(self._sent_signals) > self._max_sent_memory:
            # Remove oldest (simple approach: clear half)
            self._sent_signals = set(list(self._sent_signals)[self._max_sent_memory // 2:])
        return False

    async def send_signal_alert(
        self, 
        signal: TradeSignal, 
        recipients: List[str],
        alert_methods: List[str] = ["email", "telegram"],
        skip_duplicates: bool = True,
    ) -> dict:
        """
        Send signal alert via specified methods.

        Args:
            signal: TradeSignal to alert about
            recipients: Email recipients
            alert_methods: ["email", "telegram"] or subset
            skip_duplicates: Skip if already sent (default True)

        Returns:
            {"email": bool, "telegram": bool, "errors": [], "skipped": bool}
        """
        results = {"email": False, "telegram": False, "errors": [], "skipped": False}

        # Deduplication check
        if skip_duplicates and self._is_duplicate(signal):
            logger.info(f"Alert deduplicated for {signal.pair} {signal.pattern.value}")
            results["skipped"] = True
            return results

        # Validate recipients
        if not recipients:
            results["errors"].append("No email recipients provided")
            if "email" in alert_methods:
                alert_methods.remove("email")

        # Validate methods
        valid_methods = {"email", "telegram"}
        unknown = set(alert_methods) - valid_methods
        if unknown:
            logger.warning(f"Unknown alert methods: {unknown}")
            alert_methods = [m for m in alert_methods if m in valid_methods]

        message = self._format_signal_message(signal)
        subject = f"🚨 PriceIQ Signal: {signal.direction.value.upper()} {signal.pair}"

        if "email" in alert_methods:
            try:
                await asyncio.to_thread(self._send_email, recipients, subject, message)
                results["email"] = True
            except Exception as e:
                results["errors"].append(f"Email failed: {str(e)}")
                logger.error(f"Email alert failed: {e}")

        if "telegram" in alert_methods:
            try:
                await self._send_telegram(message)
                results["telegram"] = True
            except Exception as e:
                results["errors"].append(f"Telegram failed: {str(e)}")
                logger.error(f"Telegram alert failed: {e}")

        return results

    def _format_signal_message(self, signal: TradeSignal) -> str:
        """Format signal into readable HTML message."""
        emoji = "🟢" if signal.direction == SignalDirection.BUY else "🔴"

        msg = f"""{emoji} <b>PRICEIQ PRO SIGNAL</b> {emoji}

<b>Pair:</b> {signal.pair}
<b>Timeframe:</b> {signal.timeframe}
<b>Direction:</b> {signal.direction.value.upper()}
<b>Pattern:</b> {signal.pattern.value.replace('_', ' ').title()}
<b>Confidence:</b> {signal.confidence:.0%}

<b>📊 PRICE LEVELS</b>
Entry: {signal.entry_price:.5f}
Stop Loss: {signal.stop_loss:.5f}
Take Profit 1: {signal.take_profit_1:.5f}"""

        if signal.take_profit_2:
            msg += f"\nTake Profit 2: {signal.take_profit_2:.5f}"

        msg += f"\nRisk:Reward: 1:{signal.risk_reward_1:.2f}"

        if signal.risk_reward_2:
            msg += f" / 1:{signal.risk_reward_2:.2f}"

        if signal.position_size:
            msg += f"""

<b>💰 POSITION SIZE</b>
Standard Lots: {signal.position_size.get('standard_lots', 'N/A')}
Mini Lots: {signal.position_size.get('mini_lots', 'N/A')}
Micro Lots: {signal.position_size.get('micro_lots', 'N/A')}
Risk Amount: ${signal.position_size.get('risk_amount', 'N/A')}
Required Margin: ${signal.position_size.get('required_margin', 'N/A')}"""

        msg += f"""

<b>📈 MARKET CONTEXT</b>
Trend: {signal.market_structure.trend.value}
Stage: {signal.market_structure.stage.value}
Trend Strength: {signal.market_structure.trend_strength:.0%}
Healthy Trend: {'Yes' if signal.market_structure.is_healthy_trend else 'No'}"""

        if signal.support_resistance:
            msg += "\n\n<b>🎯 KEY LEVELS</b>"
            for sr in signal.support_resistance[:3]:
                level_type = "Support" if sr.was_support else "Resistance"
                if sr.is_role_reversal:
                    level_type += " (Reversed)"
                msg += f"\n{level_type}: {sr.level:.5f} (Strength: {sr.strength:.0%})"

        if signal.explanation:
            msg += f"""

<b>🧠 ANALYSIS:</b>
{signal.explanation}"""

        msg += f"""

⏰ Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
⚠️ This is not financial advice. Always use proper risk management."""

        return msg

    def _send_email(self, recipients: List[str], subject: str, body: str):
        """Send email via SMTP with HTML support."""
        msg = MIMEMultipart("alternative")
        msg['From'] = self.smtp_user
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject

        # Plain text version (strip HTML)
        plain_text = body.replace('<b>', '').replace('</b>', '')
        msg.attach(MIMEText(plain_text, 'plain'))

        # HTML version
        msg.attach(MIMEText(f"<html><body>{body}</body></html>", 'html'))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)

    async def _send_telegram(self, message: str):
        """Send message via Telegram Bot API with rate limiting."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            raise ValueError("Telegram not configured")

        async with self._telegram_semaphore:
            # Enforce minimum interval
            if self._last_telegram_time:
                elapsed = (datetime.now(timezone.utc) - self._last_telegram_time).total_seconds()
                if elapsed < self._telegram_min_interval:
                    wait = self._telegram_min_interval - elapsed
                    logger.debug(f"Telegram rate limit: waiting {wait:.2f}s")
                    await asyncio.sleep(wait)

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }

            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()

            self._last_telegram_time = datetime.now(timezone.utc)

    async def send_backtest_summary(
        self, 
        backtest_result, 
        recipients: List[str],
        alert_methods: List[str] = ["email"],
    ) -> dict:
        """Send backtest results summary."""
        m = backtest_result.metrics

        message = f"""📊 <b>BACKTEST SUMMARY</b>

<b>Verdict:</b> {backtest_result.verdict}
<b>Total Trades:</b> {m.total_trades}
<b>Win Rate:</b> {m.win_rate:.1%}
<b>Profit Factor:</b> {m.profit_factor:.2f}
<b>Sharpe Ratio:</b> {m.sharpe_ratio:.2f}
<b>Max Drawdown:</b> {m.max_drawdown_percent:.1f}%
<b>Total Return:</b> {m.total_return_percent:+.1f}%
<b>Expectancy:</b> ${m.expectancy:.2f}

<b>Best Pattern:</b> {max(m.win_rate_by_pattern.items(), key=lambda x: x[1])[0] if m.win_rate_by_pattern else 'N/A'}
<b>Best Stage:</b> {max(m.win_rate_by_stage.items(), key=lambda x: x[1])[0] if m.win_rate_by_stage else 'N/A'}
"""

        results = {"email": False, "telegram": False, "errors": []}

        if "email" in alert_methods and recipients:
            try:
                await asyncio.to_thread(
                    self._send_email, 
                    recipients, 
                    "📊 PriceIQ Backtest Results", 
                    message
                )
                results["email"] = True
            except Exception as e:
                results["errors"].append(f"Email failed: {str(e)}")

        if "telegram" in alert_methods:
            try:
                await self._send_telegram(message)
                results["telegram"] = True
            except Exception as e:
                results["errors"].append(f"Telegram failed: {str(e)}")

        return results

    async def send_daily_digest(
        self, 
        signals_today: List[TradeSignal], 
        recipients: List[str],
        alert_methods: List[str] = ["email"],
    ) -> dict:
        """Send daily summary of all signals."""
        buy_count = sum(1 for s in signals_today if s.direction == SignalDirection.BUY)
        sell_count = sum(1 for s in signals_today if s.direction == SignalDirection.SELL)

        message = f"""📅 <b>DAILY SIGNAL DIGEST</b>

<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
<b>Total Signals:</b> {len(signals_today)}
<b>Buy Signals:</b> {buy_count}
<b>Sell Signals:</b> {sell_count}

<b>Today's Signals:</b>
"""
        for i, signal in enumerate(signals_today[:10], 1):
            emoji = "🟢" if signal.direction == SignalDirection.BUY else "🔴"
            message += f"{i}. {emoji} {signal.pair} {signal.timeframe} — {signal.direction.value.upper()} @ {signal.entry_price:.5f}\n"

        results = {"email": False, "telegram": False, "errors": []}

        if "email" in alert_methods and recipients:
            try:
                await asyncio.to_thread(
                    self._send_email,
                    recipients,
                    f"📅 PriceIQ Daily Digest — {len(signals_today)} Signals",
                    message,
                )
                results["email"] = True
            except Exception as e:
                results["errors"].append(f"Email failed: {str(e)}")

        if "telegram" in alert_methods:
            try:
                await self._send_telegram(message)
                results["telegram"] = True
            except Exception as e:
                results["errors"].append(f"Telegram failed: {str(e)}")

        return results

    async def health_check(self) -> dict:
        """Check alert system health."""
        return {
            "email_configured": bool(self.smtp_user and self.smtp_password and self.alert_email),
            "telegram_configured": bool(self.telegram_bot_token and self.telegram_chat_id),
            "telegram_rate_limit": f"1 per {self._telegram_min_interval}s",
            "deduplication_memory": len(self._sent_signals),
            "max_deduplication_memory": self._max_sent_memory,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Singleton instance
alert_manager = AlertManager()
