import math
import os
import logging
from typing import Tuple
import httpx
from ..core.pairs import get_spec

logger = logging.getLogger(__name__)

MIN_RR_TO_DOL = 2.0
DEFAULT_RISK_PCT = 0.005

def calculate_lot_size(balance, entry, sl, pair, risk_pct=DEFAULT_RISK_PCT) -> float:
    spec = get_spec(pair)
    if spec is None:
        return 0.0
    risk_amount = balance * risk_pct
    sl_pips = abs(entry - sl) / spec.pip_size
    if sl_pips <= 0:
        return 0.0
    raw_lots = risk_amount / (sl_pips * spec.usd_per_pip_per_lot)
    floored = math.floor(raw_lots * 100) / 100
    return floored if floored >= 0.01 else 0.0

def passes_rr_gate(entry, sl, dol, direction) -> Tuple[bool, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        return False, 0.0
    reward = (dol - entry) if direction == "BUY" else (entry - dol)
    rr = reward / risk
    return rr >= MIN_RR_TO_DOL, rr

def format_smc_checklist(signal, balance, config) -> str:
    pair = signal.get("pair", "UNKNOWN")
    direction = signal.get("direction", "BUY")
    ref = signal.get("ref_price", 0.0)
    sl = signal.get("sl", 0.0)
    tp = signal.get("tp", 0.0)
    rr = signal.get("rr_to_dol", 0.0)
    ctx = signal.get("ctx", {})
    risk_pct = config.risk.pct_per_trade / 100.0
    risk_usd = balance * risk_pct
    lots = calculate_lot_size(balance, ref, sl, pair, risk_pct)
    risk_dist = abs(ref - sl)
    tp4 = ref + risk_dist * 4 if direction == "BUY" else ref - risk_dist * 4
    tp10 = ref + risk_dist * 10 if direction == "BUY" else ref - risk_dist * 10
    if signal.get("stake_based"):
        size_line = f"Stake: <b>${risk_usd:.2f}</b> (Deriv stake = your 0.5% risk amount)"
    else:
        size_line = (f"Lot Size: <b>{lots:.2f}</b>" if lots > 0
                     else "Lot Size: <b>BELOW BROKER MIN - 0.01 would exceed 0.5% risk. SKIP or accept higher risk.</b>")
    emoji = "🟢" if direction == "BUY" else "🔴"
    return (
        f"🚨 <b>SMC SCANNER: {pair}</b> {emoji} <b>{direction}</b>\n"
        f"Module: {signal.get('module')}\n"
        f"Context: {ctx.get('bias')} bias | {ctx.get('zone')} zone | RR to DOL: {rr:.2f}\n\n"
        f"📊 <b>Levels (enter at NEXT M5 open)</b>\n"
        f"Ref: <code>{ref:.5f}</code> | SL: <code>{sl:.5f}</code> | DOL: <code>{tp:.5f}</code>\n\n"
        f"📋 <b>MT5 CHECKLIST - verify before clicking:</b>\n"
        f"1️⃣ Price in correct Fib zone ({ctx.get('zone')})?\n"
        f"2️⃣ Liquidity swept (prev candle / PDH / PDL)?\n"
        f"3️⃣ Close-through CHoCH or mitigation confirmed?\n"
        f"4️⃣ Unmitigated OB/FVG at entry?\n\n"
        f"💵 Risk: <b>${risk_usd:.2f}</b> ({config.risk.pct_per_trade}%)\n{size_line}\n\n"
        f"🛡 <b>Management (hard rules)</b>\n"
        f"• 4R (<code>{tp4:.5f}</code>): close 20%, SL to BE\n"
        f"• 10R (<code>{tp10:.5f}</code>): close 50% (pro-trend)\n"
        f"• Runner target: next liquidity pool (PDH/PDL, weak H/L)\n"
        f"• Time-stop: {config.invalidation.time_stop_bars} M5 bars\n\n"
        f"⚠️ {signal.get('reason', '')}"
    )

class TelegramSender:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "") or os.environ.get("OWNER_CHAT_ID", "")

    async def send_owner_dm(self, text: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram env missing - alert not sent")
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": str(self.chat_id), "text": text, "parse_mode": "HTML"},
                )
                if r.status_code == 200:
                    return int(r.json().get("result", {}).get("message_id", 0)) or None
                return None
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return None

async def send_smc_alert(sender, signal, balance, config):
    msg = format_smc_checklist(signal, balance, config)
    mid = await sender.send_owner_dm(msg)
    if mid:
        logger.info(f"SMC alert sent: {signal.get('pair')} {signal.get('direction')} (msg {mid})")
    return mid
