"""Live SMC scanner - alerts only, NO execution.
Render start command: uvicorn smc_bot.main:app --host 0.0.0.0 --port $PORT
Telegram: /close WIN 2.4 DOL (reply to alert) | /close XAUUSD WIN 2.4 DOL | /pending"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import pandas as pd
from fastapi import FastAPI

from .config import load_config
from .core.context import ContextEngine
from .core.pairs import PAIR_SPECS, get_spread
from .data.fetcher import fetch_m5
from .engine.alerts import TelegramSender, passes_rr_gate, send_smc_alert, MIN_RR_TO_DOL
from .engine.journal import ExpectancyJournal, VALID_OUTCOMES, VALID_EXIT_REASONS
from .entries.choch_no_idm import ChoChNoIDM
from .entries.scm import SingleCandleMitigation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smc_bot")

MODULE_REGISTRY = {"choch_no_idm": ChoChNoIDM, "scm": SingleCandleMitigation}

CONFIG = load_config()
JOURNAL = ExpectancyJournal(db_path=os.environ.get("SMC_JOURNAL_DB", "smc_journal.db"))
SENDER = TelegramSender()
CTX_ENGINE = ContextEngine(swing_length=CONFIG.params.swing_length,
                           kill_zones_utc=CONFIG.conditions.kill_zones)
MODULES = [MODULE_REGISTRY[n]() for n in CONFIG.entry_modules if n in MODULE_REGISTRY]
BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "100.0"))
_last_report_week = None

async def scan_pair(pair: str):
    spread = get_spread(pair)
    if spread is None:
        logger.error(f"No spread configured for {pair} - refusing to scan")
        return
    df_m5 = await fetch_m5(pair, limit=1000)
    if df_m5 is None or len(df_m5) < 120:
        return
    now = datetime.now(timezone.utc)
    df_m15 = df_m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    valid_m15 = df_m15[df_m15.index + pd.Timedelta(minutes=15) <= now]
    if valid_m15.empty:
        return
    ctx = CTX_ENGINE.build(valid_m15, now, pair=pair)
    if ctx is None:
        return
    signal = None
    for mod in MODULES:
        signal = mod.check(df_m5, ctx, CONFIG)
        if signal:
            signal["pair"] = pair
            signal["ctx"] = {"bias": ctx.bias, "zone": ctx.zone, "pdh": ctx.pdh, "pdl": ctx.pdl}
            break
    if not signal:
        return
    bar_ts = df_m5.index[-1].isoformat()
    if JOURNAL.has_signal_for_bar(pair, bar_ts):
        logger.debug(f"Idempotency guard: {pair} already logged on {bar_ts}")
        return
    direction = signal["direction"]
    adj_ref = signal["ref_price"] + spread if direction == "BUY" else signal["ref_price"] - spread
    signal["ref_price"] = adj_ref
    ok, rr = passes_rr_gate(adj_ref, signal["sl"], signal["tp"], direction)
    if not ok:
        JOURNAL.log_signal(signal, skipped=True, skip_reason=f"RR {rr:.2f} < {MIN_RR_TO_DOL}",
                           timestamp_override=bar_ts)
        logger.info(f"Skipped {pair}: RR {rr:.2f}")
        return
    signal["rr_to_dol"] = rr
    row_id = JOURNAL.log_signal(signal, timestamp_override=bar_ts)
    mid = await send_smc_alert(SENDER, signal, BALANCE, CONFIG)
    if mid and row_id:
        JOURNAL.set_alert_msg_id(row_id, mid)

async def maybe_weekly_report():
    global _last_report_week
    now = datetime.now(timezone.utc)
    week = now.isocalendar()[:2]
    if now.isoweekday() == 7 and now.hour >= 18 and week != _last_report_week:
        _last_report_week = week
        await SENDER.send_owner_dm(JOURNAL.get_weekly_report())

async def scanner_loop():
    logger.info(f"SMC scanner started: markets={CONFIG.markets} modules={CONFIG.entry_modules}")
    while True:
        for pair in CONFIG.markets:
            try:
                await scan_pair(pair)
            except Exception as e:
                logger.error(f"Scan error {pair}: {e}")
        try:
            await maybe_weekly_report()
        except Exception as e:
            logger.error(f"Weekly report error: {e}")
        now = datetime.now(timezone.utc)
        secs = 300 - (now.minute % 5) * 60 - now.second + 5
        await asyncio.sleep(max(10, secs))

async def _reply(client, chat_id, text):
    if not SENDER.token or not chat_id:
        return
    try:
        await client.post(f"https://api.telegram.org/bot{SENDER.token}/sendMessage",
                          json={"chat_id": chat_id, "text": text})
    except Exception as e:
        logger.error(f"Reply failed: {e}")

async def handle_update(client, up):
    msg = up.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if not text.startswith("/"):
        return
    parts = text.split()
    cmd = parts[0].lower()
    if cmd == "/pending":
        rows = JOURNAL.list_open_rows()
        body = ("Open rows:\n" + "\n".join(f"#{r[0]} {r[1]} {r[2]} {r[3][:16]}" for r in rows)
                if rows else "No open journal rows.")
        await _reply(client, chat_id, body)
        return
    if cmd == "/close":
        args = parts[1:]
        row_id = None
        reply_to = (msg.get("reply_to_message") or {}).get("message_id")
        if reply_to:
            row_id = JOURNAL.get_row_id_by_alert_msg(reply_to)
        if row_id is None and len(args) >= 4 and args[0].upper() in PAIR_SPECS:
            open_row = JOURNAL.get_last_open_row(args[0].upper())
            row_id = open_row[0] if open_row else None
            args = args[1:]
        if row_id is None or len(args) < 3:
            await _reply(client, chat_id,
                         "Usage: reply to an alert with /close WIN 2.4 DOL  or  /close XAUUSD WIN 2.4 DOL")
            return
        outcome, reason = args[0].upper(), args[2].upper()
        try:
            pnl = float(args[1])
        except ValueError:
            await _reply(client, chat_id, "pnl_r must be a number: /close WIN 2.4 DOL")
            return
        if outcome not in VALID_OUTCOMES or outcome == "SKIPPED":
            await _reply(client, chat_id, f"outcome must be one of {sorted(VALID_OUTCOMES - {'SKIPPED'})}")
            return
        if reason not in VALID_EXIT_REASONS:
            await _reply(client, chat_id, f"exit_reason must be one of {sorted(VALID_EXIT_REASONS)}")
            return
        ok = JOURNAL.update_trade_outcome(row_id, outcome, pnl, reason)
        await _reply(client, chat_id,
                     f"Row #{row_id} closed: {outcome} {pnl:+.2f}R ({reason})" if ok
                     else f"Row #{row_id} is not open (already closed?). Use /pending.")

async def telegram_command_loop():
    offset = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                r = await client.get(
                    f"https://api.telegram.org/bot{SENDER.token}/getUpdates",
                    params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                )
                if r.status_code == 200:
                    for up in r.json().get("result", []):
                        offset = up["update_id"] + 1
                        await handle_update(client, up)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Polling error: {e}")
            await asyncio.sleep(5)
        await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(scanner_loop())
    t2 = asyncio.create_task(telegram_command_loop())
    yield
    t1.cancel()
    t2.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "mode": "smc_scanner"}
