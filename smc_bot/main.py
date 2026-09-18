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
from .data.deriv_fetcher import fetch_deriv_m5
from .core.markets import get_profile
from .engine.alerts import TelegramSender, passes_rr_gate, send_smc_alert, MIN_RR_TO_DOL
from .engine.journal import ExpectancyJournal, VALID_OUTCOMES, VALID_EXIT_REASONS
from .entries.choch_no_idm import ChoChNoIDM
from .entries.scm import SingleCandleMitigation

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
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
_last_briefing_date = None
_scan_count = 0
_alert_count = 0
_skip_count = 0
_ctx_none_count = 0
_skip_zone = 0
_skip_killzone = 0
_skip_module = 0
_skip_rr = 0


async def send_morning_briefing():
    """Sends a daily market narrative at 07:00 UTC (London Open)."""
    msg = "☀️ <b>GOOD MORNING! Daily SMC Briefing</b>\n\n"
    msg += "<i>Here is the institutional narrative for today's session:</i>\n\n"

    for pair in CONFIG.markets:
        try:
            prof = get_profile(pair)
            if prof and prof.source == 'deriv':
                df_m5 = await fetch_deriv_m5(prof.deriv_symbol, limit=1000,
                                             granularity=prof.granularity)
            else:
                df_m5 = await fetch_m5(pair, limit=1000)

            if df_m5 is None or len(df_m5) < 120:
                continue
            if "volume" not in df_m5.columns:
                df_m5["volume"] = 1000

            now = datetime.now(timezone.utc)
            df_m15 = df_m5.resample("15min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            valid_m15 = df_m15[df_m15.index + pd.Timedelta(minutes=15) <= now]

            ctx = CTX_ENGINE.build(
                valid_m15, now, pair=pair, profile=get_profile(pair),
                compute_poi=bool(CONFIG.conditions.require_unmitigated_zone),
            )
            if ctx:
                emoji = "🟢" if ctx.bias == "BULLISH" else "🔴"
                msg += f"<b>{pair}</b> {emoji} <b>{ctx.bias}</b>\n"
                msg += f"• Current Zone: <b>{ctx.zone}</b>\n"
                msg += f"• Equilibrium (50%): <code>{ctx.equilibrium:.5f}</code>\n"
                msg += f"• Draw on Liquidity (DOL): <code>{('none' if ctx.dol is None else format(ctx.dol, '.5f'))}</code>\n"
                msg += f"• PDH: <code>{ctx.pdh:.5f}</code> | PDL: <code>{ctx.pdl:.5f}</code>\n\n"
            else:
                msg += f"<b>{pair}</b> ⚠️ Structure mapping (Waiting for clear swings)\n\n"
        except Exception as e:
            logger.error(f"Briefing error for {pair}: {e}")
            msg += f"<b>{pair}</b> ⚠️ Error fetching data\n\n"

    msg += "🎯 <b>Today's Gameplan:</b>\n"
    msg += "• Only look for BUYS if price taps DISCOUNT (Blue) zones.\n"
    msg += "• Only look for SELLS if price taps PREMIUM (Red) zones.\n"
    msg += "• Wait for the sweep + CHoCH on M5 before entering.\n"
    msg += "• Risk: 0.5% per trade."

    await SENDER.send_owner_dm(msg)
    logger.info("☀️ Morning Briefing sent to Telegram")


async def scan_pair(pair: str):
    global _skip_count, _ctx_none_count, _skip_zone, _skip_killzone, _skip_module
    spread = get_spread(pair)
    if spread is None:
        return
    prof = get_profile(pair)
    if prof and prof.source == 'deriv':
        df_m5 = await fetch_deriv_m5(prof.deriv_symbol, limit=1000,
                                     granularity=prof.granularity)
    else:
        df_m5 = await fetch_m5(pair, limit=1000)
    if df_m5 is None or len(df_m5) < 120:
        return
    if "volume" not in df_m5.columns:
        df_m5["volume"] = 1000

    now = datetime.now(timezone.utc)
    df_m15 = df_m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    valid_m15 = df_m15[df_m15.index + pd.Timedelta(minutes=15) <= now]
    if valid_m15.empty:
        return

    ctx = CTX_ENGINE.build(
        valid_m15, now, pair=pair, profile=get_profile(pair),
        compute_poi=bool(CONFIG.conditions.require_unmitigated_zone),
    )
    if ctx is None:
        _ctx_none_count += 1
        return

    if ctx.zone == "EQUILIBRIUM" or (ctx.bias == "BULLISH" and ctx.zone != "DISCOUNT") or (ctx.bias == "BEARISH" and ctx.zone != "PREMIUM"):
        _skip_zone += 1
    if not ctx.in_kill_zone:
        _skip_killzone += 1

    # Log skip reasons for observability
    if ctx.zone == "EQUILIBRIUM" or (ctx.bias == "BULLISH" and ctx.zone != "DISCOUNT") or (ctx.bias == "BEARISH" and ctx.zone != "PREMIUM"):
        _skip_zone += 1
    if not ctx.in_kill_zone:
        _skip_killzone += 1

    logger.info(
        f"[{pair}] bias={ctx.bias} | zone={ctx.zone} | kill_zone={ctx.in_kill_zone} | "
        f"eq={ctx.equilibrium:.5f} | DOL={('none' if ctx.dol is None else format(ctx.dol, '.5f'))} | PDH={ctx.pdh:.5f} PDL={ctx.pdl:.5f} | "
        f"price={df_m5['close'].iloc[-1]:.5f}"
    )

    signal = None
    for mod in MODULES:
        signal = mod.check(df_m5, ctx, CONFIG)
        if signal:
            signal["pair"] = pair
            signal["ctx"] = {"bias": ctx.bias, "zone": ctx.zone,
                             "pdh": ctx.pdh, "pdl": ctx.pdl}
            signal["stake_based"] = bool(prof and prof.stake_based)
            break
    if not signal:
        _skip_module += 1
        return

    bar_ts = df_m5.index[-1].isoformat()
    if JOURNAL.has_signal_for_bar(pair, bar_ts):
        return

    direction = signal["direction"]
    adj_ref = (signal["ref_price"] + spread if direction == "BUY"
               else signal["ref_price"] - spread)
    signal["ref_price"] = adj_ref
    ok, rr = passes_rr_gate(adj_ref, signal["sl"], signal["tp"], direction)
    if not ok:
        _skip_count += 1
        JOURNAL.log_signal(signal, skipped=True,
                           skip_reason=f"RR {rr:.2f} < {MIN_RR_TO_DOL}",
                           timestamp_override=bar_ts)
        logger.info(f"[{pair}] ⛔ SKIPPED: RR {rr:.2f} < {MIN_RR_TO_DOL}")
        return

    signal["rr_to_dol"] = rr
    row_id = JOURNAL.log_signal(signal, timestamp_override=bar_ts)
    logger.info(
        f"[{pair}] 🎯 SIGNAL FIRED: {signal['module']} {direction} RR={rr:.2f} | "
        f"entry={adj_ref:.5f} sl={signal['sl']:.5f} tp={signal['tp']:.5f}"
    )
    mid = await send_smc_alert(SENDER, signal, BALANCE, CONFIG)
    if mid and row_id:
        JOURNAL.set_alert_msg_id(row_id, mid)
        _alert_count += 1


async def scanner_loop():
    global _scan_count, _last_briefing_date, _last_diag_hour
    logger.info(f"🚀 SMC scanner started: markets={CONFIG.markets}")
    while True:
        now = datetime.now(timezone.utc)

        if now.hour == 7 and now.minute < 5 and _last_briefing_date != now.date():
            try:
                await send_morning_briefing()
                _last_briefing_date = now.date()
            except Exception as e:
                logger.error(f"Briefing error: {e}")

        for pair in CONFIG.markets:
            try:
                _scan_count += 1
                await scan_pair(pair)
            except Exception as e:
                logger.error(f"Scan error {pair}: {e}")

        if _last_diag_hour != now.hour:
            _last_diag_hour = now.hour
            logger.info(
                f"📊 hourly diag: scans={_scan_count} alerts={_alert_count} "
                f"rr_skips={_skip_count} ctx_none={_ctx_none_count} "
                f"zone_skips={_skip_zone} kz_skips={_skip_killzone} module_miss={_skip_module}"
            )
        secs = 300 - (now.minute % 5) * 60 - now.second + 5
        await asyncio.sleep(max(10, secs))


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
        if rows:
            body = "Open rows:\n" + "\n".join(
                f"#{r[0]} {r[1]} {r[2]} {r[3][:16]}" for r in rows
            )
        else:
            body = "No open journal rows."
        await client.post(
            f"https://api.telegram.org/bot{SENDER.token}/sendMessage",
            json={"chat_id": chat_id, "text": body},
        )

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
            await client.post(
                f"https://api.telegram.org/bot{SENDER.token}/sendMessage",
                json={"chat_id": chat_id,
                      "text": "Usage: /close WIN 2.4 DOL  (or reply to an alert)"},
            )
            return
        outcome, reason = args[0].upper(), args[2].upper()
        try:
            pnl = float(args[1])
        except ValueError:
            return
        if outcome not in VALID_OUTCOMES or outcome == "SKIPPED":
            return
        if reason not in VALID_EXIT_REASONS:
            return
        ok = JOURNAL.update_trade_outcome(row_id, outcome, pnl, reason)
        reply_text = (f"✅ Row #{row_id} closed: {outcome} {pnl:+.2f}R ({reason})"
                      if ok else f"❌ Row #{row_id} not open.")
        await client.post(
            f"https://api.telegram.org/bot{SENDER.token}/sendMessage",
            json={"chat_id": chat_id, "text": reply_text},
        )


async def telegram_command_loop():
    offset = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                r = await client.get(
                    f"https://api.telegram.org/bot{SENDER.token}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                if r.status_code == 200:
                    for up in r.json().get("result", []):
                        offset = up["update_id"] + 1
                        await handle_update(client, up)
        except Exception as e:
            logger.error(f"Polling error: {e}")
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
