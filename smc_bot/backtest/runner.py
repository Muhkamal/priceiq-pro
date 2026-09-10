"""Run from repo root: python -m smc_bot.backtest.runner data/XAUUSD_M5.csv --pair XAUUSD --db backtest_journal.db"""
import argparse
import statistics
import pandas as pd
from typing import Dict, List, Any

from ..config import load_config
from ..core.context import ContextEngine
from ..core.pairs import get_spread
from ..engine.journal import ExpectancyJournal
from ..engine.alerts import MIN_RR_TO_DOL
from ..entries.choch_no_idm import ChoChNoIDM
from ..entries.scm import SingleCandleMitigation

MODULE_REGISTRY = {"choch_no_idm": ChoChNoIDM, "scm": SingleCandleMitigation}

class BacktestEngine:
    def __init__(self, journal, config):
        self.journal = journal
        self.config = config
        self.time_stop_bars = config.invalidation.time_stop_bars
        self.active_trades: List[Dict[str, Any]] = []

    def run(self, df_m5, pair: str):
        if not isinstance(df_m5.index, pd.DatetimeIndex):
            raise ValueError("df_m5 must have a DatetimeIndex")
        spread = get_spread(pair)
        if spread is None:
            raise ValueError(f"No spread configured for {pair}")
        df_m15 = df_m5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        ctx_engine = ContextEngine(swing_length=self.config.params.swing_length,
                                   kill_zones_utc=self.config.conditions.kill_zones)
        modules = [MODULE_REGISTRY[n]() for n in self.config.entry_modules if n in MODULE_REGISTRY]
        warmup = max(120, self.config.params.swing_length * 12)
        for i in range(warmup, len(df_m5) - 1):
            bar = df_m5.iloc[i]
            t = df_m5.index[i]
            self._manage_trades(bar, i)
            valid_m15 = df_m15[df_m15.index + pd.Timedelta(minutes=15) <= t]
            if valid_m15.empty:
                continue
            ctx = ctx_engine.build(valid_m15, t, pair=pair)
            if ctx is None:
                continue
            signal = None
            for mod in modules:
                signal = mod.check(df_m5.iloc[:i + 1], ctx, self.config)
                if signal:
                    signal["pair"] = pair
                    signal["ctx"] = {"bias": ctx.bias, "zone": ctx.zone,
                                     "pdh": ctx.pdh, "pdl": ctx.pdl}
                    break
            if not signal:
                continue
            direction = signal["direction"]
            next_open = df_m5.iloc[i + 1]["open"]
            entry = next_open + spread if direction == "BUY" else next_open - spread
            sl, tp = signal["sl"], signal["tp"]
            risk = abs(entry - sl)
            rr = abs(tp - entry) / risk if risk > 0 else 0.0
            ts = t.isoformat()
            if rr < MIN_RR_TO_DOL:
                self.journal.log_signal(signal, skipped=True,
                                        skip_reason=f"RR {rr:.2f} < {MIN_RR_TO_DOL}",
                                        timestamp_override=ts)
                continue
            signal.update(ref_price=entry, sl=sl, tp=tp, rr_to_dol=rr)
            row_id = self.journal.log_signal(signal, timestamp_override=ts)
            if row_id:
                self.active_trades.append({
                    "row_id": row_id, "pair": pair, "module": signal["module"],
                    "direction": direction, "entry": entry, "sl": sl, "tp": tp,
                    "risk": risk, "entry_bar_idx": i + 1,
                    "slices": [
                        {"size": 0.20, "target_r": 4.0, "status": "open", "realized_r": 0.0},
                        {"size": 0.30, "target_r": 10.0, "status": "open", "realized_r": 0.0},
                        {"size": 0.50, "target_r": None, "status": "open", "realized_r": 0.0},
                    ],
                    "mae_r": 0.0, "mfe_r": 0.0,
                })
        self._force_close_all(df_m5.iloc[-1], len(df_m5) - 1)
        return self.journal.get_all_modules_expectancy()

    def _manage_trades(self, bar, idx):
        to_remove = []
        for tr in self.active_trades:
            d = tr["direction"]
            entry, sl, tp, risk = tr["entry"], tr["sl"], tr["tp"], tr["risk"]
            if d == "BUY":
                profit_r = (bar["high"] - entry) / risk
                adverse_r = (entry - bar["low"]) / risk
                close_r = (bar["close"] - entry) / risk
                sl_hit = bar["low"] <= sl
                dol_hit = bar["high"] >= tp
            else:
                profit_r = (entry - bar["low"]) / risk
                adverse_r = (bar["high"] - entry) / risk
                close_r = (entry - bar["close"]) / risk
                sl_hit = bar["high"] >= sl
                dol_hit = bar["low"] <= tp
            tr["mae_r"] = min(tr["mae_r"], -adverse_r)
            tr["mfe_r"] = max(tr["mfe_r"], profit_r)
            bars_held = idx - tr["entry_bar_idx"]
            if sl_hit:
                stop_r = (sl - entry) / risk if d == "BUY" else (entry - sl) / risk
                self._close_open_slices(tr, stop_r, "closed_sl")
                self._finalize(tr, "SL", bars_held)
                to_remove.append(tr)
                continue
            be_moved_this_bar = False
            if profit_r >= 4.0:
                for s in tr["slices"]:
                    if s["status"] == "open" and s["target_r"] == 4.0:
                        s["realized_r"] = 4.0
                        s["status"] = "closed_4r"
                tr["sl"] = entry
                be_moved_this_bar = True
            if be_moved_this_bar:
                be_hit = (bar["low"] <= entry) if d == "BUY" else (bar["high"] >= entry)
                if be_hit:
                    self._close_open_slices(tr, 0.0, "closed_be")
                    self._finalize(tr, "SL", bars_held)
                    to_remove.append(tr)
                    continue
            if profit_r >= 10.0:
                for s in tr["slices"]:
                    if s["status"] == "open" and s["target_r"] == 10.0:
                        s["realized_r"] = 10.0
                        s["status"] = "closed_10r"
            if dol_hit:
                dol_r = abs(tp - entry) / risk
                self._close_open_slices(tr, dol_r, "closed_dol")
                self._finalize(tr, "DOL", bars_held)
                to_remove.append(tr)
                continue
            if bars_held >= self.time_stop_bars:
                self._close_open_slices(tr, close_r, "closed_timeout")
                self._finalize(tr, "TIME_STOP", bars_held)
                to_remove.append(tr)
        for tr in to_remove:
            self.active_trades.remove(tr)

    def _close_open_slices(self, tr, r, status):
        for s in tr["slices"]:
            if s["status"] == "open":
                s["realized_r"] = r
                s["status"] = status

    def _finalize(self, tr, exit_reason, bars_held):
        total_r = sum(s["size"] * s["realized_r"] for s in tr["slices"])
        outcome = "WIN" if total_r > 0.01 else "LOSS" if total_r < -0.01 else "BREAKEVEN"
        self.journal.update_trade_outcome(
            tr["row_id"], outcome, total_r, exit_reason,
            mae_r=tr["mae_r"], mfe_r=tr["mfe_r"], bars_held=bars_held)

    def _force_close_all(self, final_bar, final_idx):
        for tr in list(self.active_trades):
            d = tr["direction"]
            entry, risk = tr["entry"], tr["risk"]
            close_r = (final_bar["close"] - entry) / risk if d == "BUY" else (entry - final_bar["close"]) / risk
            self._close_open_slices(tr, close_r, "closed_manual")
            self._finalize(tr, "MANUAL", final_idx - tr["entry_bar_idx"])
            self.active_trades.remove(tr)

def _load_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.sort_index()
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=5) <= now]
    return df

def _print_mae_mfe(journal):
    import sqlite3
    conn = sqlite3.connect(journal.db_path)
    cur = conn.cursor()
    cur.execute("SELECT mae_r, mfe_r, outcome FROM trades "
                "WHERE outcome IS NOT NULL AND outcome!='SKIPPED' AND mae_r IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return
    maes = [r[0] for r in rows]
    mfes = [r[1] for r in rows]
    blown_4r = sum(1 for r in rows if r[1] >= 4.0 and r[2] != "WIN")
    print(f"\nMAE  min/median: {min(maes):.2f} / {statistics.median(maes):.2f} R")
    print(f"MFE  median/max: {statistics.median(mfes):.2f} / {max(mfes):.2f} R")
    print(f"Trades that reached 4R but did not end WIN: {blown_4r}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--pair", default="XAUUSD")
    ap.add_argument("--db", default="backtest_journal.db")
    args = ap.parse_args()
    config = load_config()
    journal = ExpectancyJournal(db_path=args.db)
    engine = BacktestEngine(journal, config)
    stats = engine.run(_load_csv(args.csv), args.pair)
    print(f"\n=== EXPECTANCY: {args.pair} ===")
    if not stats:
        print("No module reached 30 closed trades yet.")
    for m, s in stats.items():
        print(f"{m}: n={s['n']} W/L/T={s['wins']}/{s['losses']}/{s['timeouts']} "
              f"WR={s['win_rate']:.0%} exp={s['expectancy_r']:+.2f}R PF={s['profit_factor']:.2f} total={s['total_r']:+.1f}R")
    _print_mae_mfe(journal)
