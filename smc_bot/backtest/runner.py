"""Event-driven M5 backtest with gate skip counters and progress tracking."""
import argparse
import random
import statistics
import pandas as pd
from typing import Dict, List, Any
import sys

from ..config import load_config
from ..core.context import ContextEngine
from ..core.pairs import get_spread
from ..core.markets import get_profile
from ..core.data_quality import validate_m5
from ..engine.journal import ExpectancyJournal, MIN_TRADES_FOR_STATS, BOOTSTRAP_RESAMPLES
from ..engine.alerts import MIN_RR_TO_DOL
from ..entries.choch_no_idm import ChoChNoIDM
from ..entries.scm import SingleCandleMitigation
from ..entries.double_bos import DoubleBreakout
from ..entries.choch_idm import ChoChIDM
from ..entries.indicator_confluence import IndicatorConfluence
from ..entries.random_control import RandomControl, FROZEN_SEED, FROZEN_FIRE_PROB

MODULE_REGISTRY = {"choch_no_idm": ChoChNoIDM, "scm": SingleCandleMitigation, "double_bos": DoubleBreakout, "choch_idm": ChoChIDM, "indicator_confluence": IndicatorConfluence}
# RandomControl is intentionally absent from MODULE_REGISTRY: it must never be
# addable via system.yaml's entry_modules list (backtest or live). It is only
# ever run via the --control CLI flag below, which swaps the module list
# entirely for the duration of that one run.
M15_WINDOW = 500

class BacktestEngine:
    def __init__(self, journal, config):
        self.journal = journal
        self.config = config
        self.time_stop_bars = config.invalidation.time_stop_bars
        self.active_trades: List[Dict[str, Any]] = []
        self.gate_counts = {
            "total_bars": 0,
            "ctx_none": 0,
            "dol_none": 0,
            "zone_gate": 0,
            "kill_zone": 0,
            "module_miss": 0,
            "rr_skip": 0,
            "wrong_side": 0,
            "signals": 0,
            "pattern_raw_bull": 0,
            "pattern_raw_bear": 0,
            "dq_skip": 0,
        }
        self.module_signals = {}
        self.module_fired = {}

    def run(self, df_m5, pair: str, modules_override=None):
        if not isinstance(df_m5.index, pd.DatetimeIndex):
            raise ValueError("df_m5 must have a DatetimeIndex")
        
        spread = get_spread(pair)
        if spread is None:
            raise ValueError(f"No spread configured for {pair}")
        
        prof = get_profile(pair)
        compute_poi = bool(self.config.conditions.require_unmitigated_zone)
        
        df_m15 = df_m5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        m15_idx = df_m15.index
        
        ctx_engine = ContextEngine(swing_length=self.config.params.swing_length,
                                   kill_zones_utc=self.config.conditions.kill_zones)
        if modules_override is not None:
            modules = modules_override
        else:
            modules = [MODULE_REGISTRY[n]() for n in self.config.entry_modules if n in MODULE_REGISTRY]
        warmup = max(120, self.config.params.swing_length * 12)
        
        j = 0
        ctx_slot = -1
        ctx = None
        
        total_bars = len(df_m5) - 1 - warmup
        print(f"Processing {total_bars} bars (warmup={warmup})...", end='', flush=True)
        
        for i in range(warmup, len(df_m5) - 1):
            if i % 5000 == 0:
                pct = (i - warmup) / total_bars * 100
                print(f"\rProcessing {total_bars} bars: {pct:.0f}% ({i - warmup}/{total_bars})", end='', flush=True)
            
            bar = df_m5.iloc[i]
            t = df_m5.index[i]
            self.gate_counts["total_bars"] += 1
            self._manage_trades(bar, i)
            
            while j < len(m15_idx) and m15_idx[j] + pd.Timedelta(minutes=15) <= t:
                j += 1
            
            if j == 0:
                continue
            
            if ctx_slot != j:
                ctx = ctx_engine.build(df_m15.iloc[max(0, j - M15_WINDOW):j], t,
                                       pair=pair, profile=prof, compute_poi=compute_poi)
                ctx_slot = j
            
            if ctx is None:
                self.gate_counts["ctx_none"] += 1
                continue
            
            if ctx.dol is None:
                self.gate_counts["dol_none"] += 1
            
            # Zone gate: must be in correct zone for bias
            if ctx.zone == "EQUILIBRIUM" or \
               (ctx.bias == "BULLISH" and ctx.zone != "DISCOUNT") or \
               (ctx.bias == "BEARISH" and ctx.zone != "PREMIUM"):
                self.gate_counts["zone_gate"] += 1
                continue
            
            # Kill zone gate (disabled for synthetics via profile)
            if not ctx.in_kill_zone:
                self.gate_counts["kill_zone"] += 1
                continue
            
            # Raw pattern detection (no context gates)
            if i >= 2:
                c0 = df_m5.iloc[i]
                c1 = df_m5.iloc[i - 1]
                c2 = df_m5.iloc[i - 2]
                
                # Bullish: sweep prev low, close through candle-before's high
                if c0["low"] < c1["low"] and c0["close"] > c2["high"]:
                    self.gate_counts["pattern_raw_bull"] += 1
                
                # Bearish: sweep prev high, close through candle-before's low
                if c0["high"] > c1["high"] and c0["close"] < c2["low"]:
                    self.gate_counts["pattern_raw_bear"] += 1
            
            # Data-quality parity with live: live scan_pair skips the cycle when
            # validate_m5 flags recent data; backtest must skip the bar the same
            # way, or bad ticks become fake signals (esp. sweep-and-reject shapes).
            ok, _dq = validate_m5(df_m5.iloc[max(0, i - 100):i + 1], pair)
            if not ok:
                self.gate_counts["dq_skip"] += 1
                continue
            
            # Module check
            signal = None
            for mod in modules:
                signal = mod.check(df_m5.iloc[:i + 1], ctx, self.config)
                if signal:
                    self.module_fired[signal["module"]] = \
                        self.module_fired.get(signal["module"], 0) + 1
                    signal["pair"] = pair
                    signal["ctx"] = {"bias": ctx.bias, "zone": ctx.zone,
                                     "pdh": ctx.pdh, "pdl": ctx.pdl}
                    break
            
            if not signal:
                self.gate_counts["module_miss"] += 1
                continue
            
            self.gate_counts["signals"] += 1
            
            direction = signal["direction"]
            next_open = df_m5.iloc[i + 1]["open"]
            entry = next_open + 1.25 * spread if direction == "BUY" else next_open - 1.25 * spread
            sl, tp = signal["sl"], signal["tp"]
            risk = abs(entry - sl)
            reward = (tp - entry) if direction == "BUY" else (entry - tp)
            rr = reward / risk if risk > 0 else 0.0
            ts = t.isoformat()
            
            if reward <= 0:
                self.gate_counts["wrong_side"] += 1
                continue
            
            if rr < MIN_RR_TO_DOL:
                self.gate_counts["rr_skip"] += 1
                continue
            
            signal.update(ref_price=entry, sl=sl, tp=tp, rr_to_dol=rr)
            self.module_signals[signal["module"]] = self.module_signals.get(signal["module"], 0) + 1
            row_id = self.journal.log_signal(signal, timestamp_override=ts)
            if row_id:
                self.active_trades.append({
                    "row_id": row_id, "journal": self.journal,
                    "pair": pair, "module": signal["module"],
                    "direction": direction, "entry": entry, "sl": sl, "tp": tp,
                    "risk": risk, "entry_bar_idx": i + 1,
                    "slices": [
                        {"size": 0.20, "target_r": 4.0, "status": "open", "realized_r": 0.0},
                        {"size": 0.30, "target_r": 10.0, "status": "open", "realized_r": 0.0},
                        {"size": 0.50, "target_r": None, "status": "open", "realized_r": 0.0},
                    ],
                    "mae_r": 0.0, "mfe_r": 0.0,
                })
                if getattr(self, "perm_journal", None) is not None:
                    sdir = "BUY" if self.perm_rng.random() < 0.5 else "SELL"
                    se = next_open + 1.25 * spread if sdir == "BUY" else next_open - 1.25 * spread
                    d_sl, d_tp = abs(entry - sl), abs(tp - entry)
                    if sdir == "BUY":
                        ssl, stp = se - d_sl, se + d_tp
                    else:
                        ssl, stp = se + d_sl, se - d_tp
                    ssig = {"module": signal["module"] + "_PERM", "direction": sdir,
                            "ref_price": se, "sl": ssl, "tp": stp, "rr_to_dol": rr,
                            "pair": pair, "ctx": signal["ctx"]}
                    srow = self.perm_journal.log_signal(ssig, timestamp_override=ts)
                    if srow:
                        self.active_trades.append({
                            "row_id": srow, "journal": self.perm_journal,
                            "pair": pair, "module": signal["module"] + "_PERM",
                            "direction": sdir, "entry": se, "sl": ssl, "tp": stp,
                            "risk": d_sl, "entry_bar_idx": i + 1,
                            "slices": [
                                {"size": 0.20, "target_r": 4.0, "status": "open", "realized_r": 0.0},
                                {"size": 0.30, "target_r": 10.0, "status": "open", "realized_r": 0.0},
                                {"size": 0.50, "target_r": None, "status": "open", "realized_r": 0.0},
                            ],
                            "mae_r": 0.0, "mfe_r": 0.0,
                        })
        
        print("\rProcessing complete. Finalizing trades...", end='', flush=True)
        self._force_close_all(df_m5.iloc[-1], len(df_m5) - 1)
        print(" done.")
        
        # Print gate counters
        print("\n=== GATE SKIP COUNTERS ===")
        for mod in sorted(set(self.module_fired) | set(self.module_signals)):
            print(f"  fired[{mod}]".ljust(17) + f": {self.module_fired.get(mod, 0):>6}")
            print(f"  passed[{mod}]".ljust(17) + f": {self.module_signals.get(mod, 0):>6}")
        for gate, count in self.gate_counts.items():
            print(f"  {gate:<15}: {count:>6}")
        print("")
        
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
                dol_hit = tp > entry and bar["high"] >= tp
            else:
                profit_r = (entry - bar["low"]) / risk
                adverse_r = (bar["high"] - entry) / risk
                close_r = (entry - bar["close"]) / risk
                sl_hit = bar["high"] >= sl
                dol_hit = tp < entry and bar["low"] <= tp
            
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
        tr["journal"].update_trade_outcome(
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
    if "volume" not in df.columns:
        df["volume"] = 1000
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
    ap.add_argument("--start", default=None, help="inclusive start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default=None, help="exclusive end date YYYY-MM-DD (UTC)")
    ap.add_argument("--fresh", action="store_true",
                    help="Reset journal DB before running (no cross-run accumulation)")
    ap.add_argument("--target", choices=["dol", "measured_move", "next_swing"],
                    default=None, help="Override continuation_target for this run")
    ap.add_argument("--verify", action="store_true",
                    help="Run twice on temp DBs, assert identical trade counts")
    ap.add_argument("--compare-control-db", default=None,
                    help="Path to a DB produced by a --control run on the SAME "
                         "pair/window. If given, the report shows each real "
                         "module's bootstrap CI against the control's, and "
                         "flags whether the real module clears the control's "
                         "95th-percentile bound - the MHC gate.")
    ap.add_argument("--control", action="store_true",
                    help="Run ONLY the frozen RandomControl module (ignores "
                         "system.yaml entry_modules entirely) to establish the "
                         "null-hypothesis baseline this pair/window must beat.")
    ap.add_argument("--control-atr-mult", type=float, default=1.0,
                    help="RandomControl stop-width multiplier (exp-CTRL-2 grid: "
                         "0.5 / 1.0 / 2.0 / 3.0). Does not affect RNG stream.")
    ap.add_argument("--permute", action="store_true",
                    help="For every real module signal that passes, also create one "
                         "shadow trade with direction drawn from a seeded fair coin "
                         "(exp-PERM spec: seed 20260927, distance-preserving mirror). "
                         "Shadows go to --perm-db; real DB stays clean. Not allowed "
                         "with --control.")
    ap.add_argument("--perm-db", default=None,
                    help="Shadow journal path (default: <db>.perm.db)")
    ap.add_argument("--control-seed", type=int, default=FROZEN_SEED,
                    help=f"Seed for --control (default {FROZEN_SEED}, the frozen "
                         "spec). Only change this to register a NEW control spec "
                         "as its own ledger entry - never to re-roll a result.")
    args = ap.parse_args()
    config = load_config()
    if args.target:
        config.params.continuation_target = args.target

    def _one_run(db_path):
        journal = ExpectancyJournal(db_path=db_path)
        if args.fresh or args.verify:
            journal.reset()
        df = _load_csv(args.csv)
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df[df.index < pd.Timestamp(args.end, tz="UTC")]
        if len(df) < 500:
            raise SystemExit(f"Window too small: {len(df)} bars - check --start/--end")
        print(f"Backtesting {len(df)} bars: {df.index[0]} -> {df.index[-1]}  [db={db_path}]")
        print(f"CONFIG: swing_length={config.params.swing_length} target={config.params.continuation_target}")
        print(f"COST: spread(full,at-entry)=2xhalf + slippage=25% -> model charges 1.25x spread")
        modules_override = None
        if args.control:
            print(f"*** CONTROL MODE: only RandomControl(seed={args.control_seed}, "
                  f"fire_prob={FROZEN_FIRE_PROB}, atr_mult={args.control_atr_mult}) runs. "
                  f"entry_modules from system.yaml is ignored for this run. ***")
            modules_override = [RandomControl(seed=args.control_seed,
                                              atr_mult=args.control_atr_mult)]
        engine = BacktestEngine(journal, config)
        if args.permute and not args.control:
            perm_db = args.perm_db or (db_path + ".perm.db")
            perm_journal = ExpectancyJournal(db_path=perm_db)
            if args.fresh or args.verify:
                perm_journal.reset()
            engine.perm_journal = perm_journal
            engine.perm_rng = random.Random(20260927)  # frozen in exp-PERM
        stats = engine.run(df, args.pair, modules_override=modules_override)
        return stats, engine

    def _report(stats, pair, db_path):
        print(f"\n=== EXPECTANCY: {pair} ===")
        if not stats:
            print("No module reached 30 closed trades yet.")
        journal_for_ci = ExpectancyJournal(db_path=db_path)
        control_ci = None
        if args.compare_control_db:
            control_journal = ExpectancyJournal(db_path=args.compare_control_db)
            control_ci = control_journal.get_expectancy_bootstrap_ci(RandomControl.name)
            if control_ci is None:
                print(f"WARNING: --compare-control-db has no closed RandomControl "
                      f"trades - MHC comparison skipped.")
            else:
                print(f"\nCONTROL (from {args.compare_control_db}): "
                      f"n={control_ci['n']} exp={control_ci['expectancy_r']:+.3f}R "
                      f"[{control_ci['lo_pct']}%={control_ci['ci_lo']:+.3f}R, "
                      f"{control_ci['hi_pct']}%={control_ci['ci_hi']:+.3f}R]")
        for m, s in stats.items():
            print(f"{m}: n={s['n']} W/L/T={s['wins']}/{s['losses']}/{s['timeouts']} "
                  f"WR={s['win_rate']:.0%} exp={s['expectancy_r']:+.2f}R PF={s['profit_factor']:.2f} total={s['total_r']:+.1f}R")
            ci = journal_for_ci.get_expectancy_bootstrap_ci(m)
            if ci:
                print(f"    bootstrap CI: [{ci['lo_pct']}%={ci['ci_lo']:+.3f}R, "
                      f"{ci['hi_pct']}%={ci['ci_hi']:+.3f}R] (n_resamples={BOOTSTRAP_RESAMPLES})")
            if control_ci is not None and ci:
                beats = ci['expectancy_r'] > control_ci['ci_hi'] and s['n'] >= MIN_TRADES_FOR_STATS
                verdict = "CLEARS control 95th pct" if beats else "does NOT clear control - discount this result"
                print(f"    MHC gate: {verdict}")

        # --- below-gate reporter: every module with n>=1 gets its real numbers ---
        # (reporter-gap fix exp-007-corr-2: hiding exp below the gate selected for overclaim)
        import sqlite3 as _sq
        _c = _sq.connect(db_path)
        _rows = _c.execute("""
            SELECT module, COUNT(*),
                   SUM(CASE WHEN pnl_r > 0 THEN 1 ELSE 0 END),
                   AVG(pnl_r)
            FROM trades WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
            GROUP BY module""").fetchall()
        _c.close()
        _below = [r for r in _rows if r[1] < MIN_TRADES_FOR_STATS]
        if _below:
            print("\n=== BELOW-GATE MODULES (n<30 - descriptive only, NO verdict) ===")
            for m, n, w, exp in sorted(_below, key=lambda r: -r[1]):
                print(f"{m}: n={n} W/L={w}/{n-w} WR={100*w/n:.0f}% exp={exp:+.3f}R  [below 30-trade gate]")

    if args.verify:
        import tempfile, os, sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            d1 = os.path.join(tmp, "r1.db")
            d2 = os.path.join(tmp, "r2.db")
            s1, _e1 = _one_run(d1)
            s2, _e2 = _one_run(d2)

            def _counts(db):
                conn = sqlite3.connect(db)
                c = dict(conn.execute(
                    "SELECT module, COUNT(*) FROM trades GROUP BY module").fetchall())
                conn.close()
                return c

            n1, n2 = _counts(d1), _counts(d2)

        total = sum(n1.values())
        if n1 == n2:
            print(f"\nREPRODUCIBILITY OK: {n1}")
            if total < 10:
                print(f"NOTE: only {total} trades in window - deterministic but thin sample")
        else:
            print(f"\nREPRODUCIBILITY FAILED: run1={n1} run2={n2}")
            raise SystemExit(2)
        _report(s1, args.pair, d1)
    else:
        stats, engine = _one_run(args.db)
        _report(stats, args.pair, args.db)
        _print_mae_mfe(ExpectancyJournal(db_path=args.db))
        pj = getattr(engine, "perm_journal", None)
        if pj is not None:
            print("\n=== PERMUTATION SHADOW (skill null, exp-PERM) ===")
            import sqlite3 as _sq
            _c = _sq.connect(pj.db_path)
            _mods = [r[0] for r in _c.execute("SELECT DISTINCT module FROM trades").fetchall()]
            _c.close()
            for m in _mods:
                ci = pj.get_expectancy_bootstrap_ci(m)
                if ci:
                    print(f"{m}: n={ci['n']} exp={ci['expectancy_r']:+.2f}R "
                          f"shadow 95th-pct bound (skill bar): {ci['ci_hi']:+.3f}R")
