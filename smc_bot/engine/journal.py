import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

VALID_OUTCOMES = {"WIN", "LOSS", "BREAKEVEN", "TIMEOUT", "SKIPPED"}
VALID_EXIT_REASONS = {"DOL", "PARTIAL_4R", "PARTIAL_10R", "SL", "TIME_STOP", "MANUAL", "RR_GATE"}
MIN_TRADES_FOR_STATS = 30
KILL_DECISION_MIN_TRADES = 100

class ExpectancyJournal:
    def __init__(self, db_path: str = "smc_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                module TEXT NOT NULL,
                direction TEXT NOT NULL,
                context_bias TEXT NOT NULL,
                context_zone TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                rr_to_dol REAL NOT NULL,
                outcome TEXT,
                pnl_r REAL,
                exit_reason TEXT,
                mae_r REAL,
                mfe_r REAL,
                bars_held INTEGER,
                alert_msg_id INTEGER
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_module ON trades(module)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON trades(outcome)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alert_msg ON trades(alert_msg_id)")
        conn.commit()
        conn.close()

    def log_signal(self, signal, skipped=False, skip_reason=None, timestamp_override=None):
        ts = timestamp_override or datetime.now(timezone.utc).isoformat()
        ctx = signal.get("ctx", {})
        vals = (
            ts,
            signal.get("pair", "UNKNOWN"),
            signal.get("module", "UNKNOWN"),
            signal.get("direction", "N/A"),
            ctx.get("bias", "N/A"),
            ctx.get("zone", "N/A"),
            float(signal.get("ref_price", 0.0)),
            float(signal.get("sl", 0.0)),
            float(signal.get("tp", 0.0)),
            float(signal.get("rr_to_dol", 0.0)),
            "SKIPPED" if skipped else None,
            skip_reason if skipped else None,
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades
                (timestamp, pair, module, direction, context_bias, context_zone,
                 entry_price, stop_loss, take_profit, rr_to_dol, outcome, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, vals)
        row_id = cur.lastrowid if not skipped else None
        conn.commit()
        conn.close()
        return row_id

    def update_trade_outcome(self, row_id, outcome, pnl_r, exit_reason,
                             mae_r=None, mfe_r=None, bars_held=None) -> bool:
        if outcome not in VALID_OUTCOMES or exit_reason not in VALID_EXIT_REASONS:
            logger.error(f"Invalid vocabulary: outcome={outcome} reason={exit_reason}")
            return False
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE trades SET outcome=?, pnl_r=?, exit_reason=?, mae_r=?, mfe_r=?, bars_held=?
            WHERE id=? AND outcome IS NULL
        """, (outcome, pnl_r, exit_reason, mae_r, mfe_r, bars_held, row_id))
        conn.commit()
        ok = cur.rowcount == 1
        conn.close()
        return ok

    def reset(self):
        """Delete all rows. Backtest runner uses this with --fresh so runs
        never accumulate across sweeps (the n=37->74->111 bug)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM trades")
        conn.commit()
        conn.close()
        logger.info(f"Journal reset: {self.db_path}")

    def set_alert_msg_id(self, row_id, msg_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE trades SET alert_msg_id=? WHERE id=?", (msg_id, row_id))
        conn.commit()
        conn.close()

    def get_row_id_by_alert_msg(self, msg_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM trades WHERE alert_msg_id=? AND outcome IS NULL", (msg_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def has_signal_for_bar(self, pair, ts) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM trades WHERE pair=? AND timestamp=? LIMIT 1", (pair, ts))
        found = cur.fetchone() is not None
        conn.close()
        return found

    def get_last_open_row(self, pair):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, module FROM trades WHERE pair=? AND outcome IS NULL ORDER BY id DESC LIMIT 1", (pair,))
        row = cur.fetchone()
        conn.close()
        return row

    def list_open_rows(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, pair, module, timestamp FROM trades WHERE outcome IS NULL ORDER BY id DESC LIMIT 10")
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_module_expectancy(self, module):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT pnl_r, outcome FROM trades
            WHERE module=? AND outcome IS NOT NULL AND outcome != 'SKIPPED'
        """, (module,))
        rows = cur.fetchall()
        conn.close()
        if len(rows) < MIN_TRADES_FOR_STATS:
            return None
        pnls = [r[0] for r in rows]
        outcomes = [r[1] for r in rows]
        n = len(pnls)
        wins = sum(1 for o in outcomes if o == "WIN")
        losses = sum(1 for o in outcomes if o == "LOSS")
        timeouts = sum(1 for o in outcomes if o == "TIMEOUT")
        breakevens = sum(1 for o in outcomes if o == "BREAKEVEN")
        win_trades = [p for p in pnls if p > 0]
        lose_trades = [p for p in pnls if p < 0]
        avg_win = sum(win_trades) / len(win_trades) if win_trades else 0.0
        avg_loss = abs(sum(lose_trades) / len(lose_trades)) if lose_trades else 0.0
        win_rate = wins / n
        lose_rate = (losses + timeouts) / n
        expectancy = (win_rate * avg_win) - (lose_rate * avg_loss)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        return {"module": module, "n": n, "wins": wins, "losses": losses,
                "timeouts": timeouts, "breakevens": breakevens,
                "win_rate": win_rate, "avg_win_r": avg_win, "avg_loss_r": avg_loss,
                "expectancy_r": expectancy, "profit_factor": pf, "total_r": sum(pnls)}

    def get_all_modules_expectancy(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT module FROM trades")
        modules = [r[0] for r in cur.fetchall()]
        conn.close()
        out = {}
        for m in modules:
            stats = self.get_module_expectancy(m)
            if stats:
                out[m] = stats
        return out

    def get_weekly_report(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cur.execute("""
            SELECT COUNT(*),
                   SUM(outcome='WIN'), SUM(outcome='LOSS'), SUM(outcome='TIMEOUT'),
                   SUM(pnl_r)
            FROM trades WHERE timestamp > ? AND outcome IS NOT NULL AND outcome != 'SKIPPED'
        """, (week_ago,))
        row = cur.fetchone()
        conn.close()
        if not row or not row[0]:
            return "📊 <b>Weekly SMC Report</b>\n\nNo closed trades this week."
        total, w, l, t, total_r = row
        lines = [f"📊 <b>Weekly SMC Report</b>\n\nClosed: <b>{total}</b>  ({w}W / {l}L / {t}T)\nTotal: <b>{total_r:+.2f}R</b>\n"]
        for m, s in self.get_all_modules_expectancy().items():
            emoji = "✅" if s["expectancy_r"] > 0 else "❌"
            lines.append(f"{emoji} <b>{m}</b>: n={s['n']} | {s['win_rate']:.0%} WR | {s['expectancy_r']:+.2f}R exp | PF {s['profit_factor']:.2f}")
        lines.append(f"\n<i>Kill/keep decisions require {KILL_DECISION_MIN_TRADES}+ trades per module.</i>")
        return "\n".join(lines)
