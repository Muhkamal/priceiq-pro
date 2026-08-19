"""
PriceIQ Pro — MAE / MFE Analyzer (post-trade excursion analytics)

MAE (Maximum Adverse Excursion) = deepest unrealised LOSS a trade saw, in R.
MFE (Maximum Favorable Excursion) = biggest unrealised WIN a trade saw, in R.

What the stats tell you:
  * Winners with deep MAE (>0.6R)      → entries early / stops too tight.
  * Losers with high MFE (>0.5R)       → profit give-back → add BE/trail rule.
  * Winners whose MFE rarely hits TP2  → TP2 too ambitious → scale out sooner.

Self-contained: own JSON history, hooks into V5 with 4 tiny try/except calls.
Can never break the signal path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _pip_size(pair: str) -> float:
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair or "GOLD" in pair:
        return 0.1
    if "BTC" in pair or "ETH" in pair:
        return 1.0
    return 0.0001


@dataclass
class ExcursionRecord:
    trade_id:    str
    pair:        str
    direction:   str
    entry:       float
    stop_loss:   float
    risk_dist:   float
    exit_price:  float
    outcome:     str
    r_multiple:  float
    mfe_r:       float
    mae_r:       float
    mfe_pips:    float
    mae_pips:    float
    tp1_r:       float
    tp2_r:       float
    closed_at:   str


class MAEMFETracker:
    """Tracks best/worst price per open position; mines insights on close."""

    def __init__(self, history_path: str = "mae_mfe_history.json",
                 max_records: int = 1000):
        self._path = history_path
        self._max  = max_records
        self._lock = threading.Lock()
        self._open:   Dict[str, dict] = {}
        self._closed: List[ExcursionRecord] = []
        self._load()

    # ── lifecycle hooks ───────────────────────────────────────
    def register(self, pair: str, direction: str, entry: float,
                 stop_loss: float, tp1=None, tp2=None, tp3=None,
                 trade_id: str = None):
        risk = abs(entry - stop_loss)
        if risk <= 0 or entry <= 0:
            logger.debug(f"MAE tracker: invalid risk for {pair}, skipping")
            return
        sign = 1.0 if str(direction).lower() == "buy" else -1.0

        def to_r(price):
            return sign * (price - entry) / risk

        with self._lock:
            self._open[pair] = {
                "trade_id": trade_id or f"{pair}_{int(datetime.now(timezone.utc).timestamp())}",
                "pair": pair, "direction": str(direction).lower(),
                "entry": entry, "stop_loss": stop_loss, "risk": risk,
                "best_r": 0.0, "worst_r": 0.0,
                "tp1_r": round(to_r(tp1), 2) if tp1 else 0.0,
                "tp2_r": round(to_r(tp2), 2) if tp2 else 0.0,
            }
        logger.debug(f"MAE tracker registered {pair} {direction}")

    def update(self, pair: str, price: float):
        with self._lock:
            pos = self._open.get(pair)
            if not pos or not price or price <= 0:
                return
            sign = 1.0 if pos["direction"] == "buy" else -1.0
            r = sign * (price - pos["entry"]) / pos["risk"]
            if r > pos["best_r"]:
                pos["best_r"] = r
            if r < pos["worst_r"]:
                pos["worst_r"] = r

    def update_all(self, prices: Dict[str, float]):
        for pair, price in prices.items():
            self.update(pair, price)

    def finalize(self, pair: str, exit_price: float,
                 outcome: str = "unknown",
                 r_multiple: float = 0.0) -> Optional[ExcursionRecord]:
        with self._lock:
            pos = self._open.pop(pair, None)
        if not pos:
            return None
        sign   = 1.0 if pos["direction"] == "buy" else -1.0
        r_exit = sign * (exit_price - pos["entry"]) / pos["risk"]
        mfe_r  = max(pos["best_r"], r_exit)
        mae_r  = min(pos["worst_r"], r_exit)
        pip    = _pip_size(pair)

        rec = ExcursionRecord(
            trade_id=pos["trade_id"], pair=pair, direction=pos["direction"],
            entry=pos["entry"], stop_loss=pos["stop_loss"],
            risk_dist=pos["risk"], exit_price=exit_price,
            outcome=outcome, r_multiple=r_multiple,
            mfe_r=round(mfe_r, 2), mae_r=round(mae_r, 2),
            mfe_pips=round(mfe_r * pos["risk"] / pip, 1),
            mae_pips=round(mae_r * pos["risk"] / pip, 1),
            tp1_r=pos["tp1_r"], tp2_r=pos["tp2_r"],
            closed_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._closed.append(rec)
            if len(self._closed) > self._max:
                self._closed = self._closed[-self._max:]
            self._save()
        logger.info(
            f"MAE/MFE {pair} {outcome}: MAE={rec.mae_r:+.2f}R ({rec.mae_pips:+.1f}pips) "
            f"MFE={rec.mfe_r:+.2f}R ({rec.mfe_pips:+.1f}pips) final={r_multiple:+.2f}R"
        )
        return rec

    # ── analytics ─────────────────────────────────────────────
    def summary(self) -> dict:
        with self._lock:
            recs = list(self._closed)
        wins   = [r for r in recs if r.r_multiple > 0]
        losses = [r for r in recs if r.r_multiple <= 0]

        def avg(xs):
            return round(sum(xs) / len(xs), 2) if xs else 0.0

        return {
            "n_trades":     len(recs),
            "wins":         len(wins),
            "losses":       len(losses),
            "avg_mae_win":  avg([r.mae_r for r in wins]),
            "avg_mae_loss": avg([r.mae_r for r in losses]),
            "avg_mfe_win":  avg([r.mfe_r for r in wins]),
            "avg_mfe_loss": avg([r.mfe_r for r in losses]),
            "worst_mae":    min([r.mae_r for r in recs], default=0.0),
            "best_mfe":     max([r.mfe_r for r in recs], default=0.0),
        }

    def insights(self, be_threshold: float = 0.5) -> List[str]:
        with self._lock:
            recs = list(self._closed)
        if len(recs) < 5:
            return [f"MAE/MFE: collecting data ({len(recs)}/5 trades recorded)"]

        out    = []
        wins   = [r for r in recs if r.r_multiple > 0]
        losses = [r for r in recs if r.r_multiple <= 0]

        # 1. Give-back → break-even rule
        if losses:
            gave_back = [r for r in losses if r.mfe_r >= be_threshold]
            frac = len(gave_back) / len(losses)
            if frac >= 0.35:
                out.append(
                    f"⚠️ {frac:.0%} of losers were up ≥{be_threshold:.1f}R before losing "
                    f"→ move SL to break-even at +{be_threshold:.1f}R."
                )

        # 2. Stop tightness / entry timing
        if wins:
            avg_mae_win = sum(r.mae_r for r in wins) / len(wins)
            if avg_mae_win <= -0.6:
                out.append(
                    f"⚠️ Winners endure {avg_mae_win:.2f}R of heat on average → "
                    f"entries early or stops too tight; widen SL or add confirmation."
                )
            elif avg_mae_win >= -0.25:
                out.append(
                    f"✅ Winners barely draw down (avg MAE {avg_mae_win:.2f}R) — entries well timed."
                )

        # 3. TP2 ambition
        if wins:
            tp2s = [r.tp2_r for r in wins if r.tp2_r > 0]
            if tp2s:
                tp2_med = sorted(tp2s)[len(tp2s) // 2]
                reached = [r for r in wins if r.mfe_r >= tp2_med * 0.9]
                frac = len(reached) / len(wins)
                if frac < 0.3:
                    mfe_med = sorted(r.mfe_r for r in wins)[len(wins) // 2]
                    out.append(
                        f"⚠️ TP2 ({tp2_med:.1f}R) reached in only {frac:.0%} of winners "
                        f"(median MFE {mfe_med:.1f}R) → scale out earlier or lower TP2."
                    )

        # 4. Trailing opportunity
        big_giveback = [r for r in recs if r.mfe_r >= 1.0 and r.r_multiple < 0.5]
        if len(big_giveback) >= 3:
            out.append(
                f"⚠️ {len(big_giveback)} trades gave back ≥1R of profit → "
                f"activate ATR trailing after +1R."
            )

        if not out:
            out.append("✅ MAE/MFE healthy — no stop/TP changes recommended.")
        return out

    def report(self) -> str:
        s = self.summary()
        return "\n".join([
            "📊 <b>MAE/MFE REPORT</b>",
            f"Trades: {s['n_trades']} (W{s['wins']}/L{s['losses']})",
            f"Avg MAE: win {s['avg_mae_win']:+.2f}R | loss {s['avg_mae_loss']:+.2f}R",
            f"Avg MFE: win {s['avg_mfe_win']:+.2f}R | loss {s['avg_mfe_loss']:+.2f}R",
            f"Worst MAE {s['worst_mae']:+.2f}R | Best MFE {s['best_mfe']:+.2f}R",
        ] + self.insights())

    # ── persistence ──────────────────────────────────────────
    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump([asdict(r) for r in self._closed], f)
        except Exception as e:
            logger.debug(f"MAE history save error: {e}")

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    data = json.load(f)
                fields = ExcursionRecord.__dataclass_fields__
                self._closed = [
                    ExcursionRecord(**{k: d.get(k) for k in fields})
                    for d in data if isinstance(d, dict)
                ]
        except Exception as e:
            logger.debug(f"MAE history load error: {e}")
            self._closed = []


mae_tracker = MAEMFETracker()
