"""
PriceIQ Pro — Trade Journal v1.0

Structured trade journal with rich annotations, query API, and
performance breakdowns. Goes beyond the outcome store — this is
the human-readable record of every decision made.

Stores per trade:
    - Full signal context (agent, regime, confidence, EV)
    - Execution details (fill, slippage, session)
    - Management events (TP1 hit, SL moved to BE, trail)
    - Outcome (R-multiple, PnL USD, duration)
    - Notes field (for review annotations)
    - Tags (e.g. "overtraded", "missed_exit", "clean")

Query API:
    journal.query(pair="XAUUSD", outcome="win", regime="trending")
    journal.breakdown_by("agent")
    journal.breakdown_by("session")
    journal.breakdown_by("regime")
    journal.worst_trades(n=5)
    journal.best_trades(n=5)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

JOURNAL_PATH = "trade_journal.json"


@dataclass
class JournalEntry:
    # Identity
    trade_id:       str
    pair:           str
    timeframe:      str

    # Signal context
    agent:          str
    regime:         str
    regime_probs:   Dict[str, float]
    session:        str
    direction:      str
    confidence:     float
    win_probability: float
    expected_value: float
    conflict_score: float

    # Execution
    entry_price:    float
    fill_price:     float
    slippage_pips:  float
    stop_loss:      float
    tp1:            float
    tp2:            float
    tp3:            float
    lots:           float
    sizing_method:  str

    # Outcome
    exit_price:     float
    outcome:        str         # "win" | "loss" | "timeout" | "breakeven"
    r_multiple:     float
    pnl_usd:        float
    duration_hours: float
    management_events: List[str]   # ["TP1_HIT", "SL_TO_BE", "TRAIL_MOVED"]

    # Review
    opened_at:      str
    closed_at:      str
    notes:          str = ""
    tags:           List[str] = field(default_factory=list)
    reviewed:       bool = False


class TradeJournal:
    """
    Persistent trade journal with rich querying capabilities.
    """

    def __init__(self, path: str = JOURNAL_PATH):
        self._path    = path
        self._entries: List[JournalEntry] = []
        self._load()

    # ── Write ─────────────────────────────────────────────────

    def record(self, entry: JournalEntry):
        """Add a completed trade to the journal."""
        self._entries.append(entry)
        self._save()
        logger.info(
            f"Journal: {entry.pair} {entry.direction} {entry.outcome} "
            f"{entry.r_multiple:+.2f}R ${entry.pnl_usd:+.2f} "
            f"via {entry.agent}/{entry.regime}"
        )

    def annotate(self, trade_id: str, notes: str = "", tags: Optional[List[str]] = None):
        """Add notes or tags to a trade after review."""
        for e in self._entries:
            if e.trade_id == trade_id:
                e.notes    = notes
                e.tags     = tags or []
                e.reviewed = True
                self._save()
                return
        logger.warning(f"Journal: trade_id {trade_id} not found")

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        pair:     Optional[str] = None,
        agent:    Optional[str] = None,
        regime:   Optional[str] = None,
        session:  Optional[str] = None,
        outcome:  Optional[str] = None,
        timeframe: Optional[str] = None,
        last_n:   Optional[int] = None,
    ) -> List[JournalEntry]:
        """Filter entries by any combination of fields."""
        results = self._entries
        if pair:      results = [e for e in results if e.pair.upper()    == pair.upper()]
        if agent:     results = [e for e in results if e.agent           == agent]
        if regime:    results = [e for e in results if e.regime          == regime]
        if session:   results = [e for e in results if e.session         == session]
        if outcome:   results = [e for e in results if e.outcome         == outcome]
        if timeframe: results = [e for e in results if e.timeframe       == timeframe]
        if last_n:    results = results[-last_n:]
        return results

    def breakdown_by(self, field_name: str) -> Dict[str, Dict]:
        """
        Performance breakdown by any field: "agent", "regime", "session",
        "pair", "timeframe", "direction", "outcome"
        """
        groups: Dict[str, List[JournalEntry]] = {}
        for e in self._entries:
            key = str(getattr(e, field_name, "unknown"))
            groups.setdefault(key, []).append(e)

        result = {}
        for key, entries in groups.items():
            wins   = [e for e in entries if e.outcome == "win"]
            r_mults = [e.r_multiple for e in entries]
            pnls    = [e.pnl_usd for e in entries]
            result[key] = {
                "trades":    len(entries),
                "wins":      len(wins),
                "losses":    len(entries) - len(wins),
                "win_rate":  round(len(wins) / len(entries), 3),
                "total_pnl": round(sum(pnls), 2),
                "avg_r":     round(float(np.mean(r_mults)), 3) if r_mults else 0,
                "avg_duration_h": round(
                    float(np.mean([e.duration_hours for e in entries])), 1
                ),
                "avg_slippage": round(
                    float(np.mean([e.slippage_pips for e in entries])), 2
                ),
            }
        return result

    def best_trades(self, n: int = 5) -> List[JournalEntry]:
        return sorted(self._entries, key=lambda e: e.r_multiple, reverse=True)[:n]

    def worst_trades(self, n: int = 5) -> List[JournalEntry]:
        return sorted(self._entries, key=lambda e: e.r_multiple)[:n]

    def streaks(self) -> Dict:
        """Current and max win/loss streaks."""
        if not self._entries:
            return {}
        current_streak = 1
        current_type   = self._entries[-1].outcome
        max_win = max_loss = 1
        streak  = 1

        for i in range(len(self._entries) - 2, -1, -1):
            if self._entries[i].outcome == self._entries[i+1].outcome:
                streak += 1
            else:
                break
        current_streak = streak

        # Max streaks overall
        streak = 1
        for i in range(1, len(self._entries)):
            if self._entries[i].outcome == self._entries[i-1].outcome:
                streak += 1
                if self._entries[i].outcome == "win":
                    max_win  = max(max_win, streak)
                else:
                    max_loss = max(max_loss, streak)
            else:
                streak = 1

        return {
            "current_streak": current_streak,
            "current_type":   current_type,
            "max_win_streak":  max_win,
            "max_loss_streak": max_loss,
        }

    def full_stats(self) -> Dict:
        """Summary statistics for the entire journal."""
        if not self._entries:
            return {"trades": 0}
        wins     = [e for e in self._entries if e.outcome == "win"]
        r_mults  = [e.r_multiple for e in self._entries]
        pnls     = [e.pnl_usd    for e in self._entries]
        return {
            "total_trades":   len(self._entries),
            "win_rate":       round(len(wins) / len(self._entries), 3),
            "total_pnl":      round(sum(pnls), 2),
            "expectancy":     round(float(np.mean(r_mults)), 3),
            "avg_win_r":      round(float(np.mean([r for r in r_mults if r > 0])), 3)
                              if any(r > 0 for r in r_mults) else 0,
            "avg_loss_r":     round(float(np.mean([r for r in r_mults if r <= 0])), 3)
                              if any(r <= 0 for r in r_mults) else 0,
            "profit_factor":  round(
                sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 0.01), 3
            ),
            "by_agent":   self.breakdown_by("agent"),
            "by_regime":  self.breakdown_by("regime"),
            "by_session": self.breakdown_by("session"),
            "by_pair":    self.breakdown_by("pair"),
            "streaks":    self.streaks(),
        }

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump([asdict(e) for e in self._entries], f, indent=2)
        except Exception as ex:
            logger.warning(f"Journal save failed: {ex}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._entries = [JournalEntry(**d) for d in data]
            logger.info(f"Trade journal loaded: {len(self._entries)} entries")
        except Exception as ex:
            logger.warning(f"Journal load failed: {ex}")
