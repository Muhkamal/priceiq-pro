"""
PriceIQ Pro — Performance Attribution Engine v1.0

Answers the question no other component answers:
"Which EXACT combination of agent + regime + session + confidence
 + pattern is actually generating our edge?"

Without this you are optimising blindly.

Attribution dimensions:
    agent × regime          — TrendAgent in trending vs ranging
    agent × session         — LiquidityTrap in London vs Tokyo
    confidence_bin × outcome — does high confidence predict wins?
    regime × session        — which regime × time combination works best
    pair × regime           — XAUUSD in volatile vs EURUSD in trending
    timeframe × regime      — 1H signals in trending vs 4H in ranging

Output per cell:
    n_trades, win_rate, expectancy, avg_r, total_pnl, significance

Statistical significance:
    Binomial test: is win_rate significantly above 50%?
    Min 10 trades for any cell to report significance.

Usage:
    attr = PerformanceAttribution()
    attr.load_from_journal(journal)
    report = attr.full_report()
    best   = attr.best_combinations(top_n=5)
    worst  = attr.worst_combinations(top_n=5)
    print(attr.summary_table())
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_TRADES_FOR_SIGNIFICANCE = 10
CONFIDENCE_BINS = [0.40, 0.55, 0.65, 0.75, 0.85, 1.01]


def _conf_bin(c: float) -> str:
    for i in range(len(CONFIDENCE_BINS) - 1):
        if CONFIDENCE_BINS[i] <= c < CONFIDENCE_BINS[i + 1]:
            return f"{CONFIDENCE_BINS[i]:.2f}-{CONFIDENCE_BINS[i+1]:.2f}"
    return "0.85-1.00"


def _stats(outcomes: List[str], r_mults: List[float]) -> Dict:
    if not outcomes:
        return {"n": 0, "win_rate": None, "expectancy": None, "significant": False}
    n    = len(outcomes)
    wins = sum(1 for o in outcomes if o == "win")
    wr   = wins / n
    exp  = float(np.mean(r_mults)) if r_mults else 0.0

    # Binomial significance test (one-sided, H0: WR <= 0.50)
    significant = False
    if n >= MIN_TRADES_FOR_SIGNIFICANCE:
        # Normal approximation to binomial
        p0    = 0.50
        se    = np.sqrt(p0 * (1 - p0) / n)
        z     = (wr - p0) / se
        significant = z > 1.645   # one-sided 95%

    return {
        "n":          n,
        "wins":       wins,
        "losses":     n - wins,
        "win_rate":   round(wr, 3),
        "expectancy": round(exp, 3),
        "avg_r":      round(float(np.mean(r_mults)), 3) if r_mults else 0,
        "total_pnl":  round(sum(r_mults), 3),
        "significant": significant,
        "z_score":    round((wr - 0.50) / max(np.sqrt(0.25 / n), 1e-9), 3) if n > 0 else 0,
    }


@dataclass
class AttributionCell:
    dimension: str
    key:       str
    stats:     Dict


class PerformanceAttribution:
    """
    Multi-dimensional performance attribution across all trade variables.
    Loads data from TradeJournal and cross-tabulates every combination.
    """

    def __init__(self):
        self._trades: List[Dict] = []

    def load_from_journal(self, journal) -> int:
        """Load all trades from TradeJournal instance."""
        entries = journal.query()
        self._trades = [
            {
                "agent":      e.agent,
                "regime":     e.regime,
                "session":    e.session,
                "pair":       e.pair,
                "timeframe":  e.timeframe,
                "direction":  e.direction,
                "outcome":    e.outcome,
                "r_multiple": e.r_multiple,
                "pnl_usd":    e.pnl_usd,
                "confidence": e.confidence,
                "conf_bin":   _conf_bin(e.confidence),
                "duration_h": e.duration_hours,
            }
            for e in entries
        ]
        logger.info(f"Attribution: loaded {len(self._trades)} trades")
        return len(self._trades)

    def load_from_records(self, records: List[Dict]):
        """Load from raw dict records (from learning store)."""
        self._trades = records

    def attribute(self, dim1: str, dim2: Optional[str] = None) -> Dict[str, Dict]:
        """
        Cross-tabulate trades by one or two dimensions.
        dim1/dim2: any field name in trade dict.
        """
        groups: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
        for t in self._trades:
            v1 = str(t.get(dim1, "unknown"))
            v2 = str(t.get(dim2, "")) if dim2 else ""
            key = f"{v1}×{v2}" if dim2 else v1
            groups[key][0].append(t.get("outcome", ""))
            groups[key][1].append(t.get("r_multiple", 0.0))

        return {
            key: _stats(outcomes, r_mults)
            for key, (outcomes, r_mults) in groups.items()
        }

    def full_report(self) -> Dict[str, Dict]:
        """Full attribution across all key dimension pairs."""
        if not self._trades:
            return {}
        return {
            "by_agent":              self.attribute("agent"),
            "by_regime":             self.attribute("regime"),
            "by_session":            self.attribute("session"),
            "by_pair":               self.attribute("pair"),
            "by_timeframe":          self.attribute("timeframe"),
            "by_direction":          self.attribute("direction"),
            "by_confidence_bin":     self.attribute("conf_bin"),
            "agent_x_regime":        self.attribute("agent",   "regime"),
            "agent_x_session":       self.attribute("agent",   "session"),
            "regime_x_session":      self.attribute("regime",  "session"),
            "pair_x_regime":         self.attribute("pair",    "regime"),
            "timeframe_x_regime":    self.attribute("timeframe", "regime"),
            "confidence_x_regime":   self.attribute("conf_bin", "regime"),
        }

    def best_combinations(self, top_n: int = 5, min_trades: int = 5) -> List[Dict]:
        """Top N combinations by expectancy with minimum trade count."""
        all_cells = []
        for dim_pair in [("agent", "regime"), ("agent", "session"), ("pair", "regime"),
                          ("regime", "session"), ("conf_bin", "regime")]:
            cells = self.attribute(dim_pair[0], dim_pair[1])
            for key, stats in cells.items():
                if stats.get("n", 0) >= min_trades:
                    all_cells.append({
                        "dimension": f"{dim_pair[0]}×{dim_pair[1]}",
                        "key":       key,
                        **stats,
                    })
        return sorted(all_cells, key=lambda x: x.get("expectancy", 0), reverse=True)[:top_n]

    def worst_combinations(self, top_n: int = 5, min_trades: int = 5) -> List[Dict]:
        """Worst N combinations — what to AVOID."""
        all_cells = []
        for dim_pair in [("agent", "regime"), ("agent", "session"), ("pair", "regime"),
                          ("regime", "session")]:
            cells = self.attribute(dim_pair[0], dim_pair[1])
            for key, stats in cells.items():
                if stats.get("n", 0) >= min_trades:
                    all_cells.append({
                        "dimension": f"{dim_pair[0]}×{dim_pair[1]}",
                        "key": key, **stats,
                    })
        return sorted(all_cells, key=lambda x: x.get("expectancy", 0))[:top_n]

    def agent_regime_matrix(self) -> str:
        """Formatted win-rate matrix: agents as rows, regimes as columns."""
        agents  = list(set(t["agent"]  for t in self._trades))
        regimes = list(set(t["regime"] for t in self._trades))
        cells   = self.attribute("agent", "regime")

        col_w = 12
        header = f"{'Agent':<25}" + "".join(f"{r:>{col_w}}" for r in regimes)
        sep    = "-" * len(header)
        lines  = ["\nAgent × Regime Win Rate Matrix:", sep, header, sep]

        for agent in sorted(agents):
            row = f"{agent:<25}"
            for regime in regimes:
                key = f"{agent}×{regime}"
                s   = cells.get(key, {})
                wr  = s.get("win_rate")
                n   = s.get("n", 0)
                cell = f"{wr:.0%}(n={n})" if wr is not None else "N/A"
                row += f"{cell:>{col_w}}"
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)

    def summary_table(self) -> str:
        """One-page attribution summary for Telegram / dashboard."""
        if not self._trades:
            return "No trades to attribute."
        report = self.full_report()
        lines  = [f"📊 Performance Attribution ({len(self._trades)} trades)\n"]

        for dim in ["by_agent", "by_regime", "by_session"]:
            lines.append(f"\n{dim.replace('by_', '').title()}:")
            for key, stats in sorted(
                report.get(dim, {}).items(),
                key=lambda x: x[1].get("expectancy", 0), reverse=True
            ):
                n  = stats.get("n", 0)
                if n < 3:
                    continue
                wr = stats.get("win_rate", 0) or 0
                ev = stats.get("expectancy", 0) or 0
                sig = "✅" if stats.get("significant") else ""
                lines.append(f"  {key:<22} n={n:>3} WR={wr:.0%} EV={ev:+.2f}R {sig}")

        lines.append(f"\n🏆 Top combinations:")
        for cell in self.best_combinations(top_n=3):
            lines.append(
                f"  {cell['dimension']} [{cell['key']}] "
                f"WR={cell.get('win_rate', 0):.0%} "
                f"EV={cell.get('expectancy', 0):+.2f}R "
                f"n={cell.get('n', 0)}"
            )
        lines.append(f"\n⚠️  Worst combinations:")
        for cell in self.worst_combinations(top_n=3):
            lines.append(
                f"  {cell['dimension']} [{cell['key']}] "
                f"WR={cell.get('win_rate', 0):.0%} "
                f"EV={cell.get('expectancy', 0):+.2f}R "
                f"n={cell.get('n', 0)}"
            )
        return "\n".join(lines)
