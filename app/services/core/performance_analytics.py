"""PriceIQ Pro — Institutional Performance Analytics (Sharpe/Sortino/CAGR/PF/MaxDD)."""
import json, logging, threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    def __init__(self, path="performance_history.json", starting_balance=100.0):
        self._path, self._lock, self._start = path, threading.Lock(), starting_balance
        self._trades: List[dict] = []
        self._load()

    def record(self, pnl_usd, r_multiple, pair="?"):
        with self._lock:
            self._trades.append({"pnl": pnl_usd, "r": r_multiple, "pair": pair,
                                 "at": datetime.now(timezone.utc).isoformat()})
            self._trades = self._trades[-2000:]
            self._save()

    def metrics(self) -> Dict:
        with self._lock: trades = list(self._trades)
        if not trades: return {"n": 0}
        import numpy as np
        eq = [self._start]
        for t in trades: eq.append(eq[-1] + t["pnl"])
        wins = [t for t in trades if t["pnl"] > 0]; losses = [t for t in trades if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in losses)) or 1e-9
        daily = defaultdict(float)
        for t in trades: daily[t["at"][:10]] += t["pnl"]
        bal, rets = self._start, []
        for d in sorted(daily):
            rets.append(daily[d] / max(bal, 1)); bal += daily[d]
        arr = np.array(rets); mean, std = arr.mean(), arr.std()
        ds = arr[arr < 0]; dstd = ds.std() if len(ds) else 0.0
        years = max(len(rets), 1) / 365.0
        peak = mdd = 0.0
        for v in eq:
            peak = max(peak, v); mdd = max(mdd, (peak - v) / max(peak, 1))
        return {"n": len(trades), "win_rate": round(len(wins)/len(trades), 3),
                "profit_factor": round(gw/gl, 2),
                "sharpe": round(mean/std*(252**0.5), 2) if std else 0.0,
                "sortino": round(mean/dstd*(252**0.5), 2) if dstd else 0.0,
                "cagr_pct": round(((eq[-1]/self._start)**(1/years)-1)*100, 1) if years and eq[-1] > 0 else 0.0,
                "max_dd_pct": round(mdd*100, 1), "total_pnl": round(eq[-1]-self._start, 2),
                "avg_r": round(sum(t["r"] for t in trades)/len(trades), 2)}

    def report(self) -> str:
        m = self.metrics()
        if not m.get("n"): return "📊 Performance: no closed trades yet."
        g = "🟢" if m["sharpe"] > 1 and m["profit_factor"] > 1.3 else "🟡" if m["profit_factor"] > 1 else "🔴"
        return (f"{g} <b>INSTITUTIONAL REPORT</b>\nTrades: {m['n']} | WR: {m['win_rate']:.0%} | PF: {m['profit_factor']}\n"
                f"Sharpe: {m['sharpe']} | Sortino: {m['sortino']}\nCAGR: {m['cagr_pct']}% | MaxDD: {m['max_dd_pct']}% | PnL: ${m['total_pnl']:+.2f}")

    def _save(self):
        try:
            with open(self._path, "w") as f: json.dump(self._trades, f)
        except Exception as e: logger.debug(f"perf save: {e}")
    def _load(self):
        try:
            import os
            if os.path.exists(self._path):
                with open(self._path) as f: self._trades = json.load(f)
        except Exception: self._trades = []

perf_analytics = PerformanceAnalytics()
