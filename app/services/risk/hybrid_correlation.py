"""
PriceIQ Pro — Correlation Estimator with Warmup Seeding v1.1 (FIXED)

Fix: get_correlation() returned a tuple instead of float due to misplaced parenthesis.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CORR_WINDOW_NORMAL   = 60
CORR_WINDOW_VOLATILE = 20
CORR_THRESHOLD       = 0.75
EWM_SPAN             = 30
MIN_OBSERVATIONS     = 20

STATIC_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"):  0.82,
    ("EURUSD", "AUDUSD"):  0.78,
    ("EURUSD", "NZDUSD"):  0.72,
    ("EURUSD", "EURJPY"):  0.65,
    ("GBPUSD", "AUDUSD"):  0.75,
    ("GBPUSD", "GBPJPY"):  0.71,
    ("AUDUSD", "NZDUSD"):  0.88,
    ("AUDUSD", "AUDJPY"):  0.68,
    ("EURUSD", "USDJPY"): -0.72,
    ("EURUSD", "USDCHF"): -0.91,
    ("GBPUSD", "USDCHF"): -0.78,
    ("GBPUSD", "USDJPY"): -0.68,
    ("XAUUSD", "EURUSD"):  0.42,
    ("XAUUSD", "USDCHF"): -0.48,
    ("XAUUSD", "USDJPY"): -0.35,
    ("XAUUSD", "XAGUSD"):  0.80,
    ("USDJPY", "EURJPY"):  0.75,
    ("USDJPY", "GBPJPY"):  0.70,
    ("USDCAD", "AUDUSD"): -0.62,
    ("USDCAD", "EURUSD"): -0.58,
}


def _get_static(pair_a: str, pair_b: str) -> float:
    a, b = pair_a.upper(), pair_b.upper()
    return STATIC_CORRELATIONS.get((a, b), STATIC_CORRELATIONS.get((b, a), 0.0))


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


class HybridCorrelationEstimator:
    def __init__(self, window: int = CORR_WINDOW_NORMAL, threshold: float = CORR_THRESHOLD):
        self._window    = window
        self._threshold = threshold
        self._returns:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_close: Dict[str, float] = {}
        self._obs_count: Dict[str, int]   = defaultdict(int)

    def update(self, pair: str, close: float):
        pair = pair.upper()
        if pair in self._last_close and self._last_close[pair] > 0:
            log_ret = np.log(close / self._last_close[pair])
            if np.isfinite(log_ret):
                self._returns[pair].append(log_ret)
                self._obs_count[pair] += 1
        self._last_close[pair] = close

    def get_correlation(self, pair_a: str, pair_b: str, regime: str = "trending") -> float:
        pair_a, pair_b = pair_a.upper(), pair_b.upper()
        static_corr = _get_static(pair_a, pair_b)

        n_a = self._obs_count.get(pair_a, 0)
        n_b = self._obs_count.get(pair_b, 0)
        n   = min(n_a, n_b)

        w_empirical = min(1.0, n / MIN_OBSERVATIONS)
        w_static    = 1.0 - w_empirical

        if n < 3:
            return round(static_corr, 4)

        window = CORR_WINDOW_VOLATILE if regime == "volatile" else self._window
        rets_a = list(self._returns.get(pair_a, []))
        rets_b = list(self._returns.get(pair_b, []))
        n_live = min(len(rets_a), len(rets_b), window)

        if n_live < 3:
            return round(static_corr, 4)

        a = np.array(rets_a[-n_live:])
        b = np.array(rets_b[-n_live:])

        weights = np.exp(np.linspace(-1, 0, n_live))
        weights /= weights.sum()
        wa = np.average(a, weights=weights)
        wb = np.average(b, weights=weights)
        cov   = np.sum(weights * (a - wa) * (b - wb))
        std_a = np.sqrt(np.sum(weights * (a - wa) ** 2))
        std_b = np.sqrt(np.sum(weights * (b - wb) ** 2))

        if std_a == 0 or std_b == 0:
            live_corr = static_corr
        else:
            live_corr = float(np.clip(cov / (std_a * std_b), -1.0, 1.0))

        hybrid = w_static * static_corr + w_empirical * live_corr

        if n < MIN_OBSERVATIONS:
            logger.debug(
                f"Corr {pair_a}/{pair_b}: warmup {n}/{MIN_OBSERVATIONS} "
                f"static={static_corr:.2f} live={live_corr:.2f} hybrid={hybrid:.2f}"
            )

        # FIX: was returning a tuple due to misplaced parenthesis
        return round(float(np.clip(hybrid, -1.0, 1.0)), 4)

    def warmup_status(self, pairs: List[str]) -> Dict[str, Dict]:
        return {
            pair: {
                "observations": self._obs_count.get(pair.upper(), 0),
                "warmed_up":    self._obs_count.get(pair.upper(), 0) >= MIN_OBSERVATIONS,
                "warmup_pct":   round(min(1.0, self._obs_count.get(pair.upper(), 0) / MIN_OBSERVATIONS), 3),
            }
            for pair in pairs
        }

    def check_correlation_risk(self, pair: str, direction: str, open_positions: Dict, max_correlated: int = 2, regime: str = "trending") -> Dict:
        correlated = []
        for open_pair, pos in open_positions.items():
            if open_pair.upper() == pair.upper():
                continue
            if getattr(pos, "direction", "") != direction:
                continue
            corr = self.get_correlation(pair, open_pair, regime)
            if abs(corr) >= self._threshold:
                correlated.append((open_pair, corr))

        if len(correlated) >= max_correlated:
            details = ", ".join(f"{p}({c:.2f})" for p, c in correlated)
            return {
                "blocked": True,
                "reason": f"Correlation risk: {details}",
                "correlated_pairs": correlated,
            }
        return {"blocked": False, "reason": "OK", "correlated_pairs": correlated}

    def get_correlation_matrix(self, pairs: List[str], regime: str = "trending") -> Dict:
        matrix = {}
        for p1 in pairs:
            matrix[p1] = {}
            for p2 in pairs:
                matrix[p1][p2] = 1.0 if p1 == p2 else self.get_correlation(p1, p2, regime)
        return matrix

    def get_high_correlation_alerts(self, pairs: List[str], regime: str = "trending") -> List[Dict]:
        alerts = []
        for i, p1 in enumerate(pairs):
            for p2 in pairs[i+1:]:
                corr = self.get_correlation(p1, p2, regime)
                if abs(corr) >= self._threshold:
                    alerts.append({"pair_a": p1, "pair_b": p2, "correlation": corr})
        return alerts

    def status_dict(self, pairs: List[str]) -> Dict:
        return {
            "warmup":    self.warmup_status(pairs),
            "threshold": self._threshold,
            "window":    self._window,
        }

    def observations(self, pair: str) -> int:
        return self._obs_count.get(pair.upper(), 0)
