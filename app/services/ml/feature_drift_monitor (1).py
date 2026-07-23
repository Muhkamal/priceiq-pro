"""
PriceIQ Pro — Feature Drift Monitor v1.0

Detects when live market features drift from the distribution the regime
classifier was trained on. Silent model degradation is the #1 cause of
live-vs-backtest performance gaps.

Method: Population Stability Index (PSI) per feature.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → moderate drift (warn)
    PSI > 0.25  → severe drift (retrain + block signals)

Usage:
    monitor = FeatureDriftMonitor()
    monitor.fit_baseline(training_features)       # call after training classifier

    # In live cycle:
    drift = monitor.check(live_features)
    if drift.severe:
        block_signals()
        trigger_retrain()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

PSI_WARN    = 0.10
PSI_SEVERE  = 0.25
N_BINS      = 10
WINDOW_SIZE = 200   # live feature window to compare against baseline


@dataclass
class DriftReport:
    timestamp:    str
    psi_per_feature: Dict[str, float]
    max_psi:      float
    drifted_features: List[str]   # features with PSI > warn
    severe:       bool            # any feature PSI > severe
    warn:         bool            # any feature PSI > warn
    recommendation: str


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index between two distributions."""
    # Build bins from expected
    min_v = min(expected.min(), actual.min())
    max_v = max(expected.max(), actual.max()) + 1e-9
    bins  = np.linspace(min_v, max_v, n_bins + 1)

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual,   bins=bins)

    # Avoid zeros
    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-4, act_counts / len(actual))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


class FeatureDriftMonitor:
    """
    Tracks live feature distributions vs training baseline.
    Uses PSI (Population Stability Index) per feature.
    """

    FEATURE_KEYS = [
        "trend_strength", "sma_direction", "atr_pct", "atr_expansion",
        "mom5", "mom20", "realised_vol", "range_ratio",
        "close_pos_avg", "volume_ratio", "structure_score", "consecutive_dir",
    ]

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window  = window_size
        self._baseline: Dict[str, np.ndarray] = {}
        self._live_buffer: Dict[str, deque] = {
            k: deque(maxlen=window_size) for k in self.FEATURE_KEYS
        }
        self._fitted  = False
        self._last_report: Optional[DriftReport] = None

    def fit_baseline(self, feature_dicts: List[Dict[str, float]]):
        """
        Call once after training the regime classifier.
        feature_dicts = list of feature dicts from RegimeFeatureExtractor.extract()
        """
        for key in self.FEATURE_KEYS:
            vals = [f.get(key, 0.0) for f in feature_dicts if key in f]
            if vals:
                self._baseline[key] = np.array(vals, dtype=float)
        self._fitted = True
        logger.info(f"FeatureDriftMonitor: baseline fitted on {len(feature_dicts)} samples")

    def update(self, features: Dict[str, float]):
        """Feed one live feature dict into the rolling buffer."""
        for key in self.FEATURE_KEYS:
            if key in features:
                self._live_buffer[key].append(features[key])

    def check(self) -> Optional[DriftReport]:
        """
        Compute PSI across all features.
        Returns None if baseline not fitted or insufficient live data.
        """
        if not self._fitted:
            return None

        min_live = min(len(v) for v in self._live_buffer.values())
        if min_live < 30:
            return None   # need at least 30 live observations

        psi_map: Dict[str, float] = {}
        drifted: List[str] = []

        for key in self.FEATURE_KEYS:
            baseline_arr = self._baseline.get(key)
            live_arr     = np.array(list(self._live_buffer[key]), dtype=float)

            if baseline_arr is None or len(baseline_arr) < 10:
                continue

            try:
                psi_val = _psi(baseline_arr, live_arr)
                psi_map[key] = round(psi_val, 4)
                if psi_val >= PSI_WARN:
                    drifted.append(key)
            except Exception as e:
                logger.debug(f"PSI error for {key}: {e}")
                psi_map[key] = 0.0

        if not psi_map:
            return None

        max_psi = max(psi_map.values())
        severe  = max_psi >= PSI_SEVERE
        warn    = max_psi >= PSI_WARN

        if severe:
            rec = "SEVERE DRIFT: Block signals and trigger regime classifier retraining immediately."
        elif warn:
            rec = "MODERATE DRIFT: Schedule retraining. Monitor closely. Reduce position sizes."
        else:
            rec = "Feature distributions stable. No action required."

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            psi_per_feature=psi_map,
            max_psi=round(max_psi, 4),
            drifted_features=drifted,
            severe=severe,
            warn=warn,
            recommendation=rec,
        )
        self._last_report = report

        if warn:
            logger.warning(
                f"Feature drift detected — max PSI={max_psi:.3f} "
                f"drifted={drifted} {'[SEVERE]' if severe else '[WARN]'}"
            )

        return report

    def get_last_report(self) -> Optional[DriftReport]:
        return self._last_report

    def status_dict(self) -> Dict:
        r = self._last_report
        if not r:
            return {"fitted": self._fitted, "live_samples": min(len(v) for v in self._live_buffer.values())}
        return {
            "fitted":    self._fitted,
            "max_psi":   r.max_psi,
            "severe":    r.severe,
            "warn":      r.warn,
            "drifted":   r.drifted_features,
            "timestamp": r.timestamp,
        }
