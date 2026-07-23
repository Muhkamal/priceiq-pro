"""
PriceIQ Pro — ML Regime Classifier v1.0

Replaces the SMA-crossover regime detection with a trained XGBoost classifier.
Outputs regime probabilities (trending / ranging / volatile) not binary labels.

Training:
    RegimeClassifier().train(labeled_candles)   # offline, save model
    RegimeClassifier().save("regime_model.json")

Inference (real-time):
    clf = RegimeClassifier().load("regime_model.json")
    result = clf.predict(candles)
    # → {"regime": "trending", "trending": 0.72, "ranging": 0.18, "volatile": 0.10}

Auto-labeling:
    If you have no labeled data, use auto_label() which creates ground-truth
    labels from ATR / trend slope / ADX heuristics — good enough to bootstrap.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Try XGBoost; fall back to sklearn GradientBoosting ──────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — using sklearn GradientBoostingClassifier")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV


REGIME_LABELS = ["trending", "ranging", "volatile"]
REGIME_TO_INT = {r: i for i, r in enumerate(REGIME_LABELS)}
INT_TO_REGIME = {i: r for i, r in enumerate(REGIME_LABELS)}


@dataclass
class RegimePrediction:
    regime: str            # dominant regime label
    trending:  float       # probability 0–1
    ranging:   float
    volatile:  float
    confidence: float      # max probability = how certain the model is
    features_used: Dict    # for explainability / logging


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


def _safe_mean(vals: List[float], default: float = 0.0) -> float:
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else default


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class RegimeFeatureExtractor:
    """
    Builds the 12-feature observation vector for regime classification.
    All features are normalised to be scale-invariant.
    """

    def extract(self, candles: list) -> Optional[Dict[str, float]]:
        """
        Returns a dict of features or None if insufficient data.
        Requires at least 55 candles.
        """
        if len(candles) < 55:
            return None

        closes  = [c.close for c in candles if hasattr(c, "close") and np.isfinite(c.close)]
        highs   = [c.high  for c in candles if hasattr(c, "high")  and np.isfinite(c.high)]
        lows    = [c.low   for c in candles if hasattr(c, "low")   and np.isfinite(c.low)]
        volumes = [c.volume for c in candles
                   if hasattr(c, "volume") and c.volume and np.isfinite(c.volume)]

        if len(closes) < 50:
            return None

        c50  = closes[-50:]
        c20  = closes[-20:]
        c10  = closes[-10:]
        c5   = closes[-5:]

        sma20 = _safe_mean(c20)
        sma50 = _safe_mean(c50)

        # 1. Trend strength: SMA separation (normalised by price)
        trend_strength = _safe_div(abs(sma20 - sma50), sma50)

        # 2. SMA direction: +1 up, -1 down
        sma_direction = 1.0 if sma20 > sma50 else -1.0

        # 3. ATR normalised (volatility)
        trs = []
        for i in range(1, min(15, len(candles))):
            c, p = candles[-i], candles[-i - 1]
            if all(hasattr(x, a) for x in [c, p] for a in ["high", "low", "close"]):
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                trs.append(tr)
        atr = _safe_mean(trs, 0.001)
        atr_pct = _safe_div(atr, closes[-1])   # normalised to price

        # 4. ATR expansion: current ATR vs 20-bar ATR
        trs_long = []
        for i in range(1, min(22, len(candles))):
            c, p = candles[-i], candles[-i - 1]
            if all(hasattr(x, a) for x in [c, p] for a in ["high", "low", "close"]):
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                trs_long.append(tr)
        atr_long  = _safe_mean(trs_long, 0.001)
        atr_expansion = _safe_div(atr, atr_long)

        # 5. Price momentum: 5-bar / 20-bar return
        mom5  = _safe_div(closes[-1] - closes[-6],  closes[-6])  if len(closes) > 6  else 0.0
        mom20 = _safe_div(closes[-1] - closes[-21], closes[-21]) if len(closes) > 21 else 0.0

        # 6. Volatility of returns (realised vol proxy)
        rets = [_safe_div(closes[i] - closes[i-1], closes[i-1]) for i in range(1, len(c20))]
        realised_vol = float(np.std(rets)) if len(rets) > 2 else 0.0

        # 7. Range ratio: (high - low) / price  for last 5 bars
        if len(highs) >= 5 and len(lows) >= 5:
            rng5 = _safe_div(_safe_mean(highs[-5:]) - _safe_mean(lows[-5:]), closes[-1])
        else:
            rng5 = atr_pct

        # 8. Close position within bar (0=low, 1=high) — last 5 bars avg
        close_positions = []
        for i in range(1, min(6, len(candles))):
            c = candles[-i]
            if hasattr(c, "high") and hasattr(c, "low") and hasattr(c, "close"):
                rng = c.high - c.low
                if rng > 0:
                    close_positions.append(_safe_div(c.close - c.low, rng))
        close_pos_avg = _safe_mean(close_positions, 0.5)

        # 9. Volume ratio: last bar vs 20-bar avg
        if volumes and len(volumes) >= 5:
            vol_ratio = _safe_div(volumes[-1], _safe_mean(volumes[-20:], volumes[-1]))
        else:
            vol_ratio = 1.0

        # 10. Higher-high / lower-low count (trend structure proxy)
        hh_count = sum(1 for i in range(1, min(10, len(highs))) if highs[-i] > highs[-i-1])
        ll_count = sum(1 for i in range(1, min(10, len(lows)))  if lows[-i]  < lows[-i-1])
        structure_score = _safe_div(max(hh_count, ll_count), 9)

        # 11. Consecutive same-direction candles
        same_dir = 0
        if len(candles) >= 5:
            last_dir = 1 if candles[-1].close > candles[-1].open else -1
            for c in reversed(candles[-6:-1]):
                d = 1 if c.close > c.open else -1
                if d == last_dir:
                    same_dir += 1
                else:
                    break
        consecutive_dir = same_dir / 5.0

        return {
            "trend_strength":   round(trend_strength,   6),
            "sma_direction":    round(sma_direction,    1),
            "atr_pct":          round(atr_pct,          6),
            "atr_expansion":    round(atr_expansion,    4),
            "mom5":             round(mom5,             6),
            "mom20":            round(mom20,            6),
            "realised_vol":     round(realised_vol,     6),
            "range_ratio":      round(rng5,             6),
            "close_pos_avg":    round(close_pos_avg,    4),
            "volume_ratio":     round(vol_ratio,        4),
            "structure_score":  round(structure_score,  4),
            "consecutive_dir":  round(consecutive_dir,  4),
        }

    def to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict → ordered numpy vector (stable column order)."""
        keys = [
            "trend_strength", "sma_direction", "atr_pct", "atr_expansion",
            "mom5", "mom20", "realised_vol", "range_ratio",
            "close_pos_avg", "volume_ratio", "structure_score", "consecutive_dir",
        ]
        return [features.get(k, 0.0) for k in keys]


# ============================================================
# AUTO-LABELER (bootstraps labels from heuristics)
# ============================================================

def auto_label(candles: list) -> List[int]:
    """
    Heuristic ground-truth labels for training when no manual labels exist.
    Returns list of ints matching REGIME_LABELS indices.

    Logic:
    - volatile:  ATR expansion > 1.5 and realised_vol > 2× recent avg
    - trending:  |SMA20 - SMA50| / SMA50 > 0.002 and mom20 > 0.005
    - ranging:   everything else
    """
    extractor = RegimeFeatureExtractor()
    labels = []
    for i in range(55, len(candles)):
        window = candles[max(0, i - 100): i + 1]
        feats  = extractor.extract(window)
        if feats is None:
            labels.append(REGIME_TO_INT["ranging"])
            continue

        if feats["atr_expansion"] > 1.5 and feats["realised_vol"] > 0.008:
            labels.append(REGIME_TO_INT["volatile"])
        elif feats["trend_strength"] > 0.002 and abs(feats["mom20"]) > 0.005:
            labels.append(REGIME_TO_INT["trending"])
        else:
            labels.append(REGIME_TO_INT["ranging"])

    return labels


# ============================================================
# REGIME CLASSIFIER
# ============================================================

class RegimeClassifier:
    """
    XGBoost (or sklearn fallback) multi-class regime classifier.
    Predicts market regime probabilities from candle features.

    Usage:
        # Training (offline):
        clf = RegimeClassifier()
        clf.train(all_candles)          # auto-labels if no labels provided
        clf.save("regime_model.pkl")

        # Inference (live):
        clf = RegimeClassifier()
        clf.load("regime_model.pkl")
        pred = clf.predict(recent_candles)
    """

    MODEL_PATH_DEFAULT = "regime_model.pkl"

    def __init__(self):
        self.extractor = RegimeFeatureExtractor()
        self.model     = None
        self.scaler    = StandardScaler()
        self._trained  = False

    # ── Training ────────────────────────────────────────────

    def train(
        self,
        candles: list,
        labels:  Optional[List[int]] = None,
        cv_folds: int = 3,
    ) -> Dict:
        """
        Train the regime classifier.

        Args:
            candles: full candle list (at least 200 recommended)
            labels:  optional list of int labels (auto-generated if None)
            cv_folds: cross-validation folds for eval
        Returns:
            dict with cv_accuracy, n_samples
        """
        if labels is None:
            logger.info("Auto-labeling training data from heuristics...")
            labels = auto_label(candles)

        # Build feature matrix
        X_raw, y = [], []
        for i, label in enumerate(labels):
            window = candles[max(0, i - 100): i + 56]
            feats  = self.extractor.extract(window)
            if feats is None:
                continue
            X_raw.append(self.extractor.to_vector(feats))
            y.append(label)

        if len(X_raw) < 30:
            raise ValueError(f"Insufficient training samples: {len(X_raw)}. Need at least 30.")

        X = self.scaler.fit_transform(np.array(X_raw))
        y = np.array(y)

        if _XGB_AVAILABLE:
            base = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
        else:
            base = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )

        # Calibrate for reliable probabilities
        self.model = CalibratedClassifierCV(base, cv=2, method="isotonic")
        self.model.fit(X, y)
        self._trained = True

        # Cross-validation score
        base_uncalibrated = (
            xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False,
                               eval_metric="mlogloss", random_state=42, verbosity=0)
            if _XGB_AVAILABLE
            else RandomForestClassifier(n_estimators=100, random_state=42)
        )
        cv_scores = cross_val_score(base_uncalibrated, X, y, cv=min(cv_folds, 3), scoring="accuracy")

        result = {
            "n_samples":   len(X_raw),
            "cv_accuracy": round(float(np.mean(cv_scores)), 4),
            "cv_std":      round(float(np.std(cv_scores)),  4),
            "class_dist":  {INT_TO_REGIME[i]: int(np.sum(y == i)) for i in range(3)},
        }
        logger.info(f"Regime classifier trained: {result}")
        return result

    # ── Inference ────────────────────────────────────────────

    def predict(self, candles: list) -> RegimePrediction:
        """
        Predict market regime from recent candles.
        Falls back to heuristic if model not trained.
        """
        feats = self.extractor.extract(candles)
        if feats is None:
            return self._heuristic_fallback(candles)

        if not self._trained or self.model is None:
            logger.debug("Regime model not trained — using heuristic fallback")
            return self._heuristic_fallback(candles)

        try:
            vec   = np.array([self.extractor.to_vector(feats)])
            vec_s = self.scaler.transform(vec)
            probs = self.model.predict_proba(vec_s)[0]

            # Match probs to labels (sklearn orders by class index)
            pred_dict = {INT_TO_REGIME[i]: float(probs[i]) for i in range(len(probs))}
            regime    = max(pred_dict, key=pred_dict.get)

            return RegimePrediction(
                regime=regime,
                trending=pred_dict.get("trending",  0.0),
                ranging=pred_dict.get("ranging",    0.0),
                volatile=pred_dict.get("volatile",  0.0),
                confidence=pred_dict[regime],
                features_used=feats,
            )
        except Exception as e:
            logger.warning(f"Regime prediction error: {e}")
            return self._heuristic_fallback(candles)

    def _heuristic_fallback(self, candles: list) -> RegimePrediction:
        """Fast heuristic when model unavailable — same logic as auto_label."""
        feats = self.extractor.extract(candles)
        if feats is None:
            return RegimePrediction("ranging", 0.3, 0.6, 0.1, 0.6, {})

        if feats["atr_expansion"] > 1.5 and feats["realised_vol"] > 0.008:
            return RegimePrediction("volatile",  0.15, 0.15, 0.70, 0.70, feats)
        if feats["trend_strength"] > 0.002 and abs(feats["mom20"]) > 0.005:
            return RegimePrediction("trending",  0.75, 0.15, 0.10, 0.75, feats)
        return RegimePrediction("ranging",   0.15, 0.75, 0.10, 0.75, feats)

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str = MODEL_PATH_DEFAULT):
        if not self._trained:
            raise RuntimeError("Model not trained — cannot save.")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"Regime model saved to {path}")

    def load(self, path: str = MODEL_PATH_DEFAULT) -> "RegimeClassifier":
        if not os.path.exists(path):
            logger.warning(f"Regime model file not found at {path} — will use heuristic fallback")
            return self
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model   = data["model"]
        self.scaler  = data["scaler"]
        self._trained = True
        logger.info(f"Regime model loaded from {path}")
        return self
