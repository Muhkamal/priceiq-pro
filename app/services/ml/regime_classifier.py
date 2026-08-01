"""
PriceIQ Pro — Regime Classifier v2.0 (ANTI-OVERFIT)

FIXES from v1.1:
    1. FORWARD-LOOKING labels — regime determined by NEXT 20 bars, not current features
    2. Strong regularization — prevents memorizing training data
    3. Time-series cross-validation — respects temporal order
    4. Feature noise injection — forces generalization
    5. Expected accuracy: 65–80% (not 99%)
"""
from __future__ import annotations
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


PAIR_CALIBRATION = {
    "XAUUSD":  {"trend_pct": 0.012, "volatile_atr": 0.008, "range_pct": 0.006},
    "default": {"trend_pct": 0.005, "volatile_atr": 0.004, "range_pct": 0.003},
}


@dataclass
class RegimePrediction:
    regime:     str
    trending:   float
    ranging:    float
    volatile:   float
    confidence: float
    features_used: Dict = None

    def __post_init__(self):
        if self.features_used is None:
            self.features_used = {}


class RegimeFeatureExtractor:
    """Extracts numerical features from candles for regime classification."""

    def extract(self, candles: list) -> Optional[Dict]:
        if len(candles) < 55:
            return None
        try:
            closes = [getattr(c, "close", None) or (c[4] if isinstance(c, (list,tuple)) else 0)
                      for c in candles]
            highs  = [getattr(c, "high",  None) or (c[2] if isinstance(c, (list,tuple)) else 0)
                      for c in candles]
            lows   = [getattr(c, "low",   None) or (c[3] if isinstance(c, (list,tuple)) else 0)
                      for c in candles]
            closes = [float(c) for c in closes if c]
            highs  = [float(h) for h in highs  if h]
            lows   = [float(l) for l in lows   if l]

            if len(closes) < 30:
                return None

            c = np.array(closes[-55:])
            h = np.array(highs[-55:])
            l = np.array(lows[-55:])

            # ATR
            trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                   for i in range(1, len(c))]
            atr     = float(np.mean(trs[-14:]))
            atr_pct = atr / max(c[-1], 1e-9)

            # Trend strength
            sma20  = float(np.mean(c[-20:]))
            sma50  = float(np.mean(c[-50:]))
            trend_strength = (sma20 - sma50) / max(sma50, 1e-9)

            # Linear regression slope
            x = np.arange(len(c))
            slope, _ = np.polyfit(x, c, 1)
            slope_pct = slope / max(np.mean(c), 1e-9)

            # Momentum
            momentum_10 = (c[-1] - c[-10]) / max(c[-10], 1e-9)
            momentum_20 = (c[-1] - c[-20]) / max(c[-20], 1e-9)

            # Volatility ratio
            vol_recent = float(np.std(c[-10:]))
            vol_slow   = float(np.std(c[-30:]))
            vol_ratio  = vol_recent / max(vol_slow, 1e-9)

            # Higher highs / lower lows
            hh = 1 if h[-1] > max(h[-10:-1]) else 0
            ll = 1 if l[-1] < min(l[-10:-1]) else 0

            # Price position in recent range
            recent_range = max(c[-20:]) - min(c[-20:])
            close_pos = (c[-1] - min(c[-20:])) / max(recent_range, 1e-9)

            # ADX-like directional movement
            up_moves   = sum(1 for i in range(1, 11) if c[-i] > c[-i-1])
            down_moves = sum(1 for i in range(1, 11) if c[-i] < c[-i-1])
            directional_bias = abs(up_moves - down_moves) / 10.0

            return {
                "atr_pct":          atr_pct,
                "trend_strength":   trend_strength,
                "slope_pct":        slope_pct,
                "momentum_10":      momentum_10,
                "momentum_20":      momentum_20,
                "vol_ratio":        vol_ratio,
                "hh_signal":        float(hh),
                "ll_signal":        float(ll),
                "close_vs_sma20":   (c[-1] - sma20) / max(sma20, 1e-9),
                "close_pos":        close_pos,
                "directional_bias": directional_bias,
            }
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def to_vector(self, features: Dict) -> np.ndarray:
        keys = ["atr_pct","trend_strength","slope_pct","momentum_10",
                "momentum_20","vol_ratio","hh_signal","ll_signal",
                "close_vs_sma20","close_pos","directional_bias"]
        return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)


def auto_label(candles: list, pair: str = "") -> List[int]:
    """
    FORWARD-LOOKING auto-label.

    A regime is determined by what happens in the NEXT 20 bars,
    not by current indicators. This prevents the model from
    reverse-engineering the heuristic.

    0=trending, 1=ranging, 2=volatile
    """
    cal = PAIR_CALIBRATION.get(pair.upper(), PAIR_CALIBRATION["default"])
    labels = []

    for i in range(len(candles)):
        future = candles[i+1 : i+21]  # next 20 bars
        if len(future) < 10:
            labels.append(1)  # default ranging if no future data
            continue

        future_closes = [getattr(c, "close", 0) for c in future]
        future_closes = [float(c) for c in future_closes if c]
        if len(future_closes) < 5:
            labels.append(1)
            continue

        start = future_closes[0]
        end   = future_closes[-1]
        max_p = max(future_closes)
        min_p = min(future_closes)

        total_range = max_p - min_p
        total_return = abs(end - start)

        # Volatile: large swings regardless of direction
        atr_window = [getattr(c, "high", 0) - getattr(c, "low", 0) for c in future]
        avg_range = np.mean([r for r in atr_window if r > 0])

        # Directional consistency (trend = mostly one direction)
        up_moves = sum(1 for j in range(1, len(future_closes)) if future_closes[j] > future_closes[j-1])
        direction_ratio = up_moves / max(len(future_closes)-1, 1)
        directional = abs(direction_ratio - 0.5) * 2  # 0=chop, 1=one direction

        # Classification
        if total_range / max(start, 1e-9) > cal["volatile_atr"] * 2:
            labels.append(2)   # volatile: huge range
        elif (total_return / max(start, 1e-9) > cal["trend_pct"] 
              and directional > 0.6):
            labels.append(0)   # trending: sustained directional move
        else:
            labels.append(1)   # ranging: oscillation or low momentum

    return labels


class RegimeClassifier:
    """
    Probabilistic regime classifier with ANTI-OVERFIT measures.
    """

    MODEL_PATH_DEFAULT = "regime_model.pkl"

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self.MODEL_PATH_DEFAULT
        self.model      = None
        self.scaler     = None
        self.extractor  = RegimeFeatureExtractor()
        self._trained   = False
        self._try_load()

    def _try_load(self):
        if os.path.exists(self.model_path):
            self.load(self.model_path)
        else:
            logger.info(f"Regime model file not found at {self.model_path} — will use heuristic fallback")

    def load(self, path: str = None) -> bool:
        path = path or self.model_path
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model   = data.get("model")
            self.scaler  = data.get("scaler")
            self._trained = True
            logger.info(f"Regime model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Regime model load failed: {e}")
            return False

    def train(self, candles: list, pair: str = "") -> Dict:
        """
        Train with STRONG regularization to prevent 99% overfitting.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score

        labels = auto_label(candles, pair)
        X_raw, y = [], []
        for i, label in enumerate(labels):
            window = candles[max(0, i-54): i+1]
            feats = self.extractor.extract(window)
            if feats:
                X_raw.append(self.extractor.to_vector(feats))
                y.append(label)

        if len(X_raw) < 100:
            return {"success": False, "reason": f"Only {len(X_raw)} valid samples", "cv_accuracy": 0.0}

        X = np.array(X_raw)
        y = np.array(y)

        # Add noise to features — forces model to learn robust patterns
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, X.shape)
        X_noisy = X + noise

        try:
            import xgboost as xgb
            base = xgb.XGBClassifier(
                n_estimators=80,        # REDUCED from 200
                max_depth=3,            # REDUCED from 4
                learning_rate=0.08,
                subsample=0.7,          # REDUCED from 0.8
                colsample_bytree=0.7,   # REDUCED from 0.8
                reg_alpha=1.0,          # L1 regularization — NEW
                reg_lambda=2.0,         # L2 regularization — NEW
                min_child_weight=5,     # NEW — prevents leaf overfitting
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            base = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.08,
                subsample=0.7, random_state=42,
            )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_noisy)

        # Time-series cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            base.fit(X_tr, y_tr)
            preds = base.predict(X_te)
            cv_scores.append(accuracy_score(y_te, preds))

        cv_acc = float(np.mean(cv_scores)) if cv_scores else 0.0

        # Train final model on full data
        base.fit(X_scaled, y)
        self.model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        self.model.fit(X_scaled, y)
        self._trained = True

        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

        logger.info(f"Regime model trained: CV accuracy={cv_acc:.3f}, samples={len(X)}")
        return {"success": True, "cv_accuracy": cv_acc, "n_samples": len(X)}

    def predict(self, candles: list, pair: str = "") -> RegimePrediction:
        """Predict regime from candle list."""
        feats = self.extractor.extract(candles)
        if not feats:
            return self._default_prediction()

        if self._trained and self.model and self.scaler:
            try:
                vec    = self.extractor.to_vector(feats).reshape(1, -1)
                scaled = self.scaler.transform(vec)
                probs  = self.model.predict_proba(scaled)[0]
                regime_map = {0: "trending", 1: "ranging", 2: "volatile"}
                best_idx   = int(np.argmax(probs))
                return RegimePrediction(
                    regime=regime_map[best_idx],
                    trending=round(float(probs[0]), 4),
                    ranging=round(float(probs[1]),  4),
                    volatile=round(float(probs[2]), 4),
                    confidence=round(float(probs[best_idx]), 4),
                    features_used=feats,
                )
            except Exception as e:
                logger.debug(f"Model predict error: {e}")

        return self._heuristic(feats, pair)

    def _heuristic(self, feats: Dict, pair: str = "") -> RegimePrediction:
        """Rule-based fallback when model not available — PAIR-AWARE."""
        cal = PAIR_CALIBRATION.get(pair.upper(), PAIR_CALIBRATION["default"])

        atr   = feats.get("atr_pct", 0)
        tstr  = abs(feats.get("trend_strength", 0))
        slope = abs(feats.get("slope_pct", 0))
        mom   = abs(feats.get("momentum_10", 0))
        vr    = feats.get("vol_ratio", 1.0)
        hh    = feats.get("hh_signal", 0)
        ll    = feats.get("ll_signal", 0)
        close_pos = feats.get("close_pos", 0.5)
        bias  = feats.get("directional_bias", 0)

        if atr > cal["volatile_atr"] or vr > 2.0:
            regime   = "volatile"
            trending = 0.15; ranging = 0.15; volatile = 0.70
        elif (tstr > cal["trend_pct"] 
              or slope > cal["trend_pct"] * 0.6
              or (mom > cal["trend_pct"] * 0.5 and bias > 0.5)
              or (close_pos > 0.90 and hh)
              or (close_pos < 0.10 and ll)):
            regime   = "trending"
            trending = 0.75; ranging = 0.15; volatile = 0.10
        else:
            regime   = "ranging"
            trending = 0.20; ranging = 0.65; volatile = 0.15

        conf = {"trending": trending, "ranging": ranging, "volatile": volatile}[regime]
        return RegimePrediction(
            regime=regime,
            trending=trending, ranging=ranging, volatile=volatile,
            confidence=conf, features_used=feats,
        )

    def _default_prediction(self) -> RegimePrediction:
        return RegimePrediction(
            regime="ranging", trending=0.33,
            ranging=0.34, volatile=0.33,
            confidence=0.34, features_used={},
        )
