"""
PriceIQ Pro — Regime Classifier v1.0
XGBoost probabilistic regime detection with heuristic fallback.
"""
from __future__ import annotations
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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

            # Momentum
            momentum_10 = (c[-1] - c[-10]) / max(c[-10], 1e-9)

            # Volatility ratio
            vol_recent = float(np.std(c[-10:]))
            vol_slow   = float(np.std(c[-30:]))
            vol_ratio  = vol_recent / max(vol_slow, 1e-9)

            # Higher highs / lower lows
            hh = 1 if h[-1] > h[-5] else 0
            ll = 1 if l[-1]  < l[-5]  else 0

            return {
                "atr_pct":        atr_pct,
                "trend_strength": trend_strength,
                "momentum_10":    momentum_10,
                "vol_ratio":      vol_ratio,
                "hh_signal":      float(hh),
                "ll_signal":      float(ll),
                "close_vs_sma20": (c[-1] - sma20) / max(sma20, 1e-9),
            }
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def to_vector(self, features: Dict) -> np.ndarray:
        keys = ["atr_pct","trend_strength","momentum_10",
                "vol_ratio","hh_signal","ll_signal","close_vs_sma20"]
        return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)


def auto_label(candles: list) -> List[int]:
    """Auto-label candles: 0=trending, 1=ranging, 2=volatile."""
    extractor = RegimeFeatureExtractor()
    labels    = []
    for i in range(len(candles)):
        window = candles[max(0, i-54): i+1]
        feats  = extractor.extract(window)
        if not feats:
            labels.append(1)  # default ranging
            continue
        atr  = feats["atr_pct"]
        tstr = abs(feats["trend_strength"])
        if atr > 0.008:
            labels.append(2)   # volatile
        elif tstr > 0.002:
            labels.append(0)   # trending
        else:
            labels.append(1)   # ranging
    return labels


class RegimeClassifier:
    """
    Probabilistic regime classifier.
    Uses XGBoost when trained, heuristic fallback otherwise.
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

    def predict(self, candles: list) -> RegimePrediction:
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

        return self._heuristic(feats)

    def _heuristic(self, feats: Dict) -> RegimePrediction:
        """Rule-based fallback when model not available."""
        atr  = feats.get("atr_pct", 0)
        tstr = abs(feats.get("trend_strength", 0))
        mom  = abs(feats.get("momentum_10", 0))
        vr   = feats.get("vol_ratio", 1.0)

        if atr > 0.008 or vr > 1.8:
            regime   = "volatile"
            trending = 0.15; ranging = 0.15; volatile = 0.70
        elif tstr > 0.002 or mom > 0.003:
            regime   = "trending"
            trending = 0.70; ranging = 0.20; volatile = 0.10
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
