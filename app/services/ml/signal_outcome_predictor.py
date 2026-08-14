"""
Signal Outcome Predictor — Self-learning XGBoost model.
Trains on historical signal features + outcomes to predict win probability.
Retrains automatically every 20 new resolved signals.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — predictor disabled")


class SignalOutcomePredictor:
    """
    ML model that learns:
    [agent, regime, pair, session, confidence, technical_features] → win/loss
    """

    MIN_SAMPLES = 30          # Don't train until we have this many
    RETRAIN_EVERY = 20        # Retrain after this many new labels
    MODEL_PATH = Path("signal_outcome_model.json")
    DATASET_PATH = Path("signal_ml_dataset.jsonl")

    # Feature columns (must match extract_features exactly)
    FEATURE_COLS = [
        "agent_encoded", "regime_encoded", "pair_encoded", "session_encoded",
        "confidence", "ev_raw", "atr_14", "rsi_14",
        "ema20_slope", "ema50_slope", "macd_hist",
        "bb_position", "hour", "day_of_week",
        "volatility_percentile", "trend_strength",
        "candle_body_ratio", "wick_ratio",
    ]

    def __init__(self):
        self.model: Optional[Any] = None
        self._pending: Dict[str, Dict] = {}          # signal_id → features
        self._dataset: List[Dict] = []               # labeled training rows
        self._since_last_train = 0
        self._load_dataset()
        self._load_model()

        # Encoders (simple mapping — grows as we see new categories)
        self._agent_map: Dict[str, int] = {}
        self._regime_map: Dict[str, int] = {}
        self._pair_map: Dict[str, int] = {}
        self._session_map: Dict[str, int] = {}
        self._next_code = 1

    # ── Persistence ───────────────────────────────────────────

    def _load_dataset(self):
        if self.DATASET_PATH.exists():
            try:
                with open(self.DATASET_PATH, "r") as f:
                    self._dataset = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Predictor dataset loaded: {len(self._dataset)} rows")
            except Exception as e:
                logger.warning(f"Dataset load failed: {e}")
                self._dataset = []

    def _save_dataset(self):
        try:
            with open(self.DATASET_PATH, "w") as f:
                for row in self._dataset:
                    f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"Dataset save failed: {e}")

    def _load_model(self):
        if not XGB_AVAILABLE or not self.MODEL_PATH.exists():
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.MODEL_PATH))
            logger.info("Predictor model loaded from disk")
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
            self.model = None

    def _save_model(self):
        if self.model:
            try:
                self.model.save_model(str(self.MODEL_PATH))
            except Exception as e:
                logger.warning(f"Model save failed: {e}")

    # ── Encoding ──────────────────────────────────────────────

    def _encode(self, value: str, mapping: Dict[str, int]) -> int:
        if value not in mapping:
            mapping[value] = self._next_code
            self._next_code += 1
        return mapping[value]

    # ── Feature Extraction ────────────────────────────────────

    def extract_features(self, result, candles: List[Any]) -> Dict[str, float]:
        """Extract numerical features from a signal + its candle context."""
        now = datetime.now(timezone.utc)

        # Safe accessors
        agent = getattr(result, "agent_used", "unknown")
        regime = getattr(result, "regime", "unknown")
        pair = getattr(result, "pair", "UNKNOWN")
        session = getattr(result, "session", "unknown")
        conf = float(getattr(result, "confidence", 0.5))
        ev_raw = float(getattr(result, "expected_value", 0.0))

        # Technical features from candles
        closes = np.array([c.close for c in candles if hasattr(c, "close")])
        highs = np.array([c.high for c in candles if hasattr(c, "high")])
        lows = np.array([c.low for c in candles if hasattr(c, "low")])
        opens = np.array([c.open for c in candles if hasattr(c, "open")])

        if len(closes) < 20:
            # Return minimal features if insufficient data
            return {
                "agent_encoded": self._encode(agent, self._agent_map),
                "regime_encoded": self._encode(regime, self._regime_map),
                "pair_encoded": self._encode(pair, self._pair_map),
                "session_encoded": self._encode(session, self._session_map),
                "confidence": conf,
                "ev_raw": ev_raw,
                "atr_14": 0.001,
                "rsi_14": 50.0,
                "ema20_slope": 0.0,
                "ema50_slope": 0.0,
                "macd_hist": 0.0,
                "bb_position": 0.5,
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "volatility_percentile": 50.0,
                "trend_strength": 20.0,
                "candle_body_ratio": 0.5,
                "wick_ratio": 1.0,
            }

        # ATR
        tr1 = highs[-14:] - lows[-14:]
        tr2 = np.abs(highs[-14:] - np.roll(closes, 1)[-14:])
        tr3 = np.abs(lows[-14:] - np.roll(closes, 1)[-14:])
        atr = float(np.mean(np.maximum(np.maximum(tr1, tr2), tr3)))

        # RSI
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g = np.mean(gains)
        avg_l = np.mean(losses) + 1e-9
        rsi = 100 - (100 / (1 + avg_g / avg_l))

        # EMA slopes
        def _ema(arr, period):
            k = 2 / (period + 1)
            ema = [arr[0]]
            for v in arr[1:]:
                ema.append(v * k + ema[-1] * (1 - k))
            return np.array(ema)

        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        ema20_slope = (ema20[-1] - ema20[-5]) / (ema20[-5] + 1e-9) * 100
        ema50_slope = (ema50[-1] - ema50[-5]) / (ema50[-5] + 1e-9) * 100

        # MACD hist
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = ema12[-len(ema26):] - ema26
        macd_signal = _ema(macd_line, 9)
        macd_hist = float(macd_line[-1] - macd_signal[-1]) if len(macd_line) == len(macd_signal) else 0.0

        # Bollinger position (0 = lower band, 1 = upper band)
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_position = (closes[-1] - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)
        bb_position = max(0.0, min(1.0, bb_position))

        # Volatility percentile (current ATR vs last 50 bars)
        atr_history = []
        for i in range(min(50, len(closes) - 14)):
            segment_highs = highs[-(14+i):-i if i else None]
            segment_lows = lows[-(14+i):-i if i else None]
            if len(segment_highs) == 14:
                tr = np.mean(np.maximum(segment_highs - segment_lows,
                            np.abs(segment_highs - np.roll(closes, 1)[-(14+i):-i if i else None])))
                atr_history.append(tr)
        vol_pct = 50.0
        if atr_history:
            vol_pct = float(np.mean(np.array(atr_history) < atr) * 100)

        # Trend strength (ADX approx)
        trend_str = float(np.mean(tr1) / (np.mean(closes[-14:]) + 1e-9) * 100)

        # Candle body / wick
        last_body = abs(opens[-1] - closes[-1])
        last_range = highs[-1] - lows[-1]
        body_ratio = last_body / (last_range + 1e-9)
        wick_ratio = (last_range - last_body) / (last_body + 1e-9)

        return {
            "agent_encoded": self._encode(agent, self._agent_map),
            "regime_encoded": self._encode(regime, self._regime_map),
            "pair_encoded": self._encode(pair, self._pair_map),
            "session_encoded": self._encode(session, self._session_map),
            "confidence": conf,
            "ev_raw": ev_raw,
            "atr_14": round(atr, 6),
            "rsi_14": round(rsi, 2),
            "ema20_slope": round(ema20_slope, 4),
            "ema50_slope": round(ema50_slope, 4),
            "macd_hist": round(macd_hist, 6),
            "bb_position": round(bb_position, 3),
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "volatility_percentile": round(vol_pct, 1),
            "trend_strength": round(trend_str, 2),
            "candle_body_ratio": round(body_ratio, 3),
            "wick_ratio": round(wick_ratio, 2),
        }

    # ── Prediction ─────────────────────────────────────────────

    def predict(self, result, candles: List[Any]) -> tuple:
        """
        Returns (win_probability, is_cold_start).
        If cold start (not enough data), returns agent's own confidence as fallback.
        """
        features = self.extract_features(result, candles)
        signal_id = str(uuid.uuid4())[:8]

        # Store pending for later labeling
        self._pending[signal_id] = {
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": getattr(result, "pair", "UNKNOWN"),
            "direction": getattr(result, "direction", ""),
        }

        if not XGB_AVAILABLE or self.model is None or len(self._dataset) < self.MIN_SAMPLES:
            # Cold start: blend agent confidence with conservative prior
            agent_conf = float(getattr(result, "confidence", 0.5))
            cold_prob = 0.6 * agent_conf + 0.4 * 0.5
            return round(cold_prob, 3), True, signal_id

        try:
            X = np.array([[features.get(c, 0) for c in self.FEATURE_COLS]])
            prob = self.model.predict_proba(X)[0][1]  # class 1 = win
            # Conservative blend: 70% model, 30% agent
            agent_conf = float(getattr(result, "confidence", 0.5))
            blended = 0.7 * prob + 0.3 * agent_conf
            return round(blended, 3), False, signal_id
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.5, True, signal_id

    # ── Outcome Recording ─────────────────────────────────────

    async def on_signal_resolved(self, signal_id: str, outcome: str, pnl_r: float):
        """
        Call when a signal hits SL, TP1, TP2, or times out.
        outcome: 'win' (TP hit), 'loss' (SL hit), 'breakeven', 'timeout'
        """
        pending = self._pending.pop(signal_id, None)
        if not pending:
            logger.debug(f"No pending signal {signal_id} for model training")
            return

        # Label: 1 = win (TP1 or TP2 hit), 0 = loss (SL or negative timeout)
        label = 1 if outcome in ("tp1", "tp2", "win") else 0

        row = pending["features"].copy()
        row["label"] = label
        row["pnl_r"] = round(pnl_r, 3)
        row["outcome"] = outcome
        row["resolved_at"] = datetime.now(timezone.utc).isoformat()

        self._dataset.append(row)
        self._since_last_train += 1
        self._save_dataset()

        logger.info(f"Model training sample added: {outcome} (label={label}, n={len(self._dataset)})")

        # Auto-retrain
        if len(self._dataset) >= self.MIN_SAMPLES and self._since_last_train >= self.RETRAIN_EVERY:
            await self._train()

    # ── Training ──────────────────────────────────────────────

    async def _train(self):
        if not XGB_AVAILABLE:
            return

        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed — cannot train model")
            return

        try:
            df = pd.DataFrame(self._dataset)
            X = df[self.FEATURE_COLS].fillna(0)
            y = df["label"]

            # Time-based split: last 20% is test
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            if len(y_test) < 5:
                # Not enough for validation — use all for training
                X_train, y_train = X, y
                X_test, y_test = X, y

            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Evaluate
            train_acc = float(model.score(X_train, y_train))
            test_acc = float(model.score(X_test, y_test)) if len(y_test) > 0 else 0.0

            self.model = model
            self._since_last_train = 0
            self._save_model()

            # Feature importance log
            importance = dict(zip(self.FEATURE_COLS, model.feature_importances_.tolist()))
            top3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

            logger.info(
                f"🧠 Model retrained | samples={len(self._dataset)} "
                f"| train_acc={train_acc:.2%} test_acc={test_acc:.2%} "
                f"| top_features={[f[0] for f in top3]}"
            )

        except Exception as e:
            logger.error(f"Model training failed: {e}")

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        if not self._dataset:
            return {"status": "cold_start", "samples": 0}

        wins = sum(1 for r in self._dataset if r["label"] == 1)
        total = len(self._dataset)
        recent = self._dataset[-50:]
        recent_wins = sum(1 for r in recent if r["label"] == 1)

        return {
            "status": "trained" if self.model else "collecting",
            "total_samples": total,
            "win_rate": round(wins / total, 3) if total else 0,
            "recent_50_wr": round(recent_wins / len(recent), 3) if recent else 0,
            "model_loaded": self.model is not None,
            "pending_unlabeled": len(self._pending),
        }


# Singleton
outcome_predictor = SignalOutcomePredictor()
