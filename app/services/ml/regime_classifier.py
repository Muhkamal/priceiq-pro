import os
import joblib


class RegimePrediction:
    def __init__(self, regime, trending, ranging, volatile, confidence, features_used):
        self.regime = regime
        self.trending = trending
        self.ranging = ranging
        self.volatile = volatile
        self.confidence = confidence
        self.features_used = features_used

    def __repr__(self):
        return (
            f"RegimePrediction(regime='{self.regime}', "
            f"trending={self.trending}, ranging={self.ranging}, "
            f"volatile={self.volatile}, confidence={self.confidence}, "
            f"features_used={self.features_used})"
        )


class RegimeClassifier:

    MODEL_PATH_DEFAULT = "regime_model.pkl"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or self.MODEL_PATH_DEFAULT
        self.model = None
        self.scaler = None
        self.extractor = None  # make sure you set this externally
        self._trained = False

    # =========================
    # LOAD MODEL (SAFE)
    # =========================
    def load(self, path: str | None = None):
        path = path or self.model_path

        if not os.path.exists(path):
            print(f"⚠️ Model file not found: {path}")
            self._trained = False
            return

        try:
            data = joblib.load(path)
            self.model = data.get("model")
            self.scaler = data.get("scaler")
            self._trained = True
            print("✅ Model loaded")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._trained = False

    # =========================
    # SAVE MODEL
    # =========================
    def save(self, path: str | None = None):
        path = path or self.model_path

        try:
            joblib.dump({
                "model": self.model,
                "scaler": self.scaler
            }, path)
            print(f"💾 Saved model → {path}")

        except Exception as e:
            print(f"❌ Save failed: {e}")

    # =========================
    # PREDICT (RENDER-SAFE)
    # =========================
    def predict(self, df):

        # 🔥 Auto-load model if needed
        if not self._trained:
            self.load()

        # fallback if still not trained
        if not self._trained:
            return self._heuristic_fallback(df)

        if self.extractor is None:
            print("⚠️ Feature extractor missing")
            return self._heuristic_fallback(df)

        try:
            X = self.extractor.transform(df)

            if X is None or len(X) == 0:
                return self._heuristic_fallback(df)

            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[-1]

            return self._format_prediction(probs, X.iloc[-1])

        except Exception as e:
            print(f"❌ Predict failed: {e}")
            return self._heuristic_fallback(df)

    # =========================
    # FORMAT OUTPUT
    # =========================
    def _format_prediction(self, probs, features_row):

        classes = ["trending", "ranging", "volatile"]

        prob_dict = dict(zip(classes, probs))
        regime = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[regime]

        return RegimePrediction(
            regime=regime,
            trending=prob_dict["trending"],
            ranging=prob_dict["ranging"],
            volatile=prob_dict["volatile"],
            confidence=confidence,
            features_used=features_row.to_dict()
        )

    # =========================
    # FALLBACK (NEVER FAIL)
    # =========================
    def _heuristic_fallback(self, df):
        try:
            last = df.iloc[-1]

            atr = last.get("atr_pct", 0)
            trend = last.get("trend_strength", 0)

            if atr > 0.01:
                regime = "volatile"
            elif abs(trend) > 0.002:
                regime = "trending"
            else:
                regime = "ranging"

        except Exception:
            regime = "ranging"

        return RegimePrediction(
            regime=regime,
            trending=0.33,
            ranging=0.33,
            volatile=0.33,
            confidence=0.5,
            features_used={}
        )
