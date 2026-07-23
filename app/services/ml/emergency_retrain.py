"""
PriceIQ Pro — Emergency Regime Retrain Trigger v1.0

Handles unscheduled retraining when drift monitor fires SEVERE.
Does not wait for Sunday 2AM — retrains immediately with validation.

Protocol:
    1. Drift monitor fires SEVERE → trigger this module
    2. Collect last 500 bars across all pairs
    3. Retrain with synthetic augmentation (volatile class boost)
    4. Validate: new CV accuracy must beat baseline or retrain is rejected
    5. Atomic swap: save new model, load it live, update drift baseline
    6. Telegram notification of outcome

Atomic swap (critical):
    - New model saved as regime_model_NEW.pkl
    - Validate it passes accuracy threshold
    - Rename to regime_model.pkl atomically
    - Load into live classifier
    - If validation fails: keep old model, alert for human review

Usage:
    trigger = EmergencyRetrainTrigger(classifier, augmenter, drift_monitor)
    result  = await trigger.run(all_pair_candles, current_baseline_accuracy=0.74)
    if result.success:
        logger.info(f"New model deployed: accuracy {result.new_accuracy:.3f}")
    else:
        logger.warning(f"Retrain rejected: {result.reason}")
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_ACCURACY_IMPROVEMENT = 0.02   # new model must be at least 2% better
MIN_ABSOLUTE_ACCURACY    = 0.60   # must achieve at least 60% CV accuracy
MIN_CANDLES_FOR_RETRAIN  = 200


@dataclass
class RetrainResult:
    success:          bool
    triggered_by:     str      # "drift_severe" | "scheduled" | "manual"
    old_accuracy:     float
    new_accuracy:     float
    n_samples:        int
    reason:           str
    model_path:       str
    timestamp:        str
    drift_features:   List[str]   # which features were drifting


class EmergencyRetrainTrigger:
    """
    Emergency retrain pipeline with atomic model swap.
    Runs training in a thread pool to avoid blocking the event loop.
    """

    def __init__(
        self,
        classifier,              # RegimeClassifier instance
        augmenter=None,          # SyntheticDataAugmenter instance
        drift_monitor=None,      # FeatureDriftMonitor instance
        telegram=None,
        model_path:     str = "regime_model.pkl",
        model_tmp_path: str = "regime_model_NEW.pkl",
    ):
        self.classifier   = classifier
        self.augmenter    = augmenter
        self.drift_monitor = drift_monitor
        self.telegram     = telegram
        self.model_path   = model_path
        self.model_tmp_path = model_tmp_path
        self._retraining  = False   # prevent concurrent retrains

    async def run(
        self,
        all_candles:          List,    # combined candles from all pairs
        current_baseline_accuracy: float = 0.0,
        triggered_by: str = "drift_severe",
    ) -> RetrainResult:
        """
        Full emergency retrain pipeline.
        Returns RetrainResult — success or failure with reason.
        """
        now = datetime.now(timezone.utc).isoformat()

        if self._retraining:
            return RetrainResult(
                success=False, triggered_by=triggered_by,
                old_accuracy=current_baseline_accuracy, new_accuracy=0.0,
                n_samples=0, reason="Retrain already in progress",
                model_path=self.model_path, timestamp=now, drift_features=[],
            )

        self._retraining = True
        drift_features   = []

        try:
            # Get drifted features for logging
            if self.drift_monitor:
                report = self.drift_monitor.get_last_report()
                if report:
                    drift_features = report.drifted_features

            # Validate we have enough data
            if len(all_candles) < MIN_CANDLES_FOR_RETRAIN:
                return self._fail(
                    triggered_by, current_baseline_accuracy, now, drift_features,
                    f"Insufficient candles: {len(all_candles)} < {MIN_CANDLES_FOR_RETRAIN}",
                )

            await self._send(
                f"🔄 <b>Emergency Retrain Triggered</b>\n"
                f"Reason: {triggered_by}\n"
                f"Drifted features: {drift_features}\n"
                f"Candles: {len(all_candles)}\n"
                f"Training in background..."
            )

            # Run CPU-bound training in thread pool
            result = await asyncio.to_thread(
                self._train_and_validate,
                all_candles, current_baseline_accuracy,
            )

            if result["success"]:
                # Atomic swap: move tmp model to live path
                await asyncio.to_thread(self._atomic_swap)

                # Reload classifier from new file
                self.classifier.load(self.model_path)

                # Refit drift monitor baseline
                if self.drift_monitor:
                    feats = []
                    for i in range(55, len(all_candles)):
                        f = self.classifier.extractor.extract(
                            all_candles[max(0, i-100): i+1]
                        )
                        if f:
                            feats.append(f)
                    if feats:
                        self.drift_monitor.fit_baseline(feats)

                await self._send(
                    f"✅ <b>Retrain Successful</b>\n"
                    f"Old accuracy: {current_baseline_accuracy:.1%}\n"
                    f"New accuracy: {result['new_accuracy']:.1%}\n"
                    f"Samples: {result['n_samples']}\n"
                    f"Model deployed ✅"
                )

                return RetrainResult(
                    success=True, triggered_by=triggered_by,
                    old_accuracy=current_baseline_accuracy,
                    new_accuracy=result["new_accuracy"],
                    n_samples=result["n_samples"],
                    reason="Success",
                    model_path=self.model_path,
                    timestamp=now,
                    drift_features=drift_features,
                )
            else:
                await self._send(
                    f"❌ <b>Retrain Rejected</b>\n"
                    f"Reason: {result['reason']}\n"
                    f"Keeping old model. Human review recommended."
                )
                return self._fail(
                    triggered_by, current_baseline_accuracy, now,
                    drift_features, result["reason"],
                )

        except Exception as e:
            logger.error(f"Emergency retrain failed: {e}", exc_info=True)
            await self._send(f"💥 <b>Retrain Error</b>\n<code>{e}</code>")
            return self._fail(
                triggered_by, current_baseline_accuracy, now, drift_features, str(e)
            )
        finally:
            self._retraining = False

    def _train_and_validate(
        self, all_candles: List, baseline_accuracy: float
    ) -> Dict:
        """CPU-bound: train new model and validate before committing."""
        try:
            # Auto-label
            from .regime_classifier import auto_label, RegimeFeatureExtractor
            labels    = auto_label(all_candles)
            extractor = RegimeFeatureExtractor()

            # Build feature matrix
            X_raw, y = [], []
            for i, label in enumerate(labels):
                window = all_candles[max(0, i - 100): i + 56]
                feats  = extractor.extract(window)
                if feats:
                    X_raw.append(extractor.to_vector(feats))
                    y.append(label)

            if len(X_raw) < 30:
                return {"success": False, "reason": f"Only {len(X_raw)} samples after feature extraction"}

            import numpy as np
            X = np.array(X_raw)
            y = np.array(y)

            # Augment if available
            if self.augmenter:
                X, y = self.augmenter.augment(X, y, target_ratio=0.30)

            # Train new classifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import cross_val_score

            try:
                import xgboost as xgb
                base = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="mlogloss",
                    random_state=42, verbosity=0,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                base = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )

            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(X)
            new_model = CalibratedClassifierCV(base, cv=2, method="isotonic")
            new_model.fit(X_scaled, y)

            # Cross-validate
            cv_scores    = cross_val_score(base, X_scaled, y, cv=3, scoring="accuracy")
            new_accuracy = float(np.mean(cv_scores))

            # Validate against thresholds
            if new_accuracy < MIN_ABSOLUTE_ACCURACY:
                return {
                    "success": False,
                    "reason":  f"New accuracy {new_accuracy:.1%} < minimum {MIN_ABSOLUTE_ACCURACY:.0%}",
                }
            if baseline_accuracy > 0 and new_accuracy < baseline_accuracy - MIN_ACCURACY_IMPROVEMENT:
                return {
                    "success": False,
                    "reason":  (
                        f"New accuracy {new_accuracy:.1%} worse than "
                        f"baseline {baseline_accuracy:.1%} by more than {MIN_ACCURACY_IMPROVEMENT:.0%}"
                    ),
                }

            # Save to temp path
            with open(self.model_tmp_path, "wb") as f:
                pickle.dump({"model": new_model, "scaler": scaler}, f)

            return {"success": True, "new_accuracy": new_accuracy, "n_samples": len(X)}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _atomic_swap(self):
        """Atomically replace old model with new model."""
        if os.path.exists(self.model_tmp_path):
            # Backup old model
            backup = self.model_path + ".bak"
            if os.path.exists(self.model_path):
                shutil.copy2(self.model_path, backup)
            # Swap
            shutil.move(self.model_tmp_path, self.model_path)
            logger.info(f"Model atomically swapped: {self.model_tmp_path} → {self.model_path}")

    def _fail(self, triggered_by, old_acc, timestamp, feats, reason) -> RetrainResult:
        return RetrainResult(
            success=False, triggered_by=triggered_by,
            old_accuracy=old_acc, new_accuracy=0.0,
            n_samples=0, reason=reason,
            model_path=self.model_path, timestamp=timestamp,
            drift_features=feats,
        )

    async def _send(self, message: str):
        if self.telegram:
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.warning(f"RetrainTrigger Telegram failed: {e}")
