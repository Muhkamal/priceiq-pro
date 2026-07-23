"""
PriceIQ Pro — Augmentation A/B Validator v1.0

Measures whether synthetic data augmentation actually improves
the regime classifier on held-out volatile periods.

Without this, you don't know if SMOTE helps or hurts.
Augmentation can degrade performance if synthetic samples
introduce noise that confuses the classifier.

Protocol:
    1. Split data: train (70%) / validate (30%)
    2. Find volatile periods in validation set (hold out for testing)
    3. Train Model A: raw data only
    4. Train Model B: raw + augmented data
    5. Evaluate both on:
        a. Overall CV accuracy
        b. Volatile-class F1 score (the rare class we're trying to fix)
        c. Trending/ranging accuracy (ensure we didn't break these)
    6. Report improvement/regression with recommendation

Decision rule:
    Use augmentation if:
        volatile_f1_B > volatile_f1_A + 0.05   (5% F1 improvement on volatile)
        AND overall_acc_B >= overall_acc_A - 0.02  (no worse than 2% overall)
    Otherwise: use raw data

Usage:
    validator = AugmentationValidator()
    result    = validator.run(candles)
    print(result.recommendation)
    if result.use_augmentation:
        train with augmentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_VOLATILE_SAMPLES  = 10    # minimum volatile samples to run meaningful test
F1_IMPROVEMENT_MIN    = 0.05  # volatile F1 must improve by this much
ACCURACY_REGRESSION_MAX = 0.02  # overall accuracy can drop by at most this much


@dataclass
class AugmentationValidationResult:
    raw_overall_accuracy:      float
    aug_overall_accuracy:      float
    raw_volatile_f1:           float
    aug_volatile_f1:           float
    raw_trending_accuracy:     float
    aug_trending_accuracy:     float
    raw_ranging_accuracy:      float
    aug_ranging_accuracy:      float
    n_train:                   int
    n_validate:                int
    n_volatile_validate:       int
    use_augmentation:          bool
    recommendation:            str
    accuracy_delta:            float   # aug - raw (positive = improved)
    volatile_f1_delta:         float


class AugmentationValidator:
    """
    A/B tests raw vs augmented training data on held-out volatile periods.
    """

    def run(
        self,
        candles:        List,
        augmenter=None,
        classifier_cls=None,
        test_split:     float = 0.30,
        n_aug_ratio:    float = 0.30,
    ) -> AugmentationValidationResult:
        """
        Run A/B validation.

        Args:
            candles:       full candle history
            augmenter:     SyntheticDataAugmenter instance (or None → uses default)
            classifier_cls: RegimeClassifier class (or None → imports default)
            test_split:    fraction of data held out for validation
            n_aug_ratio:   target class ratio for augmentation
        """
        try:
            from ml.regime_classifier import RegimeClassifier, RegimeFeatureExtractor, auto_label
            from ml.synthetic_augmenter import SyntheticDataAugmenter

            if augmenter is None:
                augmenter = SyntheticDataAugmenter()

            extractor = RegimeFeatureExtractor()
            labels    = auto_label(candles)

            # Build feature matrix
            X_raw, y = [], []
            for i, label in enumerate(labels):
                window = candles[max(0, i - 100): i + 56]
                feats  = extractor.extract(window)
                if feats:
                    X_raw.append(extractor.to_vector(feats))
                    y.append(label)

            if len(X_raw) < 50:
                return self._insufficient(len(X_raw))

            X   = np.array(X_raw)
            y   = np.array(y)

            # Split: train / validate
            split_idx = int(len(X) * (1 - test_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            n_volatile_val = int(np.sum(y_val == 2))   # class 2 = volatile
            if n_volatile_val < 3:
                logger.warning(
                    f"AugValidation: only {n_volatile_val} volatile samples in validation set. "
                    f"Results may not be meaningful."
                )

            # ── Model A: raw data ─────────────────────────────
            acc_a, f1_a_volatile, acc_a_trend, acc_a_range = self._train_and_eval(
                X_train, y_train, X_val, y_val
            )

            # ── Model B: augmented data ───────────────────────
            X_aug, y_aug = augmenter.augment(X_train, y_train, target_ratio=n_aug_ratio)
            acc_b, f1_b_volatile, acc_b_trend, acc_b_range = self._train_and_eval(
                X_aug, y_aug, X_val, y_val
            )

            # ── Decision ──────────────────────────────────────
            f1_improved  = (f1_b_volatile - f1_a_volatile) >= F1_IMPROVEMENT_MIN
            acc_not_hurt = (acc_b - acc_a) >= -ACCURACY_REGRESSION_MAX
            use_aug      = f1_improved and acc_not_hurt

            if use_aug:
                rec = (
                    f"✅ USE AUGMENTATION: volatile F1 improved by "
                    f"{f1_b_volatile - f1_a_volatile:.1%} "
                    f"(raw={f1_a_volatile:.1%} → aug={f1_b_volatile:.1%}) "
                    f"with acceptable overall accuracy change "
                    f"({acc_a:.1%} → {acc_b:.1%})."
                )
            elif not f1_improved:
                rec = (
                    f"❌ SKIP AUGMENTATION: volatile F1 improvement "
                    f"{f1_b_volatile - f1_a_volatile:+.1%} < {F1_IMPROVEMENT_MIN:.0%} threshold. "
                    f"Augmentation did not help volatile detection."
                )
            else:
                rec = (
                    f"❌ SKIP AUGMENTATION: overall accuracy dropped by "
                    f"{acc_a - acc_b:.1%} > {ACCURACY_REGRESSION_MAX:.0%} tolerance. "
                    f"Augmentation hurt trending/ranging classes."
                )

            logger.info(f"Augmentation A/B: {rec}")

            return AugmentationValidationResult(
                raw_overall_accuracy=round(acc_a, 4),
                aug_overall_accuracy=round(acc_b, 4),
                raw_volatile_f1=round(f1_a_volatile, 4),
                aug_volatile_f1=round(f1_b_volatile, 4),
                raw_trending_accuracy=round(acc_a_trend, 4),
                aug_trending_accuracy=round(acc_b_trend, 4),
                raw_ranging_accuracy=round(acc_a_range, 4),
                aug_ranging_accuracy=round(acc_b_range, 4),
                n_train=len(X_train),
                n_validate=len(X_val),
                n_volatile_validate=n_volatile_val,
                use_augmentation=use_aug,
                recommendation=rec,
                accuracy_delta=round(acc_b - acc_a, 4),
                volatile_f1_delta=round(f1_b_volatile - f1_a_volatile, 4),
            )

        except Exception as e:
            logger.error(f"AugmentationValidator error: {e}", exc_info=True)
            return self._insufficient(0, str(e))

    def _train_and_eval(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Train a classifier and return (overall_acc, volatile_f1, trend_acc, range_acc)."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Use RandomForest for speed (we're comparing data, not model quality)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_tr_s, y_train)
        preds = clf.predict(X_val_s)

        overall_acc = float(np.mean(preds == y_val))

        # Per-class metrics
        def class_f1(cls_idx: int) -> float:
            tp = np.sum((preds == cls_idx) & (y_val == cls_idx))
            fp = np.sum((preds == cls_idx) & (y_val != cls_idx))
            fn = np.sum((preds != cls_idx) & (y_val == cls_idx))
            prec = tp / max(tp + fp, 1e-9)
            rec  = tp / max(tp + fn, 1e-9)
            return 2 * prec * rec / max(prec + rec, 1e-9)

        def class_acc(cls_idx: int) -> float:
            mask = y_val == cls_idx
            if not np.any(mask):
                return 1.0
            return float(np.mean(preds[mask] == cls_idx))

        return overall_acc, class_f1(2), class_acc(0), class_acc(1)

    def _insufficient(self, n: int, error: str = "") -> AugmentationValidationResult:
        msg = f"Insufficient data ({n} samples)" + (f": {error}" if error else "")
        return AugmentationValidationResult(
            raw_overall_accuracy=0.0, aug_overall_accuracy=0.0,
            raw_volatile_f1=0.0, aug_volatile_f1=0.0,
            raw_trending_accuracy=0.0, aug_trending_accuracy=0.0,
            raw_ranging_accuracy=0.0, aug_ranging_accuracy=0.0,
            n_train=0, n_validate=0, n_volatile_validate=0,
            use_augmentation=False, recommendation=msg,
            accuracy_delta=0.0, volatile_f1_delta=0.0,
        )
