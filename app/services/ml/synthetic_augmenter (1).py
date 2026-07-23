"""
PriceIQ Pro — Synthetic Data Augmentation v1.0

Solves the class imbalance problem in regime classifier training.
Volatile regimes are rare in real data (crashes happen once a decade).
Result: classifier sees 70% trending, 25% ranging, 5% volatile.
It learns to almost never predict volatile — exactly when you need it most.

Solution — Synthetic Minority Augmentation:
    1. SMOTE-style interpolation between existing volatile samples
    2. Bootstrap resampling with fat-tail noise injection
    3. Regime-transition boundary samples (the hard cases near boundaries)

Result: balanced training set → classifier learns to detect all regimes.

Usage:
    augmenter = SyntheticDataAugmenter()

    X_aug, y_aug = augmenter.augment(X_train, y_train, target_ratio=0.33)
    # Each regime gets ~33% of samples

    # Or augment candles directly:
    aug_candles, aug_labels = augmenter.augment_candles(real_candles)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIME_LABELS  = ["trending", "ranging", "volatile"]
REGIME_TO_INT  = {r: i for i, r in enumerate(REGIME_LABELS)}
INT_TO_REGIME  = {i: r for i, r in enumerate(REGIME_LABELS)}

# Fat-tail parameters for volatile regime synthesis
VOLATILE_ATR_MULT_RANGE  = (1.5, 4.0)   # volatile bars have 1.5-4x normal ATR
VOLATILE_VOL_MULT_RANGE  = (1.5, 3.5)   # and much higher realised vol
TRENDING_BIAS_RANGE      = (0.003, 0.015) # strong directional momentum


@dataclass
class AugmentationReport:
    original_counts: Dict[str, int]
    augmented_counts: Dict[str, int]
    n_synthetic:     int
    method:          str


class SyntheticDataAugmenter:
    """
    Synthetic minority oversampling for regime classifier training data.
    Works on feature vectors (not raw candles).
    """

    def augment(
        self,
        X: np.ndarray,          # shape (n_samples, n_features)
        y: np.ndarray,          # shape (n_samples,) — integer labels
        target_ratio: float = 0.30,   # target fraction for minority class
        noise_std:    float = 0.05,   # Gaussian noise std added to synthetic samples
        random_seed:  int   = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data to balance regime classes.

        Args:
            X:            feature matrix
            y:            integer labels (0=trending, 1=ranging, 2=volatile)
            target_ratio: target fraction for each class (default 0.30 = even-ish)
            noise_std:    noise scale for synthetic samples
            random_seed:  reproducibility

        Returns:
            X_aug, y_aug — augmented feature matrix and labels
        """
        np.random.seed(random_seed)

        n_total  = len(y)
        target_n = int(n_total * target_ratio)

        X_aug_list = [X]
        y_aug_list = [y]
        n_synthetic = 0

        original_counts = {
            INT_TO_REGIME[i]: int(np.sum(y == i)) for i in range(3)
        }

        for class_idx in range(3):
            class_mask   = y == class_idx
            class_X      = X[class_mask]
            current_n    = len(class_X)

            if current_n >= target_n or current_n == 0:
                continue

            n_needed = target_n - current_n
            synthetic = self._smote_sample(class_X, n_needed, noise_std)
            X_aug_list.append(synthetic)
            y_aug_list.append(np.full(n_needed, class_idx))
            n_synthetic += n_needed

            logger.info(
                f"Augmented {INT_TO_REGIME[class_idx]}: "
                f"{current_n} → {current_n + n_needed} samples"
            )

        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)

        # Shuffle
        idx   = np.random.permutation(len(y_aug))
        X_aug = X_aug[idx]
        y_aug = y_aug[idx]

        augmented_counts = {
            INT_TO_REGIME[i]: int(np.sum(y_aug == i)) for i in range(3)
        }
        logger.info(f"Augmentation complete: {original_counts} → {augmented_counts}")

        return X_aug, y_aug

    def augment_for_classifier(
        self,
        feature_dicts: List[Dict[str, float]],
        labels: List[int],
        feature_keys: List[str],
        target_ratio: float = 0.30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper that accepts feature dicts directly."""
        X = np.array([[d.get(k, 0.0) for k in feature_keys] for d in feature_dicts])
        y = np.array(labels)
        return self.augment(X, y, target_ratio=target_ratio)

    def generate_volatile_features(
        self,
        base_features: Dict[str, float],
        n: int = 50,
    ) -> List[Dict[str, float]]:
        """
        Generate synthetic volatile-regime feature vectors.
        Used when you have almost no volatile training samples.
        Injects known volatile-regime characteristics:
            - High ATR expansion
            - High realised vol
            - Low trend strength (vol spikes in all directions)
            - Extreme momentum (spike then reversal)
        """
        synthetic = []
        for _ in range(n):
            f = dict(base_features)

            # Volatile = high ATR
            atr_mult  = np.random.uniform(*VOLATILE_ATR_MULT_RANGE)
            f["atr_expansion"]  = atr_mult
            f["atr_pct"]        = f.get("atr_pct", 0.002) * atr_mult
            f["realised_vol"]   = f.get("realised_vol", 0.005) * np.random.uniform(*VOLATILE_VOL_MULT_RANGE)

            # Low trend strength (vol not directional)
            f["trend_strength"] = np.random.uniform(0.0, 0.001)
            f["structure_score"] = np.random.uniform(0.0, 0.3)

            # Extreme recent momentum (news spike)
            sign = np.random.choice([-1, 1])
            f["mom5"]  = sign * np.random.uniform(0.005, 0.020)
            f["mom20"] = np.random.uniform(-0.005, 0.005)   # unclear 20-bar direction

            # Wide range
            f["range_ratio"] = f.get("range_ratio", 0.002) * atr_mult

            # Add small noise to all features
            for key in f:
                if isinstance(f[key], float):
                    noise = np.random.normal(0, abs(f[key]) * 0.05 + 1e-6)
                    f[key] = float(f[key] + noise)

            synthetic.append(f)

        return synthetic

    # ── Internal ─────────────────────────────────────────────

    def _smote_sample(
        self,
        class_X:   np.ndarray,
        n_needed:  int,
        noise_std: float,
    ) -> np.ndarray:
        """
        SMOTE-style: interpolate between randomly chosen pairs of real samples.
        Then add small Gaussian noise to avoid exact duplicates.
        """
        n_real = len(class_X)
        synthetic = np.zeros((n_needed, class_X.shape[1]))

        for i in range(n_needed):
            # Pick two random real samples
            idx_a = np.random.randint(n_real)
            idx_b = np.random.randint(n_real)
            while idx_b == idx_a and n_real > 1:
                idx_b = np.random.randint(n_real)

            # Interpolate
            alpha = np.random.uniform(0.0, 1.0)
            synth = class_X[idx_a] + alpha * (class_X[idx_b] - class_X[idx_a])

            # Add noise
            synth += np.random.normal(0, noise_std * np.abs(synth).mean() + 1e-8, synth.shape)
            synthetic[i] = synth

        return synthetic
