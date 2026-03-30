"""
star/augment.py
===============
Data augmentation strategies for spatial transcriptomics graphs.

Three strategies (all compatible with NumPy and PyTorch inputs):
  1. Feature masking   — zero out random fraction of gene expression values
  2. Edge dropout      — remove random fraction of spatial k-NN edges
  3. Gaussian noise    — add N(0, σ²) to log-normalised expression

Student and teacher views are independently sampled (different rng seeds),
following the MoCo / BYOL self-supervised learning recipe.
"""

import numpy as np
from typing import Tuple
import logging

log = logging.getLogger("StaR.augment")


class SpatialAugmentor:
    """
    Stochastic augmentations for spatial transcriptomics graph data.

    Parameters
    ----------
    feat_mask_rate : float  fraction of genes zeroed per spot (default 0.10)
    edge_drop_rate : float  fraction of spatial edges removed (default 0.10)
    feat_noise_std : float  Gaussian noise σ on log-expression (default 0.05)
    strategy       : str   'random' | 'mask' | 'noise' | 'both'

    Example
    -------
    aug = SpatialAugmentor()
    Xs, ei_s = aug.augment(X, edge_index, rng=np.random.RandomState(0))
    Xt, ei_t = aug.augment(X, edge_index, rng=np.random.RandomState(1))
    """

    def __init__(self, feat_mask_rate: float = 0.10,
                 edge_drop_rate:  float = 0.10,
                 feat_noise_std:  float = 0.05,
                 strategy: str = "random"):
        assert 0 <= feat_mask_rate < 1
        assert 0 <= edge_drop_rate < 1
        assert feat_noise_std >= 0
        self.feat_mask_rate = feat_mask_rate
        self.edge_drop_rate = edge_drop_rate
        self.feat_noise_std = feat_noise_std
        self.strategy       = strategy

    # ── Individual ops ────────────────────────────────────────────────────────

    def feature_mask(self, X: np.ndarray,
                     rng: np.random.RandomState) -> np.ndarray:
        """Zero out genes with probability feat_mask_rate."""
        mask = rng.binomial(1, 1 - self.feat_mask_rate,
                             X.shape).astype(np.float32)
        return X * mask

    def edge_dropout(self, ei: np.ndarray,
                     rng: np.random.RandomState) -> np.ndarray:
        """Remove edges uniformly at random."""
        E    = ei.shape[1]
        keep = rng.binomial(1, 1 - self.edge_drop_rate, E).astype(bool)
        if keep.sum() < E // 2:      # safety: keep at least half
            keep = np.ones(E, bool)
        return ei[:, keep]

    def gaussian_noise(self, X: np.ndarray,
                       rng: np.random.RandomState) -> np.ndarray:
        """Add Gaussian noise to log-normalised expression."""
        return np.clip(X + rng.randn(*X.shape).astype(np.float32)
                       * self.feat_noise_std, 0, None)

    # ── Combined augmentation ─────────────────────────────────────────────────

    def augment(self, X: np.ndarray, ei: np.ndarray,
                rng: np.random.RandomState = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply one feature augmentation + edge dropout.
        Returns (X_aug, edge_index_aug).
        """
        if rng is None:
            rng = np.random.RandomState()

        op = (rng.choice(["mask", "noise"])
              if self.strategy == "random" else self.strategy)

        if   op == "mask":  Xa = self.feature_mask(X, rng)
        elif op == "noise": Xa = self.gaussian_noise(X, rng)
        elif op == "both":  Xa = self.gaussian_noise(
                                    self.feature_mask(X, rng), rng)
        else:               Xa = X.copy()

        return Xa, self.edge_dropout(ei, rng)

    def student_teacher_views(
            self, X: np.ndarray, ei: np.ndarray, step: int = 0
    ) -> Tuple[Tuple, Tuple]:
        """Return independent (student_view, teacher_view) pairs."""
        sv = self.augment(X, ei, np.random.RandomState(step * 2))
        tv = self.augment(X, ei, np.random.RandomState(step * 2 + 1))
        return sv, tv
