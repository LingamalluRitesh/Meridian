"""
High-Throughput Categorical Encoders: Weight of Evidence (WoE) and Smooth Target Encoders.
"""

import numpy as np
from typing import Dict

class CategoricalTargetEncoder:
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encoding_map: Dict[Any, float] = {}
        self.global_mean = 0.0

    def fit(self, categories: np.ndarray, targets: np.ndarray):
        self.global_mean = float(np.mean(targets))
        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = (categories == cat)
            n_cat = np.sum(mask)
            mean_cat = np.mean(targets[mask])
            # Smoothed Bayesian target encoding formula: (n * mean_cat + smoothing * global_mean) / (n + smoothing)
            smooth_val = (n_cat * mean_cat + self.smoothing * self.global_mean) / (n_cat + self.smoothing)
            self.encoding_map[cat] = float(smooth_val)

    def transform(self, categories: np.ndarray) -> np.ndarray:
        return np.array([self.encoding_map.get(cat, self.global_mean) for cat in categories])

class WeightOfEvidenceEncoder:
    def __init__(self):
        self.woe_map: Dict[Any, float] = {}

    def fit(self, categories: np.ndarray, binary_targets: np.ndarray):
        total_good = np.sum(binary_targets == 1) + 1e-5
        total_bad = np.sum(binary_targets == 0) + 1e-5
        for cat in np.unique(categories):
            mask = (categories == cat)
            good = np.sum(binary_targets[mask] == 1) + 1e-5
            bad = np.sum(binary_targets[mask] == 0) + 1e-5
            self.woe_map[cat] = float(np.log((good / total_good) / (bad / total_bad)))

    def transform(self, categories: np.ndarray) -> np.ndarray:
        return np.array([self.woe_map.get(cat, 0.0) for cat in categories])
