"""
PatchTST: Subseries Patching Transformer for Long-term Time Series Forecasting.
"""

import numpy as np

class PatchTSTModel:
    def __init__(self, patch_size: int = 16, horizon: int = 24, d_model: int = 64):
        self.W_patch = np.random.randn(patch_size, d_model) * 0.05
        self.W_out = np.random.randn(d_model, horizon) * 0.05

    def forward(self, patches: np.ndarray) -> np.ndarray:
        # patches is (B, Num_patches, Patch_size)
        h = np.mean(np.dot(patches, self.W_patch), axis=1)
        return np.dot(np.maximum(h, 0), self.W_out)
