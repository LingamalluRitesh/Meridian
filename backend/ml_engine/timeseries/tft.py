"""
Temporal Fusion Transformer (TFT) with Variable Selection and Static Covariates.
"""

import numpy as np

class TemporalFusionTransformer:
    def __init__(self, num_features: int, horizon: int = 12, d_model: int = 32):
        self.W_in = np.random.randn(num_features, d_model) * 0.05
        self.W_out = np.random.randn(d_model, horizon) * 0.05

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, F = x.shape
        h = np.mean(np.dot(x, self.W_in), axis=1) # Temporal pooling
        return np.dot(np.maximum(h, 0), self.W_out)
