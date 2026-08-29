"""
ExcelFormer: Advanced Tabular Architecture with Neighboring Semi-Permutable Attentions.
"""

import numpy as np

class ExcelFormer:
    def __init__(self, num_features: int, d_model: int = 64, output_dim: int = 2):
        self.W_in = np.random.randn(num_features, d_model) * 0.02
        self.W_out = np.random.randn(d_model, output_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W_in)
        h_act = np.tanh(h)
        return np.dot(h_act, self.W_out)
