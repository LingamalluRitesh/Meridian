"""
TabCaps: Capsule Routing Architecture for Tabular Datasets.
"""

import numpy as np

class TabCapsuleNetwork:
    def __init__(self, input_dim: int, n_capsules: int = 8, cap_dim: int = 16, output_dim: int = 2):
        self.W_caps = np.random.randn(input_dim, n_capsules * cap_dim) * 0.05
        self.W_out = np.random.randn(n_capsules * cap_dim, output_dim) * 0.05

    def forward(self, x: np.ndarray) -> np.ndarray:
        caps = np.dot(x, self.W_caps)
        # Squash non-linear activation for capsules
        norm = np.linalg.norm(caps, axis=-1, keepdims=True) + 1e-8
        squashed = (norm / (1.0 + norm**2)) * caps
        return np.dot(squashed, self.W_out)
