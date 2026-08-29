"""
Pre-Processing Bias Mitigation: Sample Re-Weighing Algorithm.
"""

import numpy as np

class ReWeighingMitigator:
    @staticmethod
    def compute_weights(y: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
        N = len(y)
        weights = np.ones(N)
        for p_val in [0, 1]:
            for y_val in [0, 1]:
                mask = (protected_attr == p_val) & (y == y_val)
                p_p = np.mean(protected_attr == p_val)
                p_y = np.mean(y == y_val)
                p_py = np.mean(mask) + 1e-8
                w = (p_p * p_y) / p_py
                weights[mask] = w
        return weights
