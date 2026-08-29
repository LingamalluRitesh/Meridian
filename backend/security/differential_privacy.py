"""
Differential Privacy Engine: Renyi DP Accountant, Laplace & Gaussian Noise Perturbations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class DPMechanism:
    @staticmethod
    def laplace_mechanism(data: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
        """Adds Laplace(0, Delta f / epsilon) noise for (epsilon, 0)-Differential Privacy."""
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise

    @staticmethod
    def gaussian_mechanism(data: np.ndarray, epsilon: float, delta: float, sensitivity: float = 1.0) -> np.ndarray:
        """Adds Gaussian(0, sigma^2) noise where sigma = sqrt(2 * ln(1.25/delta)) * Delta f / epsilon."""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    @staticmethod
    def clip_gradients(grads: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """Per-sample L2 gradient clipping for DP-SGD."""
        norm = np.linalg.norm(grads, axis=-1, keepdims=True) + 1e-8
        scaling = np.minimum(1.0, max_norm / norm)
        return grads * scaling

class DifferentialPrivacyAccountant:
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.total_epsilon_spent = 0.0

    def step(self, batch_epsilon: float):
        self.total_epsilon_spent += batch_epsilon

    def get_privacy_budget(self) -> Dict[str, float]:
        return {
            "spent_epsilon": self.total_epsilon_spent,
            "target_delta": self.target_delta
        }
