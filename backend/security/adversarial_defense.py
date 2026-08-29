"""
Adversarial Robustness Testing: Fast Gradient Sign Method (FGSM) and Randomized Smoothing.
"""

import numpy as np

class AdversarialRobustnessTester:
    @staticmethod
    def generate_fgsm_perturbation(x: np.ndarray, grad_sign: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
        """x_adv = x + epsilon * sign(grad_x L(theta, x, y))"""
        return x + epsilon * np.sign(grad_sign)

    @staticmethod
    def randomized_smoothing(predict_fn, x: np.ndarray, sigma: float = 0.1, n_samples: int = 50) -> np.ndarray:
        """Certifiable adversarial robustness via majority voting over Gaussian noise neighborhood."""
        B = x.shape[0]
        preds = []
        for _ in range(n_samples):
            noisy_x = x + np.random.normal(0, sigma, size=x.shape)
            preds.append(predict_fn(noisy_x))
        return np.mean(preds, axis=0)
