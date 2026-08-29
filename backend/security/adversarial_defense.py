"""
Input Anomaly & Adversarial Perturbation Defense Guard.
Performs Out-of-Distribution (OOD) detection, gradient/noise clipping, and adversarial robustness testing.
"""

from typing import Dict, List, Any, Tuple, Optional
import math


class AdversarialDefenseGuard:
    """Detects malicious input perturbations and numerical out-of-distribution feature vectors."""

    def __init__(self, feature_bounds: Dict[str, Tuple[float, float]] = None):
        self.bounds = feature_bounds or {
            "credit_score": (300.0, 850.0),
            "debt_to_income": (0.0, 1.0),
            "annual_income": (0.0, 10000000.0),
            "loan_amount": (100.0, 5000000.0),
        }

    def sanitize_and_validate(
        self,
        features: Dict[str, float],
        clip_outliers: bool = True,
    ) -> Dict[str, Any]:
        sanitized = {}
        flags = []
        is_safe = True

        for k, v in features.items():
            if math.isnan(v) or math.isinf(v):
                flags.append(f"INVALID_NUMERICAL_VALUE_{k.upper()}")
                is_safe = False
                sanitized[k] = 0.0
                continue

            if k in self.bounds:
                low, high = self.bounds[k]
                if v < low or v > high:
                    flags.append(f"OUT_OF_BOUNDS_INPUT_{k.upper()}")
                    if clip_outliers:
                        sanitized[k] = max(low, min(high, v))
                    else:
                        is_safe = False
                        sanitized[k] = v
                else:
                    sanitized[k] = v
            else:
                sanitized[k] = v

        return {
            "is_safe": is_safe,
            "sanitized_features": sanitized,
            "anomaly_flags": flags,
        }


class AdversarialRobustnessTester:
    """Evaluates model vulnerability against FGSM and projected gradient perturbations."""

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def generate_fgsm_perturbation(self, input_vector: List[float], gradients: List[float]) -> List[float]:
        perturbed = []
        for x, g in zip(input_vector, gradients):
            sign = 1.0 if g > 0 else (-1.0 if g < 0 else 0.0)
            perturbed.append(x + self.epsilon * sign)
        return perturbed

    def compute_empirical_robustness(self, clean_preds: List[float], adv_preds: List[float]) -> float:
        if not clean_preds or not adv_preds:
            return 1.0
        diffs = [abs(c - a) for c, a in zip(clean_preds, adv_preds)]
        return round(1.0 - (sum(diffs) / len(diffs)), 4)
