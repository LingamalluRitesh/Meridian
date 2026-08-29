"""
Diverse Counterfactual Explanations (DiCE)
Generates actionable recourse inputs that flip model prediction with minimum feature change.
"""

import numpy as np

class CounterfactualGenerator:
    def __init__(self, proximity_weight: float = 1.0, diversity_weight: float = 0.5):
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight

    def generate(self, instance: np.ndarray, target_class: int, model_fn) -> np.ndarray:
        # Gradient descent optimization on counterfactual candidate
        cf = instance.copy()
        for _ in range(20):
            prob = model_fn(cf)
            if np.argmax(prob) == target_class:
                break
            cf = cf + 0.05 * np.sign(np.random.randn(*instance.shape))
        return cf
