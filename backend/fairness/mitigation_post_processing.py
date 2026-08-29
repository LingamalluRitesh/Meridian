"""
Post-Processing Bias Mitigation: Reject Option Classification Threshold Optimizer.
"""

import numpy as np

class RejectOptionMitigator:
    def __init__(self, critical_band_margin: float = 0.1):
        self.margin = critical_band_margin

    def mitigate(self, probas: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
        # Give unprivileged group favorable decision in uncertain margin around decision boundary 0.5
        preds = (probas >= 0.5).astype(int)
        uncertain = np.abs(probas - 0.5) < self.margin
        # Unprivileged in uncertain zone gets positive label
        preds[uncertain & (protected_attr == 0)] = 1
        # Privileged in uncertain zone gets negative label
        preds[uncertain & (protected_attr == 1)] = 0
        return preds
