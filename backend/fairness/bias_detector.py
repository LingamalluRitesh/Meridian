"""
Algorithmic Fairness Metrics: Disparate Impact, Equal Opportunity, Statistical Parity.
"""

import numpy as np
from typing import Dict, Any

class AlgorithmicFairnessAuditor:
    @staticmethod
    def calculate_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        favorable_label: int = 1
    ) -> Dict[str, float]:
        """
        Disparate Impact Ratio = P(Y_hat=1 | Unprivileged) / P(Y_hat=1 | Privileged)
        Statistical Parity Diff = P(Y_hat=1 | Unprivileged) - P(Y_hat=1 | Privileged)
        Equal Opportunity Diff = TPR_unprivileged - TPR_privileged
        """
        unprivileged_mask = (protected_attr == 0)
        privileged_mask = (protected_attr == 1)
        
        p_unpriv = np.mean(y_pred[unprivileged_mask] == favorable_label) + 1e-8
        p_priv = np.mean(y_pred[privileged_mask] == favorable_label) + 1e-8
        
        disparate_impact = p_unpriv / p_priv
        statistical_parity_diff = p_unpriv - p_priv
        
        # True Positive Rate (TPR)
        tpr_unpriv = np.mean(y_pred[unprivileged_mask & (y_true == 1)] == favorable_label) if np.sum(unprivileged_mask & (y_true == 1)) > 0 else 0.0
        tpr_priv = np.mean(y_pred[privileged_mask & (y_true == 1)] == favorable_label) if np.sum(privileged_mask & (y_true == 1)) > 0 else 0.0
        equal_opportunity_diff = tpr_unpriv - tpr_priv
        
        return {
            "disparate_impact_ratio": float(disparate_impact),
            "statistical_parity_difference": float(statistical_parity_diff),
            "equal_opportunity_difference": float(equal_opportunity_diff),
            "is_four_fifths_compliant": bool(disparate_impact >= 0.80)
        }
