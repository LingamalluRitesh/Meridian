"""Unit Tests for Algorithmic Fairness & Bias Auditing."""
import numpy as np
import pytest
from backend.fairness.bias_detector import AlgorithmicFairnessAuditor
from backend.fairness.mitigation_pre_processing import ReWeighingMitigator

def test_disparate_impact_calculation():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    protected_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0]) # 1 = privileged, 0 = unprivileged
    
    metrics = AlgorithmicFairnessAuditor.calculate_fairness_metrics(y_true, y_pred, protected_attr)
    assert "disparate_impact_ratio" in metrics
    assert "statistical_parity_difference" in metrics
    assert "equal_opportunity_difference" in metrics

def test_reweighing_mitigation():
    y = np.array([1, 0, 1, 0])
    protected = np.array([1, 1, 0, 0])
    weights = ReWeighingMitigator.compute_weights(y, protected)
    assert len(weights) == 4
    assert np.all(weights > 0)
