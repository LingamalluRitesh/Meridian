"""
End-to-End Multi-Model Inference & Tabular Pipeline Benchmark Suite.
Validates Tabular Models, Fairness Metrics, DP Budgets, Canary Routing, and MAB Allocators.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.ml_engine.tabular.tabnet import TabNetClassifier
from backend.ml_engine.tabular.ft_transformer import FTTransformer
from backend.fairness.bias_detector import AlgorithmicFairnessAuditor
from backend.security.differential_privacy import DifferentialPrivacyAccountant, DPMechanism
from backend.serving.canary_router import CanaryTrafficRouter
from backend.serving.bandit_router import MultiArmedBanditRouter

client = TestClient(app)


def test_tabular_model_inference_pipeline():
    X = np.random.randn(20, 6)
    clf = TabNetClassifier(input_dim=6, n_classes=2)
    clf.fit(X, np.random.randint(0, 2, 20), epochs=1)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})

    X_num = np.random.randn(8, 5)
    X_cat = np.random.randint(0, 3, size=(8, 2))
    model = FTTransformer(num_features=5, cat_cardinalities=[3, 3], output_dim=2)
    logits = model.forward(X_num, X_cat)
    assert logits.shape == (8, 2)


def test_fairness_and_differential_privacy():
    # 1. Fairness
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    protected_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    metrics = AlgorithmicFairnessAuditor.calculate_fairness_metrics(y_true, y_pred, protected_attr)
    assert "disparate_impact_ratio" in metrics
    assert "statistical_parity_difference" in metrics

    # 2. DP Budget
    dp = DifferentialPrivacyAccountant(target_delta=1e-5)
    dp.step(batch_epsilon=0.25)
    budget = dp.get_privacy_budget()
    assert budget["spent_epsilon"] == 0.25
    assert budget["target_delta"] == 1e-5


def test_canary_and_bandit_routing_e2e():
    # Canary
    router = CanaryTrafficRouter("base_m1", "canary_m2", canary_weight=0.5)
    model_id, is_canary = router.route_request()
    assert model_id in ("base_m1", "canary_m2")

    # Bandit
    mab = MultiArmedBanditRouter()
    mab.register_arm("arm_1")
    mab.register_arm("arm_2")
    mab.update_reward("arm_1", 1.0)
    mab.update_reward("arm_2", 0.0)
    selected = mab.select_arm_epsilon_greedy()
    assert selected in ["arm_1", "arm_2"]
