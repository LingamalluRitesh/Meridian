import pytest
from backend.governance.drift_detector import DriftDetector
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_psi_and_ks_calculation():
    detector = DriftDetector()
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    identical = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    shifted = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]

    # Identical distribution has near-zero PSI and KS
    psi_ident = detector.calculate_psi(baseline, identical)
    ks_ident = detector.calculate_ks_statistic(baseline, identical)
    assert psi_ident < 0.05
    assert ks_ident == 0.0

    # Shifted distribution detects drift
    eval_res = detector.evaluate_feature_drift("credit_utilization", baseline, shifted)
    assert eval_res["population_stability_index"] > 0.25
    assert eval_res["drift_status"] == "SEVERE_DRIFT"


def test_drift_analysis_api_endpoint():
    payload = {
        "feature_name": "annual_income",
        "baseline": [45000, 52000, 60000, 75000, 82000],
        "current": [46000, 51000, 61000, 74000, 83000],
    }
    res = client.post("/api/v1/governance/drift-analysis", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["feature_name"] == "annual_income"
    assert "population_stability_index" in data
