import pytest
from backend.security.adversarial_defense import AdversarialDefenseGuard
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_adversarial_defense_sanitization():
    guard = AdversarialDefenseGuard()
    raw = {
        "credit_score": 9999.0,  # Extreme outlier
        "debt_to_income": -5.0,  # Negative invalid
        "annual_income": 85000.0,
    }
    result = guard.sanitize_and_validate(raw, clip_outliers=True)
    assert result["sanitized_features"]["credit_score"] == 850.0
    assert result["sanitized_features"]["debt_to_income"] == 0.0
    assert len(result["anomaly_flags"]) == 2


def test_adversarial_defense_api():
    payload = {
        "features": {"credit_score": 950.0, "annual_income": 50000.0},
        "clip_outliers": True,
    }
    res = client.post("/api/v1/security/sanitize-inputs", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "sanitized_features" in data
    assert data["sanitized_features"]["credit_score"] == 850.0
