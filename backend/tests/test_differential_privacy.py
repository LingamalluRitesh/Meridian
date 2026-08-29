"""Unit Tests for Differential Privacy & Security Mechanisms."""
import numpy as np
import pytest
from backend.security.differential_privacy import DPMechanism, DifferentialPrivacyAccountant
from backend.security.adversarial_defense import AdversarialRobustnessTester
from backend.security.model_watermarking import ModelWatermarkVerifier

def test_laplace_and_gaussian_dp():
    data = np.zeros((10, 5))
    noisy_lap = DPMechanism.laplace_mechanism(data, epsilon=1.0)
    noisy_gauss = DPMechanism.gaussian_mechanism(data, epsilon=1.0, delta=1e-5)
    assert noisy_lap.shape == data.shape
    assert noisy_gauss.shape == data.shape
    assert not np.all(noisy_lap == 0)

def test_gradient_clipping():
    grads = np.array([[3.0, 4.0]]) # Norm = 5.0
    clipped = DPMechanism.clip_gradients(grads, max_norm=1.0)
    norm_after = np.linalg.norm(clipped)
    assert np.isclose(norm_after, 1.0)

def test_model_watermarking():
    triggers = np.random.randn(5, 4)
    expected = np.array([1, 0, 1, 1, 0])
    verifier = ModelWatermarkVerifier(triggers, expected)
    
    # Model that predicts correctly
    def mock_model(x):
        probs = np.zeros((len(x), 2))
        probs[:, 1] = 1.0 # Predicts class 1
        return probs
        
    res = verifier.verify_ownership(mock_model, threshold=0.5)
    assert "match_rate" in res
