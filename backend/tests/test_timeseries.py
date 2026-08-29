"""Unit Tests for ModelForge AI Time Series Forecasting Subsystem."""
import numpy as np
import pytest
from backend.ml_engine.timeseries.nbeats import NBEATSModel
from backend.ml_engine.timeseries.deepar import DeepARModel

def test_nbeats_forecast():
    history = np.random.randn(4, 24)
    model = NBEATSModel(backcast_length=24, forecast_length=12)
    forecast = model.forward(history)
    assert forecast.shape == (4, 12)

def test_deepar_forecast():
    history = np.random.randn(4, 20)
    model = DeepARModel(hidden_dim=32, horizon=10)
    mu, sigma = model.forward(history)
    assert mu.shape == (4, 10)
    assert sigma.shape == (4, 10)
    assert np.all(sigma > 0)
