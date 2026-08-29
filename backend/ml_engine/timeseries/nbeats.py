"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
Trend and Seasonality decomposition blocks with doubly residual stacking.
"""

import numpy as np
from typing import Tuple

class NBEATSBlock:
    def __init__(self, backcast_length: int, forecast_length: int, hidden_dim: int = 128, block_type: str = "generic"):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.block_type = block_type
        
        # 4 FC layers
        self.W1 = np.random.randn(backcast_length, hidden_dim) * 0.05
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.05
        self.W3 = np.random.randn(hidden_dim, hidden_dim) * 0.05
        self.W4 = np.random.randn(hidden_dim, hidden_dim) * 0.05
        
        # Basis expansion parameters
        self.theta_b = np.random.randn(hidden_dim, backcast_length) * 0.05
        self.theta_f = np.random.randn(hidden_dim, forecast_length) * 0.05

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(np.dot(x, self.W1), 0)
        h = np.maximum(np.dot(h, self.W2), 0)
        h = np.maximum(np.dot(h, self.W3), 0)
        h = np.maximum(np.dot(h, self.W4), 0)
        
        backcast = np.dot(h, self.theta_b)
        forecast = np.dot(h, self.theta_f)
        return backcast, forecast

class NBEATSModel:
    def __init__(self, backcast_length: int = 24, forecast_length: int = 12, n_stacks: int = 4):
        self.blocks = [NBEATSBlock(backcast_length, forecast_length) for _ in range(n_stacks)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        residual = x
        total_forecast = np.zeros((x.shape[0], self.blocks[0].forecast_length))
        for block in self.blocks:
            backcast, forecast = block.forward(residual)
            residual = residual - backcast
            total_forecast += forecast
        return total_forecast
