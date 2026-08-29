"""
DeepAR: Probabilistic Autoregressive Recurrent Networks for Forecasting.
"""

import numpy as np
from typing import Tuple

class DeepARModel:
    def __init__(self, hidden_dim: int = 64, horizon: int = 12):
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.W_rec = np.random.randn(hidden_dim + 1, hidden_dim) * 0.05
        self.W_mu = np.random.randn(hidden_dim, 1) * 0.05
        self.W_sigma = np.random.randn(hidden_dim, 1) * 0.05

    def forward(self, history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B, T = history.shape
        h = np.zeros((B, self.hidden_dim))
        for t in range(T):
            step_in = np.hstack([h, history[:, t:t+1]])
            h = np.tanh(np.dot(step_in, self.W_rec))
            
        mu_list = []
        sigma_list = []
        current_input = history[:, -1:]
        for _ in range(self.horizon):
            step_in = np.hstack([h, current_input])
            h = np.tanh(np.dot(step_in, self.W_rec))
            mu = np.dot(h, self.W_mu)
            sigma = np.log1p(np.exp(np.dot(h, self.W_sigma))) + 1e-4
            mu_list.append(mu)
            sigma_list.append(sigma)
            current_input = mu
            
        return np.hstack(mu_list), np.hstack(sigma_list)
