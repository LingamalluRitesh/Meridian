"""
GrowNet: Gradient Boosting Neural Network with Stage-wise Residual Learning.
"""

import numpy as np
from typing import List

class WeakMLP:
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2):
        self.W1 = np.random.randn(in_dim, hidden_dim) * 0.05
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, out_dim) * 0.05
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(np.dot(x, self.W1) + self.b1, 0)
        return np.dot(h, self.W2) + self.b2

class GradientBoostingNeuralNetwork:
    def __init__(self, input_dim: int, n_estimators: int = 10, lr: float = 0.1, output_dim: int = 2):
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.lr = lr
        self.output_dim = output_dim
        self.learners = [WeakMLP(input_dim, 64, output_dim) for _ in range(n_estimators)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        pred = np.zeros((x.shape[0], self.output_dim))
        for learner in self.learners:
            pred += self.lr * learner.forward(x)
        return pred
