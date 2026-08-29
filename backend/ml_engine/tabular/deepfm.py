"""
DeepFM: A Factorization-Machine based Neural Network for CTR and Tabular Prediction.
Integrates linear component, Factorization Machine (FM) 2nd-order feature interactions, and deep feed-forward neural layers.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

class FactorizationMachineInteraction:
    def __init__(self, num_features: int, embedding_dim: int = 16):
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.V = np.random.randn(num_features, embedding_dim) * 0.05

    def forward(self, x: np.ndarray) -> np.ndarray:
        vx = x[:, :, None] * self.V[None, :, :]
        sum_vx = np.sum(vx, axis=1)
        sum_vx_sq = np.square(sum_vx)
        vx_sq = np.square(vx)
        sq_vx_sum = np.sum(vx_sq, axis=1)
        fm_out = 0.5 * np.sum(sum_vx_sq - sq_vx_sum, axis=1, keepdims=True)
        return fm_out

class DeepFM:
    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        output_dim: int = 2
    ):
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.w_linear = np.random.randn(num_features, 1) * 0.05
        self.b_linear = np.zeros(1)
        self.fm = FactorizationMachineInteraction(num_features, embedding_dim)
        
        deep_in_dim = num_features * embedding_dim
        self.deep_weights = []
        self.deep_biases = []
        prev_dim = deep_in_dim
        for hdim in hidden_dims:
            self.deep_weights.append(np.random.randn(prev_dim, hdim) * np.sqrt(2.0 / prev_dim))
            self.deep_biases.append(np.zeros(hdim))
            prev_dim = hdim
            
        self.head = np.random.randn(prev_dim + 2, output_dim) * 0.05
        self.head_bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        linear_out = np.dot(x, self.w_linear) + self.b_linear
        fm_out = self.fm.forward(x)
        vx = (x[:, :, None] * self.fm.V[None, :, :]).reshape(B, -1)
        h = vx
        for w, b in zip(self.deep_weights, self.deep_biases):
            h = np.maximum(np.dot(h, w) + b, 0)
            
        combined = np.hstack([linear_out, fm_out, h])
        return np.dot(combined, self.head) + self.head_bias
