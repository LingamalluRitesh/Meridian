"""
Graph Isomorphism Network (GIN) for Maximum Expressive Graph Classification.
"""

import numpy as np

class GraphIsomorphismNetwork:
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2, eps: float = 0.0):
        self.eps = eps
        self.W1 = np.random.randn(in_dim, hidden_dim) * 0.05
        self.W2 = np.random.randn(hidden_dim, out_dim) * 0.05

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        neigh_sum = np.dot(adj, x)
        h = (1 + self.eps) * x + neigh_sum
        mlp1 = np.maximum(np.dot(h, self.W1), 0)
        return np.dot(mlp1, self.W2)
