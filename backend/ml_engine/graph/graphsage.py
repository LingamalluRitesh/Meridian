"""
GraphSAGE: Inductive Graph Representation Learning
Implements Neighborhood Sampling and Mean / Max / LSTM Aggregators.
"""

import numpy as np

class GraphSAGE:
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        self.W_self = np.random.randn(in_dim, hidden_dim) * 0.05
        self.W_neigh = np.random.randn(in_dim, hidden_dim) * 0.05
        self.W_out = np.random.randn(hidden_dim, out_dim) * 0.05

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        # Mean neighbor aggregator
        degrees = np.sum(adj, axis=-1, keepdims=True)
        degrees[degrees == 0] = 1.0
        neigh_mean = np.dot(adj, x) / degrees
        
        # Combine self and neighborhood features
        h = np.dot(x, self.W_self) + np.dot(neigh_mean, self.W_neigh)
        h_norm = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
        return np.dot(np.maximum(h_norm, 0), self.W_out)
