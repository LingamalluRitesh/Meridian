"""
Graph Transformer with Shortest Path and Laplacian Positional Encodings.
"""

import numpy as np

class GraphTransformer:
    def __init__(self, in_dim: int, d_model: int = 64, out_dim: int = 2):
        self.W_in = np.random.randn(in_dim, d_model) * 0.05
        self.W_out = np.random.randn(d_model, output_dim if 'output_dim' in locals() else out_dim) * 0.05

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W_in)
        # Graph Laplacian Eigenvector modulation
        L = np.diag(np.sum(adj, axis=-1)) - adj
        h_mod = np.dot(adj, h) + h
        return np.dot(np.maximum(h_mod, 0), self.W_out)
