"""
Relational Graph Convolutional Networks (R-GCN) for Multi-Relational Knowledge Graphs.
"""

import numpy as np

class RelationalGCN:
    def __init__(self, in_dim: int, n_relations: int = 4, out_dim: int = 16):
        self.W_rel = np.random.randn(n_relations, in_dim, out_dim) * 0.05

    def forward(self, x: np.ndarray, rel_adjs: np.ndarray) -> np.ndarray:
        # rel_adjs is (R, N, N)
        out = np.zeros((x.shape[0], self.W_rel.shape[-1]))
        for r in range(self.W_rel.shape[0]):
            out += np.dot(rel_adjs[r], np.dot(x, self.W_rel[r]))
        return np.maximum(out, 0)
