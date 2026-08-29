"""
Graph Convolutional Networks (GCN) with Spectral Renormalization Trick.
H^(l+1) = sigma( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(l) * W^(l) )
"""

import numpy as np
from typing import Tuple

class GCNLayer:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)

    def forward(self, h: np.ndarray, norm_adj: np.ndarray) -> np.ndarray:
        # Spatial spectral aggregation
        support = np.dot(h, self.W)
        output = np.dot(norm_adj, support) + self.bias
        return np.maximum(output, 0) # ReLU

class GraphConvolutionalNetwork:
    def __init__(self, in_features: int, hidden_dim: int = 64, out_classes: int = 7, n_layers: int = 2):
        self.layers = []
        for i in range(n_layers):
            fin = in_features if i == 0 else hidden_dim
            fout = out_classes if i == n_layers - 1 else hidden_dim
            self.layers.append(GCNLayer(fin, fout))

    @staticmethod
    def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """Compute symmetric normalized adjacency matrix: D_hat^(-1/2) * (A + I) * D_hat^(-1/2)"""
        A_tilde = adj + np.eye(adj.shape[0])
        degree = np.sum(A_tilde, axis=-1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_hat = np.diag(d_inv_sqrt)
        return np.dot(np.dot(D_hat, A_tilde), D_hat)

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        norm_adj = self.normalize_adjacency(adj)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, norm_adj)
        return h
