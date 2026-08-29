"""
Graph Attention Networks (GAT)
Veličković et al., ICLR 2018.
"""

import numpy as np

class GATLayer:
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, alpha_leaky: float = 0.2):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha_leaky = alpha_leaky
        
        # W shape: (n_heads, in_features, out_features)
        self.W = np.random.randn(n_heads, in_features, out_features) * np.sqrt(2.0 / in_features)
        self.a_src = np.random.randn(n_heads, out_features, 1) * 0.05
        self.a_dst = np.random.randn(n_heads, out_features, 1) * 0.05
        self.out_proj = np.random.randn(n_heads * out_features, out_features) * 0.05

    def forward(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        N = h.shape[0]
        # Wh shape: (n_heads, N, out_features)
        Wh = np.matmul(h[None, :, :], self.W)
        
        f_src = np.matmul(Wh, self.a_src) # (n_heads, N, 1)
        f_dst = np.matmul(Wh, self.a_dst) # (n_heads, N, 1)
        
        e = f_src + f_dst.swapaxes(1, 2) # (n_heads, N, N)
        e = np.where(e >= 0, e, self.alpha_leaky * e)
        
        mask = (adj == 0)[None, :, :]
        e[np.repeat(mask, self.n_heads, axis=0)] = -1e9
        
        exp_e = np.exp(e - np.max(e, axis=-1, keepdims=True))
        alphas = exp_e / (np.sum(exp_e, axis=-1, keepdims=True) + 1e-9)
        
        context = np.matmul(alphas, Wh) # (n_heads, N, out_features)
        concat = context.swapaxes(0, 1).reshape(N, -1) # (N, n_heads * out_features)
        return np.dot(concat, self.out_proj)

class GraphAttentionNetwork:
    def __init__(self, in_features: int, hidden_dim: int = 16, out_classes: int = 2):
        self.gat1 = GATLayer(in_features, hidden_dim, n_heads=4)
        self.gat2 = GATLayer(hidden_dim, out_classes, n_heads=1)

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        h1 = np.maximum(self.gat1.forward(x, adj), 0)
        out = self.gat2.forward(h1, adj)
        return out
