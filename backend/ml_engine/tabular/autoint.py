"""
AutoInt: Automatic Feature Interaction Learning via Multi-Head Self-Attentive Neural Network.
"""

import numpy as np

class AutoIntNetwork:
    def __init__(self, num_features: int, emb_dim: int = 16, n_heads: int = 3, n_layers: int = 2, output_dim: int = 2):
        self.num_features = num_features
        self.emb_dim = emb_dim
        self.W_emb = np.random.randn(num_features, emb_dim) * 0.02
        self.W_q = np.random.randn(emb_dim, emb_dim) * 0.02
        self.W_k = np.random.randn(emb_dim, emb_dim) * 0.02
        self.W_v = np.random.randn(emb_dim, emb_dim) * 0.02
        self.head = np.random.randn(num_features * emb_dim, output_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, F = x.shape
        E = x[:, :, None] * self.W_emb[None, :, :] # (B, F, D)
        Q = np.dot(E, self.W_q)
        K = np.dot(E, self.W_k)
        V = np.dot(E, self.W_v)
        
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.emb_dim)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        interacted = np.matmul(weights, V)
        
        flat = interacted.reshape(B, -1)
        return np.dot(flat, self.head)
