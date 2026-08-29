"""
SAINT: Self-Attention and Intersample Attention Transformer
Applies self-attention across features and contrastive intersample attention across row instances.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

class IntersampleAttention:
    """Computes self-attention across data batch samples to model inter-instance correlations."""
    def __init__(self, d_token: int = 64, n_heads: int = 4):
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.W_q = np.random.randn(d_token, d_token) * 0.02
        self.W_k = np.random.randn(d_token, d_token) * 0.02
        self.W_v = np.random.randn(d_token, d_token) * 0.02
        self.W_o = np.random.randn(d_token, d_token) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x is (Batch, Features, Dim) -> Transpose to (Features, Batch, Dim)
        x_t = x.swapaxes(0, 1)
        F, B, D = x_t.shape
        
        Q = np.dot(x_t, self.W_q).reshape(F, B, self.n_heads, self.d_head).swapaxes(1, 2)
        K = np.dot(x_t, self.W_k).reshape(F, B, self.n_heads, self.d_head).swapaxes(1, 2)
        V = np.dot(x_t, self.W_v).reshape(F, B, self.n_heads, self.d_head).swapaxes(1, 2)
        
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_head)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        
        context = np.matmul(attn, V)
        context = context.swapaxes(1, 2).reshape(F, B, D)
        out = np.dot(context, self.W_o)
        return out.swapaxes(0, 1) # Return (Batch, Features, Dim)

class SAINTModel:
    """Full SAINT Architecture with alternating self-attention and intersample attention blocks."""
    def __init__(self, num_features: int, d_token: int = 64, n_stages: int = 3, n_classes: int = 2):
        self.num_features = num_features
        self.d_token = d_token
        self.W_emb = np.random.randn(num_features, d_token) * 0.02
        self.intersample_blocks = [IntersampleAttention(d_token) for _ in range(n_stages)]
        self.classifier = np.random.randn(num_features * d_token, n_classes) * 0.02
        self.bias = np.zeros(n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, F = x.shape
        tokens = x[:, :, None] * self.W_emb[None, :, :]
        for block in self.intersample_blocks:
            tokens = tokens + block.forward(tokens)
            
        flat_repr = tokens.reshape(B, -1)
        logits = np.dot(flat_repr, self.classifier) + self.bias
        return logits
