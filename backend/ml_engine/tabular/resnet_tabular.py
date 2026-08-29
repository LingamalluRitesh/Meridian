"""
ResNet Tabular Architecture
Deep Residual Networks customized for tabular representation with dense skip-connections and Swish/Mish activations.
"""

import numpy as np
from typing import List, Optional

class ResNetBlock:
    def __init__(self, dim: int, dropout: float = 0.1):
        self.dim = dim
        self.W1 = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.b1 = np.zeros(dim)
        self.W2 = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.b2 = np.zeros(dim)
        self.gamma1 = np.ones(dim)
        self.gamma2 = np.ones(dim)

    def _swish(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # LayerNorm 1 + Swish + Linear 1
        mean1 = np.mean(x, axis=-1, keepdims=True)
        var1 = np.var(x, axis=-1, keepdims=True)
        norm1 = self.gamma1 * (x - mean1) / np.sqrt(var1 + 1e-5)
        h1 = self._swish(np.dot(norm1, self.W1) + self.b1)
        
        # LayerNorm 2 + Swish + Linear 2
        mean2 = np.mean(h1, axis=-1, keepdims=True)
        var2 = np.var(h1, axis=-1, keepdims=True)
        norm2 = self.gamma2 * (h1 - mean2) / np.sqrt(var2 + 1e-5)
        h2 = np.dot(norm2, self.W2) + self.b2
        return x + h2

class ResNetTabular:
    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 256, n_blocks: int = 4):
        self.input_proj = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.proj_b = np.zeros(hidden_dim)
        self.blocks = [ResNetBlock(hidden_dim) for _ in range(n_blocks)]
        self.head = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.head_b = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.input_proj) + self.proj_b
        for block in self.blocks:
            h = block.forward(h)
        logits = np.dot(h, self.head) + self.head_b
        return logits
