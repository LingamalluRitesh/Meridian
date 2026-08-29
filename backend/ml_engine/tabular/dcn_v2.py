"""
DCN-v2: Improved Deep & Cross Network for Explicit Feature Interactions.
"""

import numpy as np
from typing import List

class CrossLayer:
    def __init__(self, in_features: int):
        self.in_features = in_features
        self.W = np.random.randn(in_features, in_features) * 0.05
        self.bias = np.zeros(in_features)

    def forward(self, x0: np.ndarray, xl: np.ndarray) -> np.ndarray:
        linear = np.dot(xl, self.W) + self.bias
        return x0 * linear + xl

class DeepCrossNetworkV2:
    def __init__(self, in_features: int, num_cross_layers: int = 3, deep_layers: List[int] = [128, 64], out_classes: int = 2):
        self.cross_layers = [CrossLayer(in_features) for _ in range(num_cross_layers)]
        self.deep_weights = []
        self.deep_biases = []
        prev = in_features
        for h in deep_layers:
            self.deep_weights.append(np.random.randn(prev, h) * np.sqrt(2.0 / prev))
            self.deep_biases.append(np.zeros(h))
            prev = h
            
        self.head = np.random.randn(in_features + prev, out_classes) * 0.05
        self.head_bias = np.zeros(out_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        xl = x
        for cl in self.cross_layers:
            xl = cl.forward(x, xl)
        h = x
        for w, b in zip(self.deep_weights, self.deep_biases):
            h = np.maximum(np.dot(h, w) + b, 0)
        combined = np.hstack([xl, h])
        return np.dot(combined, self.head) + self.head_bias
