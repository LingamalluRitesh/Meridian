"""ConvNeXt Architecture."""
import numpy as np

class ConvNeXtBackbone:
    def __init__(self, num_classes: int = 1000, dim: int = 96):
        self.dim = dim
        self.head = np.random.randn(dim, num_classes) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        h = np.random.randn(B, self.dim) * 0.1
        return np.dot(np.maximum(h, 0), self.head)
