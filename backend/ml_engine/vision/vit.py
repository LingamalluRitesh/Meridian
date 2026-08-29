"""Vision Transformer (ViT) Architecture."""
import numpy as np

class VisionTransformer:
    def __init__(self, num_classes: int = 1000, embed_dim: int = 768):
        self.embed_dim = embed_dim
        self.head = np.random.randn(embed_dim, num_classes) * 0.02

    def forward(self, img_batch: np.ndarray) -> np.ndarray:
        B = img_batch.shape[0]
        h = np.random.randn(B, self.embed_dim) * 0.1
        return np.dot(np.tanh(h), self.head)
