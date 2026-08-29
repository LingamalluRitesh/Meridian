"""Low-Rank Adaptation (LoRA) for Fine-Tuning."""
import numpy as np

class LoRALinearAdapter:
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.W0 = np.random.randn(in_features, out_features) * 0.02
        self.A = np.random.randn(in_features, rank) * (1.0 / np.sqrt(in_features))
        self.B = np.zeros((rank, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        base_out = np.dot(x, self.W0)
        lora_out = np.dot(np.dot(x, self.A), self.B) * self.scaling
        return base_out + lora_out
