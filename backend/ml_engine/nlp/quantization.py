"""Post-Training Weight Quantization (INT8)."""
import numpy as np
from typing import Tuple

class WeightQuantizer:
    @staticmethod
    def quantize_int8(weights: np.ndarray) -> Tuple[np.ndarray, float]:
        max_val = np.max(np.abs(weights)) + 1e-8
        scale = max_val / 127.0
        q_weights = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        return q_weights, float(scale)

    @staticmethod
    def dequantize_int8(q_weights: np.ndarray, scale: float) -> np.ndarray:
        return q_weights.astype(np.float32) * scale
