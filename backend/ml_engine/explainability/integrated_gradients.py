"""
Integrated Gradients for Deep Neural Network Axiomatic Attribution.
"""

import numpy as np
from typing import Callable

class IntegratedGradients:
    def __init__(self, steps: int = 50):
        self.steps = steps

    def attribute(self, model_grad_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
        if baseline is None:
            baseline = np.zeros_like(x)
            
        alphas = np.linspace(0.0, 1.0, self.steps)[:, None, None]
        # (Steps, Batch, Features)
        interpolated = baseline[None, :, :] + alphas * (x - baseline)[None, :, :]
        
        grads = []
        for step in range(self.steps):
            g = model_grad_fn(interpolated[step])
            grads.append(g)
            
        avg_grads = np.mean(np.array(grads), axis=0)
        return (x - baseline) * avg_grads
