"""
Heterogeneous Graph Neural Network with Type-Specific Projections.
"""

import numpy as np
from typing import Dict

class HeterogeneousGNN:
    def __init__(self, node_dims: Dict[str, int], out_dim: int = 32):
        self.node_projections = {
            ntype: np.random.randn(dim, out_dim) * 0.05 for ntype, dim in node_dims.items()
        }

    def forward(self, node_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            ntype: np.maximum(np.dot(feat, self.node_projections[ntype]), 0)
            for ntype, feat in node_features.items()
        }
