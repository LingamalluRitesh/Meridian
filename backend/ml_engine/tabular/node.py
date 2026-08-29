"""
Neural Oblivious Decision Ensembles (NODE)
Differentiable ensemble of oblivious decision trees with entmax transformation.
"""

import numpy as np

class NeuralObliviousDecisionEnsemble:
    def __init__(self, input_dim: int, n_trees: int = 64, depth: int = 6, output_dim: int = 2):
        self.input_dim = input_dim
        self.n_trees = n_trees
        self.depth = depth
        self.output_dim = output_dim
        
        # Split feature selection weights (n_trees, depth, input_dim)
        self.feature_selectors = np.random.randn(n_trees, depth, input_dim) * 0.1
        self.split_thresholds = np.zeros((n_trees, depth))
        # Response table leaf values (n_trees, 2^depth, output_dim)
        self.leaf_responses = np.random.randn(n_trees, 1 << depth, output_dim) * 0.05

    def _entmax15(self, z: np.ndarray) -> np.ndarray:
        # Approximate 1.5-entmax using soft thresholding
        z_pos = np.maximum(z, 0)
        sum_z = np.sum(z_pos, axis=-1, keepdims=True) + 1e-8
        return z_pos / sum_z

    def forward(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        # Aggregate responses across all oblivious trees
        total_output = np.zeros((B, self.output_dim))
        for tree_idx in range(self.n_trees):
            # Compute leaf index routing probabilities
            leaf_idx_probs = np.ones((B, 1))
            for d in range(self.depth):
                selector = self._entmax15(self.feature_selectors[tree_idx, d])
                chosen_feature = np.dot(x, selector) # (B,)
                split_prob = 1.0 / (1.0 + np.exp(-(chosen_feature - self.split_thresholds[tree_idx, d])))
                split_prob = split_prob[:, None]
                leaf_idx_probs = np.hstack([leaf_idx_probs * (1 - split_prob), leaf_idx_probs * split_prob])
                
            leaf_vals = self.leaf_responses[tree_idx] # (2^depth, output_dim)
            tree_pred = np.dot(leaf_idx_probs[:, :1 << self.depth], leaf_vals)
            total_output += tree_pred
            
        return total_output / self.n_trees
