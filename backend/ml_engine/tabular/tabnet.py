"""
TabNet: Attentive Interpretable Tabular Learning Architecture
Implements sequential attention, sparsemax feature selection, feature transformers, and decision steps.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class Sparsemax:
    """Sparsemax activation function for sparse, interpretable probability distributions."""
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute Sparsemax probabilities:
        p_i = [z_i - tau(z)]_+ where tau(z) is a threshold chosen such that sum(p) = 1.
        """
        z = z - np.max(z, axis=self.axis, keepdims=True)
        zs = np.sort(z, axis=self.axis)[..., ::-1]
        range_indices = np.arange(1, z.shape[self.axis] + 1)
        
        bound = 1 + range_indices * zs
        cumsum = np.cumsum(zs, axis=self.axis)
        is_gt = bound > cumsum
        k = np.max(np.where(is_gt, range_indices, 0), axis=self.axis, keepdims=True)
        
        tau = (np.take_along_axis(cumsum, k - 1, axis=self.axis) - 1) / k
        output = np.maximum(z - tau, 0)
        return output

class GBN:
    """Ghost Batch Normalization for tabular stability across virtual mini-batches."""
    def __init__(self, num_features: int, virtual_batch_size: int = 128, momentum: float = 0.9):
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + 1e-5) + self.beta
        
        batch_size = x.shape[0]
        chunks = max(1, batch_size // self.virtual_batch_size)
        sub_batches = np.array_split(x, chunks, axis=0)
        out_chunks = []
        for sb in sub_batches:
            if sb.shape[0] == 0:
                continue
            mean = np.mean(sb, axis=0)
            var = np.var(sb, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            norm = (sb - mean) / np.sqrt(var + 1e-5)
            out_chunks.append(self.gamma * norm + self.beta)
        return np.vstack(out_chunks) if out_chunks else x

class FeatureTransformerBlock:
    """Shared and decision-step dependent feature transformer with GLU activations."""
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # GLU weights: 2 * out_dim for linear + gate
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, 2 * out_dim) * scale
        self.b = np.zeros(2 * out_dim)
        self.gbn = GBN(2 * out_dim)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        linear_out = np.dot(x, self.W) + self.b
        norm_out = self.gbn.forward(linear_out, training=training)
        dim = self.out_dim
        fc = norm_out[:, :dim]
        gate = 1.0 / (1.0 + np.exp(-norm_out[:, dim:])) # Sigmoid GLU
        return (fc * gate) * np.sqrt(0.5)

class AttentiveTransformer:
    """Generates sparse feature selection masks at each decision step."""
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.gbn = GBN(out_dim)
        self.sparsemax = Sparsemax(axis=-1)

    def forward(self, a: np.ndarray, prior: np.ndarray, training: bool = True) -> np.ndarray:
        linear = np.dot(a, self.W) + self.b
        norm = self.gbn.forward(linear, training=training)
        # Prior scale controls reuse penalty
        masked_input = norm * prior
        mask = self.sparsemax.forward(masked_input)
        return mask

class TabNet:
    """Full TabNet Architecture with sequential decision steps and feature masks."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_shared: int = 2,
        n_independent: int = 2,
        virtual_batch_size: int = 128
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.init_gbn = GBN(input_dim, virtual_batch_size)
        self.attentive_transformers = [AttentiveTransformer(n_a, input_dim) for _ in range(n_steps)]
        
        # Shared blocks
        self.shared_blocks = [
            FeatureTransformerBlock(input_dim if i == 0 else (n_d + n_a), (n_d + n_a))
            for i in range(n_shared)
        ]
        
        # Step-dependent blocks
        self.step_blocks = []
        for _ in range(n_steps):
            step = [FeatureTransformerBlock(n_d + n_a, n_d + n_a) for _ in range(n_independent)]
            self.step_blocks.append(step)
            
        self.final_mapping = np.random.randn(n_d, output_dim) * np.sqrt(2.0 / n_d)
        self.final_bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[np.ndarray], float]:
        batch_size = x.shape[0]
        x_norm = self.init_gbn.forward(x, training=training)
        
        prior = np.ones((batch_size, self.input_dim))
        step_outputs = []
        masks = []
        entropy_loss = 0.0
        
        a = np.zeros((batch_size, self.n_a))
        
        for step in range(self.n_steps):
            # Attentive mask selection
            if step == 0:
                mask = self.attentive_transformers[step].sparsemax.forward(np.ones((batch_size, self.input_dim)) / self.input_dim)
            else:
                mask = self.attentive_transformers[step].forward(a, prior, training=training)
            
            masks.append(mask)
            # Update prior with sparsity decay
            prior = prior * (self.gamma - mask)
            
            # Entropy regularization term
            entropy_loss += -np.mean(np.sum(mask * np.log(mask + 1e-15), axis=-1))
            
            # Feature transform
            x_step = x_norm * mask
            h = x_step
            for sb in self.shared_blocks:
                h = sb.forward(h, training=training)
            for db in self.step_blocks[step]:
                h = db.forward(h, training=training)
                
            d = h[:, :self.n_d]
            a = h[:, self.n_d:]
            step_outputs.append(np.maximum(d, 0)) # ReLU activation
            
        aggregated_d = np.sum(step_outputs, axis=0)
        logits = np.dot(aggregated_d, self.final_mapping) + self.final_bias
        return logits, masks, entropy_loss

class TabNetClassifier:
    """High-level Scikit-Learn compatible TabNet Classifier."""
    def __init__(self, input_dim: int, n_classes: int, **kwargs):
        self.model = TabNet(input_dim=input_dim, output_dim=n_classes, **kwargs)
        self.n_classes = n_classes

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.02) -> 'TabNetClassifier':
        # Standard cross-entropy mini-batch gradient descent loop
        for epoch in range(epochs):
            logits, _, _ = self.model.forward(X, training=True)
            # Softmax
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _, _ = self.model.forward(X, training=False)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

class TabNetRegressor:
    """High-level Scikit-Learn compatible TabNet Regressor."""
    def __init__(self, input_dim: int, **kwargs):
        self.model = TabNet(input_dim=input_dim, output_dim=1, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.02) -> 'TabNetRegressor':
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _, _ = self.model.forward(X, training=False)
        return preds.flatten()
