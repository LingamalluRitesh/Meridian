"""
Feature Tokenizer Transformer (FT-Transformer)
Transforms numerical and categorical tabular features into token embeddings and applies Multi-Head Self-Attention.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

class FeatureTokenizer:
    """Transforms raw numerical and categorical variables into D-dimensional token vectors."""
    def __init__(self, num_features: int, cat_cardinalities: List[int], d_token: int = 64):
        self.num_features = num_features
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # Numerical feature projection: W_num is (num_features, d_token), b_num is (num_features, d_token)
        self.W_num = np.random.randn(num_features, d_token) * 0.02
        self.b_num = np.zeros((num_features, d_token))
        
        # Categorical feature embeddings
        self.cat_embeddings = [
            np.random.randn(card, d_token) * 0.02 for card in cat_cardinalities
        ]
        
        # [CLS] Token
        self.cls_token = np.random.randn(1, 1, d_token) * 0.02

    def forward(self, x_num: Optional[np.ndarray], x_cat: Optional[np.ndarray]) -> np.ndarray:
        tokens = []
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        
        # Add CLS token
        cls_expanded = np.repeat(self.cls_token, batch_size, axis=0)
        tokens.append(cls_expanded)
        
        if x_num is not None and self.num_features > 0:
            # (B, F_num) -> (B, F_num, D)
            num_tokens = x_num[:, :, None] * self.W_num[None, :, :] + self.b_num[None, :, :]
            tokens.append(num_tokens)
            
        if x_cat is not None and len(self.cat_cardinalities) > 0:
            cat_tokens_list = []
            for i, emb_table in enumerate(self.cat_embeddings):
                col_indices = x_cat[:, i].astype(int)
                cat_tokens_list.append(emb_table[col_indices][:, None, :])
            cat_tokens = np.concatenate(cat_tokens_list, axis=1)
            tokens.append(cat_tokens)
            
        return np.concatenate(tokens, axis=1) # Shape: (B, 1 + N_features, D)

class MultiHeadSelfAttention:
    """Multi-head scaled dot-product self-attention with LayerNorm."""
    def __init__(self, d_token: int = 64, n_heads: int = 8, dropout: float = 0.1):
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        
        self.W_q = np.random.randn(d_token, d_token) * 0.02
        self.W_k = np.random.randn(d_token, d_token) * 0.02
        self.W_v = np.random.randn(d_token, d_token) * 0.02
        self.W_out = np.random.randn(d_token, d_token) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, N, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, N, self.n_heads, self.d_head).swapaxes(1, 2)
        K = np.dot(x, self.W_k).reshape(B, N, self.n_heads, self.d_head).swapaxes(1, 2)
        V = np.dot(x, self.W_v).reshape(B, N, self.n_heads, self.d_head).swapaxes(1, 2)
        
        # Attention scores
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_head)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        
        context = np.matmul(attn_weights, V)
        context = context.swapaxes(1, 2).reshape(B, N, D)
        out = np.dot(context, self.W_out)
        return out

class TransformerLayer:
    """Pre-Norm Transformer Encoder Block with Feedforward GeLU network."""
    def __init__(self, d_token: int = 64, n_heads: int = 8, d_ffn_factor: float = 4.0):
        self.mha = MultiHeadSelfAttention(d_token, n_heads)
        d_ffn = int(d_token * d_ffn_factor)
        self.W_ffn1 = np.random.randn(d_token, d_ffn) * 0.02
        self.b_ffn1 = np.zeros(d_ffn)
        self.W_ffn2 = np.random.randn(d_ffn, d_token) * 0.02
        self.b_ffn2 = np.zeros(d_token)
        
        self.gamma1 = np.ones(d_token)
        self.gamma2 = np.ones(d_token)

    def _layernorm(self, x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-Norm Attention
        norm1 = self._layernorm(x, self.gamma1)
        x = x + self.mha.forward(norm1)
        
        # Pre-Norm FFN
        norm2 = self._layernorm(x, self.gamma2)
        ffn = self._gelu(np.dot(norm2, self.W_ffn1) + self.b_ffn1)
        x = x + np.dot(ffn, self.W_ffn2) + self.b_ffn2
        return x

class FTTransformer:
    """Feature Tokenizer Transformer with Deep Stack of Multi-Head Self-Attentions."""
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        output_dim: int = 2,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 8
    ):
        self.tokenizer = FeatureTokenizer(num_features, cat_cardinalities, d_token)
        self.layers = [TransformerLayer(d_token, n_heads) for _ in range(n_layers)]
        self.head = np.random.randn(d_token, output_dim) * 0.02
        self.bias = np.zeros(output_dim)

    def forward(self, x_num: Optional[np.ndarray], x_cat: Optional[np.ndarray]) -> np.ndarray:
        tokens = self.tokenizer.forward(x_num, x_cat)
        for layer in self.layers:
            tokens = layer.forward(tokens)
            
        cls_rep = tokens[:, 0, :] # Extract [CLS] Token representation
        logits = np.dot(cls_rep, self.head) + self.bias
        return logits

    def predict_proba(self, x_num: Optional[np.ndarray], x_cat: Optional[np.ndarray]) -> np.ndarray:
        logits = self.forward(x_num, x_cat)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)
