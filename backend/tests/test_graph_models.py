"""Unit Tests for ModelForge AI Graph Neural Network Subsystem."""
import numpy as np
import pytest
from backend.ml_engine.graph.gcn import GraphConvolutionalNetwork
from backend.ml_engine.graph.gat import GraphAttentionNetwork
from backend.ml_engine.graph.graphsage import GraphSAGE

def test_gcn_forward():
    X = np.random.randn(6, 8)
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    gcn = GraphConvolutionalNetwork(in_features=8, hidden_dim=16, out_classes=3)
    out = gcn.forward(X, adj)
    assert out.shape == (6, 3)

def test_gat_forward():
    X = np.random.randn(4, 8)
    adj = np.ones((4, 4))
    gat = GraphAttentionNetwork(in_features=8, hidden_dim=16, out_classes=2)
    out = gat.forward(X, adj)
    assert out.shape == (4, 2)

def test_graphsage_forward():
    X = np.random.randn(5, 6)
    adj = np.eye(5)
    sage = GraphSAGE(in_dim=6, hidden_dim=12, out_dim=4)
    out = sage.forward(X, adj)
    assert out.shape == (5, 4)
