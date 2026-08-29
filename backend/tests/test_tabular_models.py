"""Unit Tests for ModelForge AI Deep Tabular Learning Subsystem."""
import numpy as np
import pytest
from backend.ml_engine.tabular.tabnet import TabNet, TabNetClassifier, TabNetRegressor, Sparsemax
from backend.ml_engine.tabular.ft_transformer import FTTransformer, FeatureTokenizer
from backend.ml_engine.tabular.saint import SAINTModel
from backend.ml_engine.tabular.resnet_tabular import ResNetTabular
from backend.ml_engine.tabular.node import NeuralObliviousDecisionEnsemble
from backend.ml_engine.tabular.grownet import GradientBoostingNeuralNetwork
from backend.ml_engine.tabular.autoint import AutoIntNetwork

def test_sparsemax_forward():
    sm = Sparsemax(axis=-1)
    z = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    p = sm.forward(z)
    assert p.shape == z.shape
    assert np.allclose(np.sum(p, axis=-1), 1.0)
    assert np.all(p >= 0)

def test_tabnet_forward_and_prediction():
    X = np.random.randn(10, 8)
    tabnet = TabNet(input_dim=8, output_dim=2, n_steps=3)
    logits, masks, loss = tabnet.forward(X, training=True)
    assert logits.shape == (10, 2)
    assert len(masks) == 3
    assert loss >= 0.0

def test_tabnet_classifier_fit_predict():
    X = np.random.randn(20, 6)
    clf = TabNetClassifier(input_dim=6, n_classes=2)
    clf.fit(X, np.random.randint(0, 2, 20), epochs=2)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})

def test_ft_transformer():
    X_num = np.random.randn(8, 5)
    X_cat = np.random.randint(0, 3, size=(8, 2))
    model = FTTransformer(num_features=5, cat_cardinalities=[3, 3], output_dim=2)
    logits = model.forward(X_num, X_cat)
    assert logits.shape == (8, 2)
    probas = model.predict_proba(X_num, X_cat)
    assert np.allclose(np.sum(probas, axis=-1), 1.0)

def test_saint_model():
    X = np.random.randn(8, 4)
    model = SAINTModel(num_features=4, d_token=32, n_classes=2)
    logits = model.forward(X)
    assert logits.shape == (8, 2)

def test_resnet_tabular():
    X = np.random.randn(12, 10)
    model = ResNetTabular(input_dim=10, output_dim=3)
    logits = model.forward(X)
    assert logits.shape == (12, 3)

def test_node_ensemble():
    X = np.random.randn(8, 6)
    node = NeuralObliviousDecisionEnsemble(input_dim=6, n_trees=8, depth=4, output_dim=2)
    preds = node.forward(X)
    assert preds.shape == (8, 2)

def test_grownet():
    X = np.random.randn(10, 5)
    model = GradientBoostingNeuralNetwork(input_dim=5, n_estimators=4)
    preds = model.forward(X)
    assert preds.shape == (10, 2)

def test_autoint():
    X = np.random.randn(8, 6)
    model = AutoIntNetwork(num_features=6, emb_dim=16, output_dim=2)
    logits = model.forward(X)
    assert logits.shape == (8, 2)
