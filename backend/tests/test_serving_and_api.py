"""Unit Tests for Serving Gateway and API Endpoints."""
import pytest
import numpy as np
from backend.serving.inference_server import AsynchronousInferenceServer
from backend.serving.canary_router import CanaryTrafficRouter
from backend.orchestrator.dag_engine import DAGExecutionEngine, DAGNode

def test_inference_server():
    server = AsynchronousInferenceServer()
    class DummyModel:
        def predict_proba(self, x):
            return np.ones((len(x), 2)) * 0.5
            
    server.register_model("dummy_v1", DummyModel())
    res = server.score_batch("dummy_v1", np.zeros((4, 5)))
    assert res.shape == (4, 2)

def test_canary_router():
    router = CanaryTrafficRouter("base_m1", "canary_m2", canary_weight=0.5)
    model_id, is_canary = router.route_request()
    assert model_id in ("base_m1", "canary_m2")

def test_dag_topological_sort():
    nodes = [
        DAGNode("train", dependencies=["extract", "validate"]),
        DAGNode("validate", dependencies=["extract"]),
        DAGNode("extract", dependencies=[])
    ]
    order = DAGExecutionEngine.topological_sort(nodes)
    assert order == ["extract", "validate", "train"]
