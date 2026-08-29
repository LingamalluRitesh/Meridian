"""Unit Tests for ModelForge AI Python SDK Client."""
import pytest
from sdk.modelforge.client import ModelForgeClient
from sdk.modelforge.experiments import ExperimentTracker
from sdk.modelforge.pipelines import PipelineDAGBuilder

def test_sdk_client():
    client = ModelForgeClient()
    health = client.health_check()
    assert health["status"] == "HEALTHY"
    reg = client.register_model("TabNet Risk", "TabNet", {"auc": 0.94})
    assert reg["status"] == "STAGED"

def test_sdk_experiment_tracker():
    tracker = ExperimentTracker("credit_risk_tuning")
    run_id = tracker.log_run({"lr": 0.01, "depth": 6}, {"auc": 0.952})
    assert run_id.startswith("run_")
    assert len(tracker.runs) == 1

def test_sdk_pipeline_dag_builder():
    builder = PipelineDAGBuilder("churn_prediction_pipeline")
    builder.add_step("ingest", "DataIngestionOperator")
    builder.add_step("train", "AutoMLTrainingOperator", dependencies=["ingest"])
    assert len(builder.nodes) == 2
