import pytest
from backend.serving.telemetry import InferenceTelemetryRegistry
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_telemetry_registry_metrics():
    registry = InferenceTelemetryRegistry()
    registry.record_inference("model_v1", 12.5, success=True)
    registry.record_inference("model_v1", 15.0, success=True)
    registry.record_inference("model_v1", 20.0, success=False)

    summary = registry.get_summary("model_v1")
    assert summary["total_requests"] == 3
    assert summary["total_errors"] == 1
    assert 15.0 <= summary["avg_latency_ms"] <= 16.0

    prom = registry.export_prometheus_text()
    assert 'modelforge_inference_requests_total{model_id="model_v1"} 3' in prom


def test_metrics_api_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "modelforge_inference_requests_total" in res.text
