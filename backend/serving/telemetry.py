"""
Prometheus ML Inference Telemetry & Real-Time SLA Monitor.
Tracks inference request counts, latency percentiles, error rates, and canary distributions.
"""

from typing import Dict, List, Any
import time


class InferenceTelemetryRegistry:
    """In-memory telemetry collector for ML serving metrics."""

    def __init__(self):
        self.inference_requests_total: Dict[str, int] = {}
        self.inference_latencies: Dict[str, List[float]] = {}
        self.model_errors_total: Dict[str, int] = {}

    def record_inference(self, model_id: str, latency_ms: float, success: bool = True) -> None:
        self.inference_requests_total[model_id] = self.inference_requests_total.get(model_id, 0) + 1
        if model_id not in self.inference_latencies:
            self.inference_latencies[model_id] = []
        self.inference_latencies[model_id].append(latency_ms)

        if not success:
            self.model_errors_total[model_id] = self.model_errors_total.get(model_id, 0) + 1

    def get_summary(self, model_id: str) -> Dict[str, Any]:
        latencies = self.inference_latencies.get(model_id, [])
        count = self.inference_requests_total.get(model_id, 0)
        errors = self.model_errors_total.get(model_id, 0)

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0

        return {
            "model_id": model_id,
            "total_requests": count,
            "total_errors": errors,
            "avg_latency_ms": round(avg_lat, 2),
            "p99_latency_ms": round(p99, 2),
        }

    def export_prometheus_text(self) -> str:
        lines = [
            "# HELP modelforge_inference_requests_total Total ML inference requests scored",
            "# TYPE modelforge_inference_requests_total counter",
        ]
        for m_id, count in self.inference_requests_total.items():
            lines.append(f'modelforge_inference_requests_total{{model_id="{m_id}"}} {count}')

        lines.extend([
            "# HELP modelforge_model_errors_total Total inference prediction failures",
            "# TYPE modelforge_model_errors_total counter",
        ])
        for m_id, errors in self.model_errors_total.items():
            lines.append(f'modelforge_model_errors_total{{model_id="{m_id}"}} {errors}')

        return "\n".join(lines) + "\n"
