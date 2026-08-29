"""
Pipeline DAG Builder for Distributed AutoML Workflows.
"""

from typing import List, Dict, Any

class PipelineDAGBuilder:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.nodes: List[Dict[str, Any]] = []

    def add_step(self, step_name: str, operator_type: str, dependencies: List[str] = None):
        self.nodes.append({
            "name": step_name,
            "operator": operator_type,
            "dependencies": dependencies or []
        })
        return self
