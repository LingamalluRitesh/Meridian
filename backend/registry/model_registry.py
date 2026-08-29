"""
Model Registry with Semantic Versioning and Automated Staging Transitions.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import time

class ModelArtifact(BaseModel):
    model_id: str
    name: str
    version: str
    stage: str = "STAGED" # STAGED, CANARY, PRODUCTION, ARCHIVED
    metrics: Dict[str, float]
    created_at: float = Field(default_factory=time.time)

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelArtifact] = {}

    def register(self, name: str, version: str, metrics: Dict[str, float]) -> ModelArtifact:
        mid = f"{name}:{version}"
        artifact = ModelArtifact(model_id=mid, name=name, version=version, metrics=metrics)
        self.models[mid] = artifact
        return artifact

    def promote_to_production(self, model_id: str) -> Optional[ModelArtifact]:
        if model_id in self.models:
            self.models[model_id].stage = "PRODUCTION"
            return self.models[model_id]
        return None
