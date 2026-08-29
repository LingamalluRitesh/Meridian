"""
Feature Registry & Schema Catalog
Manages entities, feature views, data types, and semantic descriptions.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str
    join_key: str
    description: str

class FeatureView(BaseModel):
    name: str
    entities: List[str]
    features: Dict[str, str] # name -> dtype
    ttl_seconds: int = 86400
    description: Optional[str] = None

class FeatureRegistry:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.views: Dict[str, FeatureView] = {}

    def register_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def register_feature_view(self, view: FeatureView):
        self.views[view.name] = view

    def get_view(self, name: str) -> Optional[FeatureView]:
        return self.views.get(name)
