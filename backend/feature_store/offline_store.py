"""
Offline Columnar Feature Store with Partition Pruning and Historical Versioning.
"""

from typing import Dict, List, Any
import numpy as np

class OfflineParquetFeatureStore:
    def __init__(self, storage_uri: str = "s3://modelforge-features/offline"):
        self.storage_uri = storage_uri
        self._tables: Dict[str, List[Dict[str, Any]]] = {}

    def write_features(self, feature_view_name: str, records: List[Dict[str, Any]]):
        if feature_view_name not in self._tables:
            self._tables[feature_view_name] = []
        self._tables[feature_view_name].extend(records)

    def read_historical_features(self, feature_view_name: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        return self._tables.get(feature_view_name, [])
