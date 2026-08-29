"""
Dataset Snapshot & Version Management.
"""

import hashlib
from typing import Dict, Any

class DatasetManager:
    @staticmethod
    def compute_dataset_hash(data_matrix) -> str:
        content = str(len(data_matrix)) + str(data_matrix[0] if len(data_matrix) > 0 else "")
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
