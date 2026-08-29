"""
W3C PROV-Compliant Cryptographic Lineage Tracker
Records datasets, hyperparameters, code hashes, and artifact DAGs.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional

class ArtifactNode:
    def __init__(self, node_id: str, node_type: str, metadata: Dict[str, Any]):
        self.node_id = node_id
        self.node_type = node_type
        self.metadata = metadata
        self.parents: List[str] = []
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.node_id}:{self.node_type}:{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

class LineageTracker:
    def __init__(self):
        self.nodes: Dict[str, ArtifactNode] = {}

    def record_artifact(self, node_id: str, node_type: str, metadata: Dict[str, Any], parent_ids: Optional[List[str]] = None) -> ArtifactNode:
        node = ArtifactNode(node_id, node_type, metadata)
        if parent_ids:
            node.parents.extend(parent_ids)
        self.nodes[node_id] = node
        return node

    def get_provenance_chain(self, node_id: str) -> List[Dict[str, Any]]:
        chain = []
        curr = self.nodes.get(node_id)
        if curr:
            chain.append({"id": curr.node_id, "type": curr.node_type, "hash": curr.hash})
            for p in curr.parents:
                chain.extend(self.get_provenance_chain(p))
        return chain
