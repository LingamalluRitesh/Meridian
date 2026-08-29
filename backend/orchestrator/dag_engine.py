"""
Directed Acyclic Graph (DAG) Execution Engine with Topological Sorting.
"""

from typing import Dict, List, Set

class DAGNode:
    def __init__(self, name: str, dependencies: List[str] = None):
        self.name = name
        self.dependencies = dependencies or []

class DAGExecutionEngine:
    @staticmethod
    def topological_sort(nodes: List[DAGNode]) -> List[str]:
        node_map = {n.name: n for n in nodes}
        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def visit(node_name: str):
            if node_name in temp:
                raise ValueError(f"Cycle detected in DAG at node {node_name}")
            if node_name not in visited:
                temp.add(node_name)
                for dep in node_map[node_name].dependencies:
                    visit(dep)
                temp.remove(node_name)
                visited.add(node_name)
                order.append(node_name)

        for n in nodes:
            if n.name not in visited:
                visit(n.name)
                
        return order
