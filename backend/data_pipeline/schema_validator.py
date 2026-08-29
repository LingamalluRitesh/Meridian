"""
Schema Validator & Great-Expectations style Data Contract Enforcement.
"""

from typing import Dict, List, Any

class DataContract:
    def __init__(self, required_columns: List[str], column_types: Dict[str, type], allowable_null_pct: float = 0.05):
        self.required_columns = required_columns
        self.column_types = column_types
        self.allowable_null_pct = allowable_null_pct

class SchemaValidator:
    def __init__(self, contract: DataContract):
        self.contract = contract

    def validate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {"is_valid": True, "errors": []}
            
        errors = []
        for col in self.contract.required_columns:
            missing_count = sum(1 for row in batch if col not in row or row[col] is None)
            null_pct = missing_count / len(batch)
            if null_pct > self.contract.allowable_null_pct:
                errors.append(f"Column '{col}' null rate {null_pct:.2%} exceeds threshold {self.contract.allowable_null_pct:.2%}")
                
        return {"is_valid": len(errors) == 0, "errors": errors}
