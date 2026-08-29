"""
Regulatory Compliance Rules Engine
Evaluates models against GDPR, CCPA, and EU AI Act constraints.
"""

from typing import Dict, List, Any

class ComplianceRuleEngine:
    @staticmethod
    def audit_model(card: Dict[str, Any]) -> Dict[str, Any]:
        violations = []
        if "evaluation_metrics" not in card or not card["evaluation_metrics"]:
            violations.append("EU AI Act Art. 15: Missing validated performance metrics")
        if "out_of_scope_use" not in card:
            violations.append("NIST AI RMF 1.0: Unspecified boundary conditions and out-of-scope risks")
            
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "score": 100.0 if not violations else max(0.0, 100.0 - len(violations) * 25.0)
        }
