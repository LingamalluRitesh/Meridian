"""Unit Tests for Governance, Model Cards & Lineage Tracking."""
import pytest
from backend.governance.model_cards import ModelCardGenerator
from backend.governance.lineage_tracker import LineageTracker
from backend.governance.compliance_checker import ComplianceRuleEngine

def test_model_card_generation():
    card = ModelCardGenerator.generate_card(
        model_name="TabNet Risk",
        version="1.0.0",
        architecture="TabNet",
        metrics={"auc": 0.945},
        dataset_hash="sha256:abc12345"
    )
    assert card.model_name == "TabNet Risk"
    assert card.risk_category == "High-Risk (Annex III)"

def test_lineage_provenance_chain():
    tracker = LineageTracker()
    ds = tracker.record_artifact("ds_01", "Dataset", {"rows": 1000})
    mdl = tracker.record_artifact("mdl_01", "Model", {"auc": 0.95}, parent_ids=["ds_01"])
    chain = tracker.get_provenance_chain("mdl_01")
    assert len(chain) == 2
    assert chain[0]["id"] == "mdl_01"
    assert chain[1]["id"] == "ds_01"

def test_compliance_rule_engine():
    audit = ComplianceRuleEngine.audit_model({
        "evaluation_metrics": {"accuracy": 0.92},
        "out_of_scope_use": "Prohibited use cases defined"
    })
    assert audit["compliant"] is True
    assert audit["score"] == 100.0
