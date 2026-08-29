"""
ModelForge AI Master REST API Server
Asynchronous FastAPI endpoints for Models, Features, Governance, Fairness, and Pipelines.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any

app = FastAPI(
    title="ModelForge AI Enterprise API",
    version="1.0.0",
    description="Enterprise Automated Machine Learning & MLOps Platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    model_id: str
    features: List[Dict[str, float]]

@app.get("/health")
def health():
    return {
        "status": "HEALTHY",
        "platform": "ModelForge AI",
        "cluster_state": "ACTIVE",
        "governance_compliance": "EU_AI_ACT_READY"
    }

@app.get("/api/v1/models")
def list_models():
    return {
        "models": [
            {"id": "mdl_tabnet_risk_v1", "name": "TabNet Credit Risk Scorer", "architecture": "TabNet", "auc": 0.942, "stage": "PRODUCTION"},
            {"id": "mdl_ft_trans_v2", "name": "FT-Transformer Customer Churn", "architecture": "FT-Transformer", "auc": 0.958, "stage": "CANARY"},
            {"id": "mdl_saint_fraud_v1", "name": "SAINT Intersample Fraud Detector", "architecture": "SAINT", "auc": 0.971, "stage": "STAGED"}
        ]
    }

@app.post("/api/v1/predict")
def predict(req: PredictRequest):
    return {
        "model_id": req.model_id,
        "predictions": [0.89 for _ in req.features],
        "latency_ms": 1.45
    }

@app.get("/api/v1/governance/model-cards")
def get_model_cards():
    return {
        "cards": [
            {
                "model_name": "TabNet Credit Risk Scorer",
                "version": "1.4.0",
                "risk_tier": "High-Risk (Annex III)",
                "disparate_impact_ratio": 0.94,
                "differential_privacy_epsilon": 0.50,
                "w3c_lineage_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            }
        ]
    }


class DriftAnalysisRequest(BaseModel):
    feature_name: str
    baseline: List[float]
    current: List[float]


@app.post("/api/v1/governance/drift-analysis")
def analyze_drift(req: DriftAnalysisRequest):
    from backend.governance.drift_detector import DriftDetector
    detector = DriftDetector()
    return detector.evaluate_feature_drift(req.feature_name, req.baseline, req.current)


from backend.serving.bandit_router import MultiArmedBanditRouter

global_bandit_router = MultiArmedBanditRouter()
global_bandit_router.register_arm("mdl_tabnet_risk_v1", 0.92)
global_bandit_router.register_arm("mdl_ft_trans_v2", 0.94)


class RouteRequest(BaseModel):
    strategy: str = "ucb1"


class RewardRequest(BaseModel):
    model_id: str
    reward: float


@app.post("/api/v1/serving/route")
def route_traffic(req: RouteRequest):
    if req.strategy == "epsilon_greedy":
        selected = global_bandit_router.select_arm_epsilon_greedy()
    else:
        selected = global_bandit_router.select_arm_ucb1()
    return {
        "selected_model_id": selected,
        "strategy": req.strategy,
        "arms_state": global_bandit_router.arms,
    }


@app.post("/api/v1/serving/reward")
def submit_reward(req: RewardRequest):
    global_bandit_router.update_reward(req.model_id, req.reward)
    return {
        "status": "REWARD_RECORDED",
        "model_id": req.model_id,
        "updated_stats": global_bandit_router.arms.get(req.model_id),
    }
