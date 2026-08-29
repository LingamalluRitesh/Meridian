import pytest
from backend.serving.bandit_router import MultiArmedBanditRouter
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_bandit_arm_registration_and_rewards():
    router = MultiArmedBanditRouter(epsilon=0.0)
    router.register_arm("model_v1")
    router.register_arm("model_v2")

    router.update_reward("model_v1", 0.5)
    router.update_reward("model_v2", 0.9)

    selected = router.select_arm_epsilon_greedy()
    assert selected == "model_v2"


def test_bandit_ucb1_exploration():
    router = MultiArmedBanditRouter()
    router.register_arm("model_a")
    router.register_arm("model_b")

    # Initial selections must try unpulled arms
    arm1 = router.select_arm_ucb1()
    router.update_reward(arm1, 1.0)
    arm2 = router.select_arm_ucb1()
    assert arm1 != arm2


def test_bandit_api_routes():
    res = client.post("/api/v1/serving/route", json={"strategy": "ucb1"})
    assert res.status_code == 200
    data = res.json()
    assert "selected_model_id" in data

    reward_res = client.post(
        "/api/v1/serving/reward",
        json={"model_id": data["selected_model_id"], "reward": 0.95},
    )
    assert reward_res.status_code == 200
    assert reward_res.json()["status"] == "REWARD_RECORDED"
