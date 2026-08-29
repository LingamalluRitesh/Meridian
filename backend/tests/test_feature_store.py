"""Unit Tests for ModelForge AI Feature Store & AS-OF Joins."""
import pytest
from backend.feature_store.online_store import OnlineRedisFeatureStore
from backend.feature_store.asof_joiner import PointInTimeJoiner
from backend.feature_store.statistics import FeatureStatisticsEngine
import numpy as np

def test_online_redis_store():
    store = OnlineRedisFeatureStore()
    store.set_online_features("user_101", {"score": 750, "balance": 1200.50})
    res = store.get_online_features(["user_101", "user_999"], ["score", "balance"])
    assert len(res) == 2
    assert res[0]["score"] == 750
    assert res[1]["score"] is None

def test_point_in_time_asof_join():
    joiner = PointInTimeJoiner()
    observations = [
        {"user_id": "u1", "timestamp": 100, "label": 1},
        {"user_id": "u1", "timestamp": 250, "label": 0}
    ]
    features = [
        {"user_id": "u1", "timestamp": 50, "feature_a": 10.0},
        {"user_id": "u1", "timestamp": 200, "feature_a": 25.0}
    ]
    joined = joiner.asof_join(observations, features, entity_key="user_id", timestamp_key="timestamp")
    assert len(joined) == 2
    assert joined[0]["feature_a"] == 10.0
    assert joined[1]["feature_a"] == 25.0

def test_psi_calculation():
    base = np.random.normal(0, 1, 1000)
    curr = np.random.normal(0, 1, 1000)
    psi = FeatureStatisticsEngine.calculate_psi(base, curr)
    assert psi >= 0.0
