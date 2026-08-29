"""
Multi-Armed Bandit (MAB) Dynamic Traffic Allocator.
Implements Epsilon-Greedy and UCB1 algorithms for automated champion/challenger online experimentation.
"""

from typing import Dict, List, Any, Optional
import math
import random


class MultiArmedBanditRouter:
    """Dynamically balances exploration and exploitation across competing ML model versions."""

    def __init__(self, epsilon: float = 0.1, exploration_constant: float = 1.414):
        self.epsilon = epsilon
        self.c = exploration_constant
        self.arms: Dict[str, Dict[str, Any]] = {}
        self.total_pulls = 0

    def register_arm(self, model_id: str, initial_value: float = 0.0) -> None:
        if model_id not in self.arms:
            self.arms[model_id] = {
                "pulls": 0,
                "total_reward": 0.0,
                "average_reward": initial_value,
            }

    def select_arm_epsilon_greedy(self) -> str:
        if not self.arms:
            raise ValueError("No model arms registered in bandit router.")

        # Epsilon chance of random exploration
        if random.random() < self.epsilon:
            return random.choice(list(self.arms.keys()))

        # Exploitation: pick arm with highest average reward
        best_arm = max(self.arms.keys(), key=lambda arm: self.arms[arm]["average_reward"])
        return best_arm

    def select_arm_ucb1(self) -> str:
        if not self.arms:
            raise ValueError("No model arms registered in bandit router.")

        # Ensure all arms are pulled at least once
        for arm, stats in self.arms.items():
            if stats["pulls"] == 0:
                return arm

        # Calculate Upper Confidence Bound for each arm
        best_score = -float("inf")
        best_arm = None

        log_total = math.log(max(1, self.total_pulls))
        for arm, stats in self.arms.items():
            avg_r = stats["average_reward"]
            confidence_bound = self.c * math.sqrt(log_total / stats["pulls"])
            ucb_score = avg_r + confidence_bound

            if ucb_score > best_score:
                best_score = ucb_score
                best_arm = arm

        return best_arm or random.choice(list(self.arms.keys()))

    def update_reward(self, model_id: str, reward: float) -> None:
        if model_id not in self.arms:
            self.register_arm(model_id)

        stats = self.arms[model_id]
        stats["pulls"] += 1
        stats["total_reward"] += reward
        stats["average_reward"] = stats["total_reward"] / stats["pulls"]
        self.total_pulls += 1
