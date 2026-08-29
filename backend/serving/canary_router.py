"""
Canary Traffic Splitter and Shadow Deployment Controller.
"""

import random
from typing import Tuple

class CanaryTrafficRouter:
    def __init__(self, baseline_model_id: str, canary_model_id: str, canary_weight: float = 0.10):
        self.baseline_id = baseline_model_id
        self.canary_id = canary_model_id
        self.canary_weight = canary_weight

    def route_request(self) -> Tuple[str, bool]:
        is_canary = random.random() < self.canary_weight
        chosen_id = self.canary_id if is_canary else self.baseline_id
        return chosen_id, is_canary
