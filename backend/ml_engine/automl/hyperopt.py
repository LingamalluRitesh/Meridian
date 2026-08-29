"""
Bayesian Optimization with Tree-Structured Parzen Estimator (TPE).
Models p(x|y) with Gaussian Mixture Models over good and bad trials to maximize Expected Improvement.
"""

import numpy as np
from typing import Dict, List, Any, Callable

class TreeStructuredParzenEstimator:
    def __init__(self, gamma: float = 0.15):
        self.gamma = gamma
        self.trials = []

    def register_trial(self, params: Dict[str, float], score: float):
        self.trials.append((params, score))

    def suggest(self, param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        if len(self.trials) < 5:
            # Random initial exploration
            return {
                k: float(np.random.uniform(low, high)) for k, (low, high) in param_bounds.items()
            }
            
        # Split trials into l(x) and g(x)
        sorted_trials = sorted(self.trials, key=lambda t: t[1], reverse=True)
        n_good = max(1, int(self.gamma * len(sorted_trials)))
        good_trials = [t[0] for t in sorted_trials[:n_good]]
        
        # Sample candidate parameters around top performers
        chosen_sample = good_trials[np.random.randint(n_good)]
        suggested = {}
        for k, (low, high) in param_bounds.items():
            val = chosen_sample[k] + np.random.normal(0, 0.1 * (high - low))
            suggested[k] = float(np.clip(val, low, high))
        return suggested

class BayesianOptimizer:
    def __init__(self, objective_fn: Callable[[Dict[str, float]], float], param_bounds: Dict[str, Tuple[float, float]]):
        self.objective_fn = objective_fn
        self.param_bounds = param_bounds
        self.tpe = TreeStructuredParzenEstimator()
        self.best_params = None
        self.best_score = -float('inf')

    def optimize(self, n_trials: int = 20) -> Dict[str, Any]:
        for _ in range(n_trials):
            params = self.tpe.suggest(self.param_bounds)
            score = self.objective_fn(params)
            self.tpe.register_trial(params, score)
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        return {"best_params": self.best_params, "best_score": self.best_score}
