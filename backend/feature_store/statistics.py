"""
Feature Statistics & Drift Analysis Engine
Calculates Kolmogorov-Smirnov statistics, Population Stability Index (PSI), and Wasserstein distances.
"""

import numpy as np
from typing import Dict, Any

class FeatureStatisticsEngine:
    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Population Stability Index: sum( (Actual% - Expected%) * ln(Actual% / Expected%) )"""
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(baseline, quantiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        
        base_counts, _ = np.histogram(baseline, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)
        
        base_pct = base_counts / len(baseline) + 1e-5
        curr_pct = curr_counts / len(current) + 1e-5
        
        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi)

    @staticmethod
    def compute_summary_stats(data: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "p25": float(np.percentile(data, 25)),
            "median": float(np.median(data)),
            "p75": float(np.percentile(data, 75)),
            "max": float(np.max(data))
        }
