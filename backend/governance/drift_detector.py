"""
Kolmogorov-Smirnov & Population Stability Index (PSI) Drift Detection Engine.
Calculates statistical feature and target drift across high-dimensional production ML inference batches.
"""

from typing import List, Dict, Any, Tuple
import math


class DriftDetector:
    """Computes statistical distribution shifts using KS-Test and PSI."""

    @staticmethod
    def calculate_psi(baseline: List[float], current: List[float], num_bins: int = 10) -> float:
        if not baseline or not current:
            return 0.0

        min_val = min(min(baseline), min(current))
        max_val = max(max(baseline), max(current))

        if min_val == max_val:
            return 0.0

        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins + 1)]

        def get_bin_counts(data: List[float]) -> List[int]:
            counts = [0] * num_bins
            for val in data:
                placed = False
                for i in range(num_bins):
                    if (i == num_bins - 1 and bins[i] <= val <= bins[i+1]) or (bins[i] <= val < bins[i+1]):
                        counts[i] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1
            return counts

        base_counts = get_bin_counts(baseline)
        curr_counts = get_bin_counts(current)

        n_base = len(baseline)
        n_curr = len(current)

        psi = 0.0
        for b_cnt, c_cnt in zip(base_counts, curr_counts):
            # Laplace smoothing to avoid division by zero
            base_pct = (b_cnt + 1e-4) / (n_base + 1e-4 * num_bins)
            curr_pct = (c_cnt + 1e-4) / (n_curr + 1e-4 * num_bins)
            psi += (curr_pct - base_pct) * math.log(curr_pct / base_pct)

        return round(abs(psi), 4)

    @staticmethod
    def calculate_ks_statistic(baseline: List[float], current: List[float]) -> float:
        """Calculates 2-sample Kolmogorov-Smirnov maximum empirical CDF distance."""
        if not baseline or not current:
            return 0.0

        all_vals = sorted(list(set(baseline + current)))
        n_base = len(baseline)
        n_curr = len(current)

        max_diff = 0.0
        for v in all_vals:
            cdf_base = sum(1 for x in baseline if x <= v) / n_base
            cdf_curr = sum(1 for x in current if x <= v) / n_curr
            diff = abs(cdf_base - cdf_curr)
            if diff > max_diff:
                max_diff = diff

        return round(max_diff, 4)

    def evaluate_feature_drift(
        self,
        feature_name: str,
        baseline: List[float],
        current: List[float],
    ) -> Dict[str, Any]:
        psi = self.calculate_psi(baseline, current)
        ks = self.calculate_ks_statistic(baseline, current)

        if psi < 0.1:
            status = "NO_DRIFT"
            action = "CONTINUE_NORMAL_OPERATION"
        elif psi < 0.25:
            status = "MODERATE_DRIFT"
            action = "SCHEDULE_MODEL_RETRAINING"
        else:
            status = "SEVERE_DRIFT"
            action = "TRIGGER_AUTOMATED_FALLBACK"

        return {
            "feature_name": feature_name,
            "population_stability_index": psi,
            "ks_statistic": ks,
            "drift_status": status,
            "recommended_action": action,
        }
