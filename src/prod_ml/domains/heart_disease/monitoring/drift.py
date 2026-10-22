# domains/fraud_detection/monitoring/drift.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Any


class FeatureDriftDetector:
    """Detects distribution shift between training baselines and live inference batches."""

    def __init__(self, baseline_df: pd.DataFrame, numerical_features: List[str]):
        self.baseline_df = baseline_df
        self.numerical_features = numerical_features

    def compute_psi(self, reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index (PSI) for continuous features."""
        buckets = np.linspace(min(reference.min(), current.min()),
                              max(reference.max(), current.max()),
                              num_buckets + 1)

        ref_counts, _ = np.histogram(reference, bins=buckets)
        curr_counts, _ = np.histogram(current, bins=buckets)

        ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts) / len(reference)
        curr_pct = np.where(curr_counts == 0, 0.0001, curr_counts) / len(current)

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def analyze_batch_drift(self, current_df: pd.DataFrame, psi_threshold: float = 0.25) -> Dict[str, Any]:
        """Runs drift detection across all domain features."""
        drift_report = {}
        flagged_features = []

        for feature in self.numerical_features:
            ref_data = self.baseline_df[feature].dropna().values
            curr_data = current_df[feature].dropna().values

            # 1. Kolmogorov-Smirnov Test
            ks_stat, p_value = ks_2samp(ref_data, curr_data)

            # 2. Population Stability Index (PSI)
            psi_val = self.compute_psi(ref_data, curr_data)

            is_drifted = psi_val >= psi_threshold or p_value < 0.01
            if is_drifted:
                flagged_features.append(feature)

            drift_report[feature] = {
                "psi": round(psi_val, 4),
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": is_drifted
            }

        return {
            "summary": {
                "total_features": len(self.numerical_features),
                "drifted_features_count": len(flagged_features),
                "drifted_features": flagged_features,
                "requires_retraining": len(flagged_features) > (len(self.numerical_features) * 0.3)
            },
            "feature_details": drift_report
        }