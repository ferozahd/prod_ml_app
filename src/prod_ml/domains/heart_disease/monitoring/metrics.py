# domains/fraud_detection/monitoring/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Any


class FraudDomainMetrics:
    """Computes domain-specific statistical and business metrics."""

    @staticmethod
    def compute_business_impact(
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            threshold: float,
            avg_chargeback_cost: float = 150.0,
            false_positive_penalty: float = 15.0
    ) -> Dict[str, float]:
        """Calculates net business savings vs loss from model decisions."""
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))  # Caught fraud
        fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed fraud (Chargeback)
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarm (Customer friction)

        fraud_caught_savings = tp * avg_chargeback_cost
        chargeback_losses = fn * avg_chargeback_cost
        friction_cost = fp * false_positive_penalty

        net_roi = fraud_caught_savings - (chargeback_losses + friction_cost)

        return {
            "fraud_caught_count": int(tp),
            "missed_fraud_count": int(fn),
            "false_positive_count": int(fp),
            "net_roi_usd": round(net_roi, 2),
            "block_rate": float(np.mean(y_pred)),
        }