# domains/fraud_detection/models/lifecycle/evaluate.py
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np


def compute_domain_metrics(y_true, y_pred_probs) -> dict:
    auc = roc_auc_score(y_true, y_pred_probs)

    # Compute optimal decision threshold for business ROI
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)

    return {
        "roc_auc": float(auc),
        "optimal_threshold": float(thresholds[optimal_idx]),
        "max_f1": float(f1_scores[optimal_idx])
    }