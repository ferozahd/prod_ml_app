# domains/fraud_detection/pipelines/evaluation_pipeline.py
import logging
import pandas as pd
from typing import Dict, Any

from prod_ml.domains.heart_disease.models import compute_domain_metrics, FraudPredictor

logger = logging.getLogger(__name__)


class FraudEvaluationPipeline:
    """Runs evaluation backtests on historical datasets or live production logs."""

    def __init__(self, model_path: str):
        self.predictor = FraudPredictor(model_path=model_path)

    def run_backtest(self, test_data_path: str, feature_cols: list, label_col: str) -> Dict[str, Any]:
        """Executes offline evaluation backtest."""
        logger.info(f"Starting model backtesting on {test_data_path}...")

        test_df = pd.read_parquet(test_data_path)
        scored_df = self.predictor.predict_batch(test_df, feature_columns=feature_cols)

        metrics = compute_domain_metrics(
            y_true=scored_df[label_col].values,
            y_pred_probs=scored_df["fraud_probability"].values
        )

        logger.info(f"Backtest completed. Metrics: {metrics}")
        return metrics