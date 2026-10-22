# domains/fraud_detection/pipelines/inference_pipeline.py
import logging
import pandas as pd
from typing import Dict, Any

from prod_ml.domains.heart_disease.features import FraudFeatureStoreClient
from prod_ml.domains.heart_disease.models import run_inference
from prod_ml.domains.heart_disease.monitoring import FeatureDriftDetector

logger = logging.getLogger(__name__)


class FraudInferencePipeline:
    """Executes scheduled batch inference and drift monitoring."""

    def __init__(self, model_artifact_path: str, baseline_data_path: str):
        self.model_path = model_artifact_path
        self.baseline_df = pd.read_parquet(baseline_data_path)

    def run(self, input_batch_path: str, output_sink_path: str, feature_cols: list) -> Dict[str, Any]:
        """Runs end-to-end batch inference pipeline."""
        logger.info(f"--- Starting Fraud Batch Inference Pipeline ---")

        # Step 1: Load Batch Feature Data
        logger.info(f"Step 1: Reading input inference batch from {input_batch_path}...")
        batch_df = pd.read_parquet(input_batch_path)

        # Step 2: Feature Drift Detection (Pre-Inference Observability)
        logger.info("Step 2: Checking batch data drift against training baseline...")
        drift_detector = FeatureDriftDetector(
            baseline_df=self.baseline_df,
            numerical_features=feature_cols
        )
        drift_report = drift_detector.analyze_batch_drift(batch_df)

        if drift_report["summary"]["requires_retraining"]:
            logger.warning("Significant drift detected during batch inference! Alerting monitoring channels...")

        # Step 3: Run Batch Scoring Execution
        logger.info("Step 3: Running batch model predictions...")
        scored_df = run_inference(
            model_path=self.model_path,
            features_df=batch_df,
            feature_columns=feature_cols
        )

        # Step 4: Export Scored Results to Target Data Sink (S3, Snowflake, BigQuery)
        logger.info(f"Step 4: Writing scored predictions to {output_sink_path}...")
        scored_df.to_parquet(output_sink_path, index=False)

        return {
            "status": "COMPLETED",
            "processed_records": len(scored_df),
            "drift_summary": drift_report["summary"]
        }