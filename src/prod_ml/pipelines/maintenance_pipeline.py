"""
Global System Maintenance Pipeline.
Runs system-wide administrative audits and housekeeping tasks across all domains.
"""

import logging
from typing import Dict, Any, List
import pandas as pd

from prod_ml.pipelines.registry import PipelineRegistry
from prod_ml.domains.heart_disease.monitoring import FeatureDriftDetector as FraudDrift
from prod_ml.domains.heart_disease.monitoring import FeatureDriftDetector as RecSysDrift

logger = logging.getLogger(__name__)


class GlobalSystemHealthCheckPipeline:
    """Runs global drift detection and health auditing across all active enterprise domains."""

    def __init__(self, domain_baseline_paths: Dict[str, str]):
        self.baseline_paths = domain_baseline_paths

    def run_global_drift_audit(self, current_batch_paths: Dict[str, str]) -> Dict[str, Any]:
        """Iterates through all registered domains and audits feature drift."""
        logger.info("--- Starting Global Multi-Domain Feature Drift Audit ---")

        audit_results = {}
        flagged_domains_for_retraining = []

        for domain, baseline_path in self.baseline_paths.items():
            current_path = current_batch_paths.get(domain)
            if not current_path:
                logger.warning(f"No current batch data provided for domain '{domain}'. Skipping.")
                continue

            logger.info(f"Auditing domain: {domain}...")
            ref_df = pd.read_parquet(baseline_path)
            curr_df = pd.read_parquet(current_path)

            # Route to domain-specific drift analyzers
            if domain == "fraud_detection":
                detector = FraudDrift(baseline_df=ref_df, numerical_features=["tx_amount", "avg_tx_amount_7d"])
                report = detector.analyze_batch_drift(curr_df)
            elif domain == "user_recommendation":
                detector = RecSysDrift(baseline_df=ref_df, numerical_features=["user_click_rate", "item_popularity"])
                report = detector.analyze_batch_drift(curr_df)
            else:
                continue

            audit_results[domain] = report["summary"]
            if report["summary"]["requires_retraining"]:
                flagged_domains_for_retraining.append(domain)

        logger.info(f"Audit completed. Domains needing retraining: {flagged_domains_for_retraining}")
        return {
            "status": "COMPLETED",
            "flagged_domains": flagged_domains_for_retraining,
            "detailed_reports": audit_results,
        }