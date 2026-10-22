# domains/fraud_detection/monitoring/alerts.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DomainAlertManager:
    """Translates monitoring findings into operational alerts or trigger events."""

    def evaluate_and_notify(self, drift_summary: Dict[str, Any], quality_issues: Dict[str, Any]):
        """Evaluates health reports and routes actions."""

        # 1. Data quality failure alert
        if quality_issues:
            logger.critical(f"[ALERT - DATA QUALITY FAIL] Domain: Fraud. Issues: {quality_issues}")
            self._send_slack_notification(f"🚨 Data Quality check failed for Fraud pipeline: {quality_issues}")

        # 2. Severe feature drift trigger (Automated Retraining Signal)
        if drift_summary.get("requires_retraining"):
            logger.warning("[ALERT - FEATURE DRIFT] Over 30% of features have drifted. Triggering retraining workflow.")
            self._trigger_retraining_workflow(domain="fraud_detection", reason="feature_drift")

    def _send_slack_notification(self, message: str):
        """Mock Slack webhook integration."""
        logger.info(f"Notification sent to #fraud-ml-alerts: {message}")

    def _trigger_retraining_workflow(self, domain: str, reason: str):
        """Emits event to orchestrator (Argo/Airflow/Kubeflow) to run workflows/retraining.yaml."""
        logger.info(f"Triggering automated retraining pipeline for domain='{domain}' due to '{reason}'")