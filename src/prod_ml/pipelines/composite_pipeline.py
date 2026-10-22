"""
Global Composite Pipeline.
Coordinates multi-domain execution sequences where output from one domain feeds into another.
"""

import logging
from typing import Dict, Any, List

from prod_ml.domains.user_recommendation.pipelines import RecSysInferencePipeline
from prod_ml.domains.fraud_detection.pipelines import FraudInferencePipeline

logger = logging.getLogger(__name__)


class CheckoutOrchestratorPipeline:
    """
    Cross-domain workflow combining Personalization and Fraud Detection
    during user checkout sessions.
    """

    def __init__(self, recsys_model_path: str, fraud_model_path: str):
        self.recsys_pipeline = RecSysInferencePipeline(model_path=recsys_model_path)
        self.fraud_pipeline = FraudInferencePipeline(model_path=fraud_model_path)

    def execute_checkout_flow(
        self, user_id: str, cart_items: List[Dict[str, Any]], total_amount: float
    ) -> Dict[str, Any]:
        """Executes sequential domain steps."""
        logger.info(f"--- Starting Composite Checkout Flow for User: {user_id} ---")

        # Step 1: Run Personalization domain to get real-time add-on upsell recommendations
        logger.info("Step 1: Generating user recommendations...")
        recommended_addons = self.recsys_pipeline.get_top_k_recommendations(
            user_id=user_id, k=3
        )

        # Step 2: Run Fraud Detection domain to evaluate transaction risk
        logger.info("Step 2: Assessing fraud risk score...")
        fraud_assessment = self.fraud_pipeline.evaluate_transaction_risk(
            user_id=user_id,
            transaction_amount=total_amount,
            cart_item_count=len(cart_items)
        )

        # Step 3: Global System Business Rule Evaluation
        if fraud_assessment["action"] == "BLOCK":
            logger.warning(f"Checkout blocked for User {user_id} due to high risk score.")
            return {
                "checkout_status": "REJECTED",
                "reason": "HIGH_FRAUD_RISK",
                "recommended_addons": [],
            }

        logger.info(f"Checkout approved for User {user_id}.")
        return {
            "checkout_status": "APPROVED",
            "fraud_risk_score": fraud_assessment["risk_score"],
            "recommended_addons": recommended_addons,
        }