"""
Global Pipeline Registry.
Exposes a central look-up interface for CLI triggers, Airflow DAGs, and API routers.
"""

import logging
from typing import Dict, Type, Any

# Import domain pipelines
from prod_ml.domains.heart_disease.pipelines import (
    FraudTrainingPipeline,
    FraudInferencePipeline,
)
from prod_ml.domains.heart_disease.pipelines import (
    RecSysTrainingPipeline,
    RecSysInferencePipeline,
)

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Central registry mapping 'domain:action' strings to executable pipeline classes."""

    # Explicit pipeline mapping
    _REGISTRY: Dict[str, Type[Any]] = {
        "fraud_detection:train": FraudTrainingPipeline,
        "fraud_detection:infer": FraudInferencePipeline,
        "user_recommendation:train": RecSysTrainingPipeline,
        "user_recommendation:infer": RecSysInferencePipeline,
    }

    @classmethod
    def get(cls, domain: str, action: str) -> Type[Any]:
        """
        Retrieves a pipeline class given a domain and action.

        Usage:
            PipelineClass = PipelineRegistry.get("fraud_detection", "train")
            pipeline_instance = PipelineClass(config)
        """
        key = f"{domain}:{action}".lower()
        if key not in cls._REGISTRY:
            available_keys = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Pipeline '{key}' not found in global registry. "
                f"Available registered pipelines: {available_keys}"
            )
        logger.info(f"Resolved pipeline key '{key}' to {cls._REGISTRY[key].__name__}")
        return cls._REGISTRY[key]

    @classmethod
    def list_registered_pipelines(cls) -> list[str]:
        """Returns all registered domain pipeline keys."""
        return list(cls._REGISTRY.keys())