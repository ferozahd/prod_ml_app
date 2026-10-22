# domains/fraud_detection/models/__init__.py
from .lifecycle.train import run_training
from .lifecycle.evaluate import compute_domain_metrics
from .lifecycle.predict import run_inference
from .factory import ModelFactory
from .registry import DomainModelRegistry
#
__all__ = [
    "run_training",
    "compute_domain_metrics",
    "run_inference",
    "ModelFactory",
    "DomainModelRegistry",
]


class FraudPredictor:
    pass