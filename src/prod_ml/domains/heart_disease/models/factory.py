# domains/fraud_detection/models/factory.py
from .algorithms.xgboost_wrapper import FraudXGBoostModel

class ModelFactory:
    @staticmethod
    def create(model_type: str, params: dict):
        if model_type == "xgboost":
            return FraudXGBoostModel(params)
        raise ValueError(f"Unsupported model type: {model_type}")