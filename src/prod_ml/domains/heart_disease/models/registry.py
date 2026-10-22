# domains/fraud_detection/models/registry.py
import mlflow

class DomainModelRegistry:
    def register_candidate_model(self, run_id: str, model_name: str = "fraud_detection_model"):
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        return result.version