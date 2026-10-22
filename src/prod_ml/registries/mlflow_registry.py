import mlflow
from prod_ml.core.interfaces.model_registry import ModelRegistry

class MLflowRegistry(ModelRegistry):

    def __init__(self, model_name):
        self.model_name = model_name

    def register(self, model_artifact, evaluation_result):

        result = mlflow.register_model(
            model_uri=model_artifact.model_uri,
            name=self.model_name
        )

        return result