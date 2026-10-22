from prod_ml.core.interfaces.model_registry import ModelRegistry

class RegistryService:

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def register(self, model_artifact, evaluation_result):

        return self.registry.register(
            model_artifact,
            evaluation_result
        )