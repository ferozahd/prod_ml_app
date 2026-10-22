from abc import ABC, abstractmethod


class ModelRegistry(ABC):

    @abstractmethod
    def register(self, model_artifact, evaluation_result):
        pass