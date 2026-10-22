from abc import ABC, abstractmethod


class EvaluationMetric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def calculate(
            self,
            y_true,
            y_pred=None,
            y_score=None,
            **kwargs
    ) -> float:
        ...