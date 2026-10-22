from sklearn.metrics import precision_score

from prod_ml.core.interfaces.evaluation_metric import EvaluationMetric


class PrecisionMetric(EvaluationMetric):


    @property
    def name(self):
        return "precision"


    def calculate(self, y_true, y_pred=None, y_score=None, **kwargs) -> float:
        return precision_score(
                 y_true, y_pred, zero_division=0
        )