from sklearn.metrics import f1_score

from prod_ml.core.interfaces.evaluation_metric import EvaluationMetric


class F1Metric(EvaluationMetric):


    @property
    def name(self):
        return "f1"


    def calculate(self, y_true, y_pred=None, y_score=None, **kwargs) -> float:
        return f1_score(y_true, y_pred)
