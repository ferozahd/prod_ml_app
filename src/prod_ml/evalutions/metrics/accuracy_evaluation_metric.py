from sklearn.metrics import accuracy_score


from prod_ml.core.interfaces.evaluation_metric import EvaluationMetric


class AccuracyMetric(EvaluationMetric):


    @property
    def name(self):
        return "accuracy"

    def calculate(self, y_true, y_pred=None, y_score=None, **kwargs) -> float:
        return accuracy_score(y_true, y_pred)