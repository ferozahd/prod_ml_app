from prod_ml.core.interfaces.evaluation_metric import EvaluationMetric
from prod_ml.resources.evaluation_result import EvaluationResult


class EvaluationService:

    def __init__(
        self,
        metrics: list[EvaluationMetric],
        primary_metric: str
    ):
        self.metrics = metrics
        self.primary_metric = primary_metric

        names = [metric.name for metric in metrics]

        if len(names) != len(set(names)):
            raise ValueError("Duplicate metric names")

        if primary_metric not in names:
            raise ValueError("Primary metric not configured")

    def evaluate(self, model_artifact, training_data):

        # Evaluate on held-out validation data
        model = model_artifact.model

        y_true = training_data.y_val
        y_pred = model.predict(training_data.X_val)

        results = {}

        for metric in self.metrics:

            results[metric.name] = metric.calculate(
                y_true,
                y_pred
            )

        return EvaluationResult(
            score=results[self.primary_metric],
            metrics=results
        )