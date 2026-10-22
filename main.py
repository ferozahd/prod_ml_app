from datetime import datetime

from sklearn.linear_model import LogisticRegression

from prod_ml.services import RegistryService
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from prod_ml.registries.mlflow_registry import MLflowRegistry
from prod_ml.core.interfaces.data_provider import TrainingDataProvider
from prod_ml.domains.heart_disease.pipelines.training_pipeline import TrainingPipeline
from prod_ml.domains.heart_disease.services import DataQualityValidator, HeartCompleteFeatureTransformation, \
    SchemaValidator, CategoricalFeatureEncoder
from prod_ml.evalutions.metrics.accuracy_evaluation_metric import AccuracyMetric
from prod_ml.evalutions.metrics.f1_evalution_matric import F1Metric
from prod_ml.evalutions.metrics.precision_evalutaion_metric import PrecisionMetric
from prod_ml.services import ValidatorService, FeatureService, DefaultTrainingService
from prod_ml.services.evalution_service import EvaluationService

from types import SimpleNamespace

# from prod_ml.domains.heart_disease.pipelines import TrainingPipeline

# from prod_ml.data.ETL import load_dataset
# from prod_ml.models.train import run_training
# from prod_ml.domains.heart_disease.pipelines import HeartDiseaseTrainingPipeline

# @hydra.main(config_path="configs", config_name="base", version_base=None)
# def pipeline(cfg: DictConfig):
#     print("data path",cfg.domains.heart_disease.data_path)
#     pipeline = HeartDiseaseTrainingPipeline(config={"run_id": "run_001"})
#     result = pipeline.run(raw_data_path=cfg.domains.heart_disease.data_path)
#     # typer.echo(f"Training Result: {result}")
#     # print(cfg)
#     print("hello hello ")
#
#
#
# if __name__ == "__main__":
#     pipeline()

#
# # main.py
# import typer
# # from prod_ml.domains.heart_disease.pipelines import FraudTrainingPipeline, FraudInferencePipeline
#
# app = typer.Typer(help="Enterprise Production ML CLI Engine")
#
# @app.command()
# def train():
#     domain ='fraud_detection'
#     data_path ='something'
#     print(f"Training domain: {domain}")
#     """Triggers the training pipeline for a given domain."""
#     if domain == "fraud_detection":
#         # Load YAML config dict here...
#         # pipeline = FraudTrainingPipeline(config={"run_id": "run_001"})
#         # result = pipeline.run(raw_data_path=data_path)
#         # typer.echo(f"Training Result: {result}")
#         print("hello hello ")
#     else:
#         typer.echo(f"Unknown domain: {domain}", err=True)
#
# if __name__ == "__main__":
#     app()

class XGBoostTrainer:

    def __init__(self, params):
        self.params = params


    def fit(self, data):
        X = data.drop('HeartDisease', axis=1)
        y = data['HeartDisease']

        model = LogisticRegression(max_iter=2000)

        model.fit(X,y)

        return model



class MLFlowTracker:

    def start_run(self):
        mlflow.start_run()


    def log_model(self, model):
        mlflow.sklearn.log_model(
            model,
            "model"
        )


class ModelArtifact:
    def __init__(self, model, framework , created_at):
        self.model = model
        self.framework = framework
        self.created_at = created_at



class ModelArtifactBuilder:

    def build(self, model):

        return ModelArtifact(
            model=model,
            framework="xgboost",
            created_at=datetime.now()
        )
    
    
    
class DataSource(TrainingDataProvider):

    def get_training_dataset(self) -> pd.DataFrame:
        return pd.read_csv("data/heart.csv")

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    print("Hello world")
    context = SimpleNamespace(config= {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42
    })
    print(context.config)

    evaluation_service = EvaluationService(
        metrics=[
            F1Metric(),
            PrecisionMetric(),
            AccuracyMetric()
        ],
        primary_metric="f1"
    )

    registry_service = RegistryService(
        registry=MLflowRegistry(
            model_name="fraud_detection"
        )
    )

    datasource = DataSource()
    dataset =  datasource.get_training_dataset()
    print(dataset.columns)

    train_pipeline= TrainingPipeline(
        data_source=DataSource(),
        schema_validator= ValidatorService(validators=[SchemaValidator()]),
        feature_service=FeatureService(
            transformers=[HeartCompleteFeatureTransformation(),        CategoricalFeatureEncoder()]

        ),

        validator_service=ValidatorService(
            validators=[
                DataQualityValidator()
            ]
        ),

        training_service=DefaultTrainingService(
                trainer=XGBoostTrainer(context.config),
                experiment_tracker=MLFlowTracker(),
                artifact_builder=ModelArtifactBuilder()
        ),
        evalution_service=evaluation_service,
        registry_service=registry_service,
        context=context
    )


    print(train_pipeline.run())