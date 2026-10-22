from typing import Any, Dict
import logging

from prod_ml.core.interfaces.data_provider import TrainingDataProvider
from prod_ml.core.interfaces.train_service import TrainingService
from prod_ml.domains.heart_disease.services import SchemaValidator
from prod_ml.services.evalution_service import  EvaluationService
from prod_ml.services.feature_transformer import FeatureService
from prod_ml.services.validator_service import ValidatorService
from prod_ml.services import RegistryService

from prod_ml.utils.adapters.pandas_dataset_adapter import PandasDatasetAdapter

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the complete ML training lifecycle.

    Principle:
    - Pipeline coordinates workflow only.
    - Business logic stays inside injected services.
    - Data source, model, evaluation, and registry implementations
      are external concerns.
    """

    def __init__(self,
                 data_source:TrainingDataProvider,
                 schema_validator:ValidatorService,
                 feature_service:FeatureService,
                 validator_service:ValidatorService,
                 training_service:TrainingService,
                 evalution_service:EvaluationService,
                 registry_service:RegistryService,
                 context: Any):
        """
        Context contains all required ML services.

        Expected context:
            context.data_service
            context.validation_service
            context.feature_service
            context.training_service
            context.evaluation_service
            context.registry_service
            context.config
        """

        self.context = context
        self.schema_validator = schema_validator
        self.data_source:TrainingDataProvider = data_source
        self.feature_service = feature_service
        self.validation_service = validator_service
        self.training_service = training_service
        self.evaluation_service = evalution_service,
        self.registry_service = registry_service
        self.config = context.config


    def run(self) -> Dict[str, Any]:
        """
        Execute end-to-end training workflow.
        """

        logger.info(
            "Starting training pipeline"
        )


        # --------------------------------------------------
        # Step 1: Data Acquisition
        # --------------------------------------------------

        logger.info(
            "Loading training dataset"
        )

        df = (
            self.data_source
            .get_training_dataset()
        )

        schema_validate_result=self.schema_validator.validate(df)
        if not schema_validate_result.passed:
            raise RuntimeError(
                f"Data validation failed: "
                f"{schema_validate_result.errors}"
            )

        # Convert at boundary

        # --------------------------------------------------
        # Step 2: Feature Engineering
        # --------------------------------------------------

        logger.info(
            "Generating features"
        )

        df = (
            self.feature_service
            .fit_transform(df)
        )

        # --------------------------------------------------
        # Step 3: Data Validation
        # --------------------------------------------------

        logger.info(
            "Validating dataset quality"
        )

        validation_result = (
            self.validation_service
            .validate(df)
        )


        if not validation_result.passed:

            raise RuntimeError(
                f"Data validation failed: "
                f"{validation_result.errors}"
            )



        # --------------------------------------------------
        # Step 4: Model Training
        # --------------------------------------------------

        logger.info(
            "Training model"
        )

        model_artifact = (
            self.training_service
            .train(df)
        )


        # --------------------------------------------------
        # Step 5: Model Evaluation
        # --------------------------------------------------

        logger.info(
            "Evaluating model"
        )

        evaluation_result = (
            self.evaluation_service
            .evaluate(
                model_artifact,
                df
            )
        )


        # --------------------------------------------------
        # Step 6: Quality Gate
        # --------------------------------------------------

        deployment_config = (
            self.config
            .deployment
        )

        minimum_score = (
            deployment_config
            .minimum_metric
        )


        if (
            evaluation_result.score
            <
            minimum_score
        ):

            logger.warning(
                "Model failed quality gate"
            )

            return {

                "status": "REJECTED",

                "metrics":
                    evaluation_result.metrics

            }


        # --------------------------------------------------
        # Step 7: Model Registration
        # --------------------------------------------------

        logger.info(
            "Registering model"
        )

        registry_result = (
            self.registry_service
            .register(
                model_artifact,
                evaluation_result
            )
        )


        logger.info(
            "Training pipeline completed successfully"
        )


        return {

            "status": "SUCCESS",

            "model_version":
                registry_result.version,

            "metrics":
                evaluation_result.metrics

        }