from prod_ml.core.interfaces.train_service import TrainingService


class DefaultTrainingService(TrainingService):

    def __init__(
        self,
        trainer,
        experiment_tracker,
        artifact_builder
    ):
        self.trainer = trainer
        self.experiment_tracker = experiment_tracker
        self.artifact_builder = artifact_builder


    def train(self, training_data):

        self.experiment_tracker.start_run()


        model = self.trainer.fit(
            training_data
        )


        self.experiment_tracker.log_model(
            model
        )


        artifact = self.artifact_builder.build(
            model
        )


        return artifact