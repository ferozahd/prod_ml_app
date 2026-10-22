import pandas as pd

from prod_ml.core.contracts.training_dataset import TrainingDataset


class PandasDatasetAdapter:

    def to_dataframe(
        self,
        dataset: TrainingDataset
    ):
        return pd.DataFrame(
            dataset.features
        )