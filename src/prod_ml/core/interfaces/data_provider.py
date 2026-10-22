from abc import ABC, abstractmethod
import pandas as pd



class TrainingDataProvider(ABC):
    """
    Contract between data world and ML world.

    Any data source must implement this.
    """

    @abstractmethod
    def get_training_dataset(self) -> pd.DataFrame:
        pass