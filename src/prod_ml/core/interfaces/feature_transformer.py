from abc import abstractmethod
from typing import Protocol

from pandas import DataFrame


class FeatureTransformer(Protocol):

    @abstractmethod
    def fit(self, data: DataFrame) -> None:
        ...
    @abstractmethod
    def transform(self, data:DataFrame) -> DataFrame:
        ...
