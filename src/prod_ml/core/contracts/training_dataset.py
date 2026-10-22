from typing import TypeVar, Generic

T = TypeVar("T")


class TrainingDataset(Generic[T]):

    features: T
    target: T
    metadata: dict