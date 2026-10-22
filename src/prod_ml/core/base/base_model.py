"""
Abstract Base Class for all ML Models in the Enterprise Application.

Every domain model (XGBoost, PyTorch, LightGBM, Scikit-Learn) MUST inherit
from BaseModel and implement its abstract methods to guarantee interface uniformity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract Base Class defining the contract for model lifecycle management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model hyperparameter configurations and state.

        Args:
            config: Optional dictionary containing model hyperparameters or metadata.
        """
        self.config = config or {}
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            eval_set: Optional[list] = None,
            **kwargs: Any
    ) -> "BaseModel":
        """
        Trains the model on input features X and target y.

        Must set self.is_fitted = True upon completion and return self.
        """
        pass

    @abstractmethod
    def predict(
            self,
            X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generates point predictions (class labels or regression targets).

        Args:
            X: Input feature matrix.

        Returns:
            1D numpy array of predictions.
        """
        pass

    @abstractmethod
    def predict_proba(
            self,
            X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generates probability estimates for classification models.

        Args:
            X: Input feature matrix.

        Returns:
            Numpy array of shape (N, C) or (N,) containing prediction probabilities.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Serializes model artifacts and state to storage (S3, local disk, ONNX, etc.).

        Args:
            filepath: Target path for the saved model artifact.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseModel":
        """
        Deserializes model artifacts and restores model state from storage.

        Args:
            filepath: Path to the serialized model artifact.

        Returns:
            An instantiated instance of BaseModel with loaded weights/state.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Returns model hyperparameters and configuration.
        """
        return self.config

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(is_fitted={self.is_fitted})>"