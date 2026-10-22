"""
ONNX Model Export & Inference Runtime Layer.
Handles converting trained models to the portable ONNX format and 
executing sub-millisecond predictions via ONNX Runtime.
"""

import logging
import time
from typing import Any, List, Union
import numpy as np
import pandas as pd

# ONNX Conversion dependencies
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. MODEL EXPORTER FUNCTION
# -----------------------------------------------------------------------------
def export_to_onnx(
        model: Any,
        sample_input: Union[pd.DataFrame, np.ndarray],
        export_path: str,
        target_opset: int = 15
) -> str:
    """
    Converts a trained scikit-learn/XGBoost model to ONNX format and saves it to disk.

    Args:
        model: Trained scikit-learn pipeline or XGBoost/LightGBM model instance.
        sample_input: Sample DataFrame or NumPy array representing model input shape.
        export_path: File path where the output '.onnx' artifact should be written.
        target_opset: ONNX operator set version (default: 15).

    Returns:
        str: Path to the exported ONNX artifact.
    """
    logger.info(f"Converting model '{type(model).__name__}' to ONNX format...")

    # Determine number of features from sample input
    num_features = sample_input.shape[1]

    # Define input type specification for ONNX tensor shape: [Batch_Size (Dynamic), Num_Features]
    initial_types = [("float_input", FloatTensorType([None, num_features]))]

    try:
        # Convert scikit-learn / XGBoost model to ONNX format
        onnx_model = skl2onnx.convert_sklearn(
            model,
            initial_types=initial_types,
            target_opset=target_opset,
            options={"zipmap": False}  # Disables dict output format to preserve clean array probability outputs
        )

        # Save binary artifact to target path
        with open(export_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Successfully exported ONNX model to: {export_path}")
        return export_path

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {str(e)}")
        raise RuntimeError(f"ONNX conversion error: {e}") from e


# -----------------------------------------------------------------------------
# 2. LOW-LATENCY ONNX INFERENCE RUNNER CLASS
# -----------------------------------------------------------------------------
class ONNXInferenceRunner:
    """
    High-performance ONNX Runtime wrapper for executing real-time online 
    and batch predictions with sub-millisecond latency.
    """

    def __init__(self, model_path: str):
        """
        Initializes the ONNX runtime inference session.

        Args:
            model_path: Path to the saved '.onnx' model file.
        """
        self.model_path = model_path

        # Configure execution provider (CPU by default; can add 'CUDAExecutionProvider' for GPU)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

        # Extract metadata from ONNX computational graph
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Detect feature count expected by graph
        self.expected_features = self.session.get_inputs()[0].shape[1]

    def warmup(self, num_warmup_requests: int = 10) -> None:
        """
        Executes dummy predictions to pre-allocate memory buffers and initialize
        CPU/GPU execution providers to avoid cold-start latency spikes.
        """
        if self.expected_features is None:
            return

        dummy_matrix = np.zeros((1, self.expected_features), dtype=np.float32)
        logger.debug(f"Warming up ONNX runtime session with {num_warmup_requests} dummy inputs...")

        for _ in range(num_warmup_requests):
            self.session.run(self.output_names, {self.input_name: dummy_matrix})

    def run(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Executes real-time single-record scoring (<1ms target).

        Args:
            input_matrix: 2D NumPy array of shape (1, num_features) and dtype float32.

        Returns:
            np.ndarray: Vector of predicted probabilities for positive class (e.g. Heart Disease / Fraud).
        """
        input_data = self._validate_and_format_input(input_matrix)

        # Execute ONNX runtime graph
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Extract probability tensor (Output index 1 contains class probabilities in skl2onnx)
        if len(raw_outputs) > 1:
            probabilities = raw_outputs[1]  # Shape: [1, 2] -> [Prob_Class_0, Prob_Class_1]
            return probabilities[:, 1]  # Extract positive class probability
        else:
            return raw_outputs[0]

    def run_batch(self, batch_matrix: np.ndarray) -> np.ndarray:
        """
        Executes vectorized batch scoring over multiple rows.

        Args:
            batch_matrix: 2D NumPy array of shape (batch_size, num_features).

        Returns:
            np.ndarray: 1D NumPy array containing positive class probabilities for all rows.
        """
        input_data = self._validate_and_format_input(batch_matrix)
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_data})

        if len(raw_outputs) > 1:
            probabilities = raw_outputs[1]
            return probabilities[:, 1]
        else:
            return raw_outputs[0].flatten()

    def _validate_and_format_input(self, input_data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Ensures incoming data strictly matches ONNX tensor float32 requirements."""
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.to_numpy()

        if input_data.ndim == 1:
            input_data = np.expand_dims(input_data, axis=0)

        # Cast to float32 (ONNX requires explicit type matching)
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        return input_data