
import pandas as pd

from prod_ml.resources.validation_result import ValidationResult
from prod_ml.services.validator_service import Validator


class SchemaValidator(Validator):

    REQUIRED_COLUMNS = (
        "Age", "Sex", "ChestPainType", "RestingBP",
        "Cholesterol", "FastingBS", "RestingECG",
        "MaxHR", "ExerciseAngina", "Oldpeak",
        "ST_Slope", "HeartDisease"
    )

    NUMERIC_COLUMNS = (
        "Age", "RestingBP", "Cholesterol",
        "FastingBS", "MaxHR", "Oldpeak",
        "HeartDisease"
    )

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(False, ["Expected DataFrame"])

        errors = []

        if data.empty:
            errors.append("Dataset is empty")

        duplicates = data.columns[
            data.columns.duplicated()
        ].tolist()

        if duplicates:
            errors.append(f"Duplicate columns: {duplicates}")

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in data.columns
        ]

        if missing:
            errors.append(f"Missing columns: {missing}")

        for col in self.NUMERIC_COLUMNS:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"{col} must be numeric")

        return ValidationResult(not errors, errors)
