from prod_ml.resources.validation_result import ValidationResult
from prod_ml.services.validator_service import Validator


import pandas as pd

from prod_ml.resources.validation_result import ValidationResult
from prod_ml.services.validator_service import Validator


class DataQualityValidator(Validator):

    POSITIVE_COLUMNS = ("Cholesterol", "RestingBP")

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(False, ["Expected DataFrame"])

        if data.columns.duplicated().any():
            return ValidationResult(
                False, ["Duplicate columns detected"]
            )

        errors = []

        # Missing values
        for col, count in data.isna().sum().items():
            if count > 0:
                errors.append(
                    f"{col}: {count} missing values"
                )

        # Invalid measurements
        for col in self.POSITIVE_COLUMNS:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")
                continue

            values = pd.to_numeric(data[col], errors="coerce")
            invalid_count = int((values <= 0).sum())

            if invalid_count > 0:
                errors.append(
                    f"{col}: {invalid_count} non-positive values"
                )

        return ValidationResult(not errors, errors)
