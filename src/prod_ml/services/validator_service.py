from typing import Protocol

from prod_ml.core.interfaces.validator import Validator
from prod_ml.resources.validation_result import ValidationResult




class ValidatorService:
    """
    Executes a collection of validators.
    Business rules stay outside.
    """

    def __init__(
        self,
        validators: list[Validator]
    ):
        self.validators = validators


    def validate(
        self,
        data
    ) -> ValidationResult:

        errors = []

        for validator in self.validators:

            result = validator.validate(data)

            if not result.passed:
                errors.extend(
                    result.errors
                )


        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors
        )