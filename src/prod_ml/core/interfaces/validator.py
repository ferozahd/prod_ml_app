from typing import Protocol

from prod_ml.resources.validation_result import ValidationResult


class Validator(Protocol):

    def validate(self, data) -> ValidationResult:
        ...
