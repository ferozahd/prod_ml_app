from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationResult:
    score: float
    metrics: Dict[str, float]