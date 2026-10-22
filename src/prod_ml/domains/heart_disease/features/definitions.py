# domains/fraud_detection/features/definitions.py
from dataclasses import dataclass

@dataclass(frozen=True)
class FraudFeatureSet:
    entity_id: str = "user_id"
    features: list[str] = (
        "tx_count_1h",
        "tx_count_24h",
        "avg_tx_amount_7d",
        "is_high_risk_country",
    )