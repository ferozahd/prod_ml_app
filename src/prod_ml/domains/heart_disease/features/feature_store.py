# domains/fraud_detection/features/feature_store.py
from typing import List, Dict, Any

class FraudFeatureStoreClient:
    def __init__(self, store_backend):
        self.store = store_backend

    def get_online_features(self, entity_keys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Low-latency fetch for FastAPI inference endpoints (<10ms)."""
        return self.store.get_online_features(
            keys=entity_keys,
            features=["fraud_features:tx_count_24h", "fraud_features:avg_tx_amount_7d"]
        )

    def get_historical_features(self, entity_df, timestamp_col: str):
        """Point-in-time join to prevent target leakage during training."""
        return self.store.get_historical_features(entity_df=entity_df, timestamp_col=timestamp_col)