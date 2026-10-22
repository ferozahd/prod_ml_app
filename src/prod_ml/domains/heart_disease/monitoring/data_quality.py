# domains/fraud_detection/monitoring/data_quality.py
import pandas as pd
from typing import Dict, List, Tuple, Any


class DomainDataQualityChecker:
    """Validates batch data quality invariants prior to feature store or model execution."""

    def __init__(self, required_columns: List[str], max_allowed_null_pct: float = 0.05):
        self.required_columns = required_columns
        self.max_allowed_null_pct = max_allowed_null_pct

    def inspect_batch(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Executes invariant health checks."""
        issues = {}

        # 1. Check missing required schema columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            issues["missing_columns"] = list(missing_cols)

        # 2. Check null rate per column
        null_rates = df[self.required_columns].isnull().mean().to_dict()
        high_null_cols = {col: round(rate, 4) for col, rate in null_rates.items() if rate > self.max_allowed_null_pct}

        if high_null_cols:
            issues["high_null_rate_columns"] = high_null_cols

        # 3. Check for exact duplicate records
        duplicate_count = int(df.duplicated(subset=["transaction_id"]).sum()) if "transaction_id" in df.columns else 0
        if duplicate_count > 0:
            issues["duplicate_records_count"] = duplicate_count

        passed = len(issues) == 0
        return passed, issues