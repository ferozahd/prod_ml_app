# domains/fraud_detection/features/transformations.py
import pandas as pd

def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes spend velocity ratios."""
    df["amount_to_avg_ratio"] = df["tx_amount"] / (df["avg_tx_amount_7d"] + 1e-5)
    return df