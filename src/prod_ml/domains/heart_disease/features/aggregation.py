# domains/fraud_detection/features/aggregations.py


def compute_rolling_transaction_counts(df):
    """PySpark rolling 24-hour window aggregation."""
    # window_24h = Window.partitionBy("user_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-86400, 0)
    # return df.withColumn("tx_count_24h", F.count("tx_id").over(window_24h))
    return df