"""
features/feast_repo/feature_views/behavioral_feature_view.py
Defines the Feast FeatureView for behavioral signals.
Online store (Redis) serves real-time features for scoring (<1ms).
Offline store (Parquet files) serves training data.
"""
from datetime import timedelta
from pathlib import Path
 
from feast import Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int32
 
from features.feast_repo.entities.customer_entity import customer
 
# Offline source — Parquet files written by the feature pipeline
OFFLINE_DATA_PATH = str(Path(__file__).parents[3] / "data" / "behavioral_features.parquet")
 
behavioral_source = FileSource(
    path=OFFLINE_DATA_PATH,
    timestamp_field="computed_at",
    created_timestamp_column="computed_at",
)
 
behavioral_feature_view = FeatureView(
    name="behavioral_signals",
    entities=[customer],
    ttl=timedelta(hours=24),    # Features expire after 24h — force refresh
    schema=[
        Feature(name="salary_delay_days",           dtype=Float32),
        Feature(name="balance_wow_drop_pct",        dtype=Float32),
        Feature(name="upi_lending_spike_ratio",     dtype=Float32),
        Feature(name="utility_payment_latency",     dtype=Float32),
        Feature(name="discretionary_contraction",   dtype=Float32),
        Feature(name="atm_withdrawal_spike",        dtype=Float32),
        Feature(name="failed_auto_debit_count",     dtype=Int32),
        Feature(name="credit_utilization_delta",    dtype=Float32),
        Feature(name="drift_score",                 dtype=Float32),
    ],
    online=True,
    source=behavioral_source,
    tags={"team": "sentinel", "version": "1.0"},
)
