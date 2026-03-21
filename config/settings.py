"""
config/settings.py
──────────────────────────────────────────────────────────────────────────────
Central configuration loaded from .env
EVERY module imports Settings from here.
Never hardcode URLs, credentials, or thresholds anywhere else.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
from functools import lru_cache
from typing import Literal
 
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
 
    # Application
    environment: Literal["development", "staging", "production"] = "development"
    secret_key: str = Field(default="dev-secret-key-change-me")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    log_level: str = "INFO"
 
    # Database
    database_url: str = "postgresql://sentinel:sentinel2024@localhost:5432/sentinel_db"
 
    # Redis
    redis_url: str = "redis://localhost:6379"
 
    # Kafka
    kafka_bootstrap_servers: str = "localhost:29092"
    kafka_schema_registry_url: str = "http://localhost:8081"
    kafka_topic_transactions_raw: str = "transactions-raw"
    kafka_topic_transactions_enriched: str = "transactions-enriched"
    kafka_topic_pulse_scores: str = "pulse-scores"
    kafka_consumer_group_id: str = "sentinel-group"
 
    # AWS
    aws_region: str = "ap-south-1"
    aws_access_key_id: str = "local"
    aws_secret_access_key: str = "local"
    dynamodb_endpoint: str = "http://localhost:8000"
    dynamodb_table_scores: str = "sentinel-customer-scores"
    dynamodb_table_interventions: str = "sentinel-interventions"
    dynamodb_table_audit: str = "sentinel-audit-log"
    dynamodb_table_transactions: str = "sentinel-transactions"
 
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "sentinel-risk-models"
 
    # Model thresholds (tier thresholds are in scoring_utils.py — single source of truth)
    drift_threshold: float = 1.5
    psi_threshold: float = 0.25
    air_threshold: float = 0.80
 
    # Intervention
    intervention_cooldown_hours: int = 72
    max_interventions_per_month: int = 3
 
    # Bedrock
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
 
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
 
    @property
    def kafka_servers_list(self) -> list[str]:
        return self.kafka_bootstrap_servers.split(",")
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Import this everywhere."""
    return Settings()
