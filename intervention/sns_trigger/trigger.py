"""
intervention/sns_trigger/trigger.py
──────────────────────────────────────────────────────────────────────────────
Checks new Pulse Scores in DynamoDB and triggers interventions when
scores cross risk thresholds. Enforces cooldown periods to prevent
overwhelming customers with messages.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
 
import boto3
from botocore.exceptions import ClientError
 
from config.logging_config import get_logger
from config.settings import get_settings
 
logger = get_logger(__name__)
settings = get_settings()
 
 
class InterventionTrigger:
    def __init__(self) -> None:
        self._dynamodb = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        self._scores_table = self._dynamodb.Table(settings.dynamodb_table_scores)
        self._interventions_table = self._dynamodb.Table(settings.dynamodb_table_interventions)
 
    def _is_in_cooldown(self, customer_id: str) -> bool:
        """Check if customer received an intervention recently (cooldown period)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=settings.intervention_cooldown_hours)).isoformat()
        try:
            response = self._interventions_table.query(
                KeyConditionExpression="customer_id = :cid AND triggered_at > :cutoff",
                ExpressionAttributeValues={":cid": customer_id, ":cutoff": cutoff},
                Limit=1,
            )
            return len(response.get("Items", [])) > 0
        except ClientError:
            return False
 
    def _count_monthly_interventions(self, customer_id: str) -> int:
        """Count interventions sent this calendar month."""
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0).isoformat()
        try:
            response = self._interventions_table.query(
                KeyConditionExpression="customer_id = :cid AND triggered_at > :start",
                ExpressionAttributeValues={":cid": customer_id, ":start": month_start},
                Select="COUNT",
            )
            return response.get("Count", 0)
        except ClientError:
            return 0
 
    def process_score(self, customer_id: str, pulse_score: int, risk_tier: str, top_factor: str) -> Optional[str]:
        """
        Evaluate a new Pulse Score and trigger intervention if warranted.
        Returns intervention type if triggered, None otherwise.
        """
        # Only intervene for orange and red
        if risk_tier not in ("orange", "red"):
            return None
 
        # Cooldown check
        if self._is_in_cooldown(customer_id):
            logger.debug("Customer in cooldown", customer_id=customer_id)
            return None
 
        # Monthly limit check
        if self._count_monthly_interventions(customer_id) >= settings.max_interventions_per_month:
            logger.debug("Monthly intervention limit reached", customer_id=customer_id)
            return None
 
        # Determine intervention type
        if risk_tier == "red":
            intervention_type = "payment_holiday"
        else:
            intervention_type = "flexible_emi"
 
        # Log to DynamoDB
        now = datetime.now(timezone.utc).isoformat()
        self._interventions_table.put_item(Item={
            "customer_id":        customer_id,
            "triggered_at":       now,
            "intervention_type":  intervention_type,
            "pulse_score":        pulse_score,
            "risk_tier":          risk_tier,
            "top_factor":         top_factor,
            "status":             "pending",
        })
 
        logger.info(
            "Intervention triggered",
            customer_id=customer_id,
            intervention_type=intervention_type,
            pulse_score=pulse_score,
        )
        return intervention_type
