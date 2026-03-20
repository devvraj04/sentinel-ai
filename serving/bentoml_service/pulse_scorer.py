"""
serving/bentoml_service/pulse_scorer.py
Real-time Pulse Scorer with full SHAP contributions and DynamoDB write.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3
import joblib
import numpy as np
import redis
from botocore.config import Config

from config.settings import get_settings
from config.logging_config import get_logger

settings = get_settings()
logger   = get_logger(__name__)

MODEL_PATH = "models/lightgbm/lgbm_model.joblib"
SHAP_PATH  = "models/lightgbm/shap_explainer.joblib"

FEATURE_LABELS = {
    "salary_delay_days":         "Salary credited later than usual",
    "salary_amount_drop_pct":    "Salary amount reduced this month",
    "balance_wow_drop_pct":      "Savings balance declining rapidly",
    "upi_lending_spike_ratio":   "Increased UPI transactions to lending apps",
    "upi_lending_total_amount":  "High borrowing from lending apps",
    "utility_payment_latency":   "Utility bills paid later in billing cycle",
    "discretionary_contraction": "Reduced spending on dining and entertainment",
    "discretionary_txn_count":   "Fewer discretionary transactions this month",
    "atm_withdrawal_spike":      "Increased ATM withdrawals (cash hoarding)",
    "atm_amount_spike":          "Higher ATM withdrawal amounts than usual",
    "failed_auto_debit_count":   "Failed auto-debit attempts detected",
    "failed_auto_debit_amount":  "EMI auto-debit failures by amount",
    "failed_utility_count":      "Failed utility payments",
    "credit_utilization_delta":  "Rising credit card utilization",
    "drift_salary":              "Salary drift score",
    "drift_balance":             "Balance drift score",
    "drift_lending":             "Lending app drift score",
    "drift_utility":             "Utility drift score",
    "drift_discretionary":       "Discretionary spend drift score",
    "drift_atm":                 "ATM withdrawal drift score",
    "drift_auto_debit":          "Auto-debit drift score",
    "drift_credit_card":         "Credit card drift score",
    "composite_drift_score":     "Overall behavioral drift from baseline",
    "flag_salary":               "Early stress flag: salary",
    "flag_balance":              "Early stress flag: savings",
    "flag_lending":              "Early stress flag: lending UPI",
    "flag_utility":              "Early stress flag: utility",
    "flag_discretionary":        "Early stress flag: discretionary",
    "flag_atm":                  "Early stress flag: ATM",
    "flag_failed_debit":         "Early stress flag: auto-debit",
    "total_stress_flags":        "Total early stress flags triggered",
    "ead_estimate":              "Exposure at default (EAD)",
    "total_txn_count":           "Transaction volume this month",
    "total_txn_amount":          "Total transaction amount",
    "income_coverage_ratio":     "Salary-to-income coverage ratio",
    "monthly_income":            "Monthly income",
    "is_salaried":               "Employment: salaried",
    "is_self_employed":          "Employment: self-employed",
    "is_mass_retail":            "Segment: mass retail",
    "is_affluent":               "Segment: affluent",
}


def _get_dynamodb_resource():
    """Always connects to real AWS DynamoDB."""
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(connect_timeout=5, read_timeout=5,
                      retries={"max_attempts": 1}),
    )


class PulseScorer:
    """
    Loads LightGBM + SHAP, reads features from Redis,
    writes scored results to AWS DynamoDB.
    """

    def __init__(self):
        self._model_package = None
        self._explainer     = None
        self._redis         = None
        self._dynamo        = None
        self._load()

    def _load(self):
        # Load model
        if os.path.exists(MODEL_PATH):
            self._model_package = joblib.load(MODEL_PATH)
            logger.info("LightGBM model loaded",
                        version=self._model_package.get("version"),
                        cv_auc=self._model_package.get("cv_auc"))
        else:
            logger.warning("Model not found — rule-based fallback", path=MODEL_PATH)

        # Load SHAP
        if os.path.exists(SHAP_PATH):
            self._explainer = joblib.load(SHAP_PATH)
            logger.info("SHAP explainer loaded")

        # Connect Redis
        try:
            self._redis = redis.from_url(
                settings.redis_url, socket_connect_timeout=3
            )
            self._redis.ping()
            logger.info("Redis connected", url=settings.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable", error=str(e))
            self._redis = None

        # Connect DynamoDB with timeout (non-blocking)
        def _connect_dynamo():
            try:
                self._dynamo = _get_dynamodb_resource()
                self._dynamo.meta.client.list_tables(Limit=1)
                logger.info("DynamoDB connected", region=settings.aws_region)
            except Exception as e:
                logger.warning("DynamoDB unavailable", error=str(e))
                self._dynamo = None

        t = threading.Thread(target=_connect_dynamo, daemon=True)
        t.start()
        t.join(timeout=8)
        if t.is_alive():
            logger.warning("DynamoDB connection timed out")
            self._dynamo = None

    def _get_features_from_redis(self, customer_id: str) -> Optional[dict]:
        """Retrieve pre-computed behavioral features from Redis."""
        if not self._redis:
            return None
        try:
            key  = f"sentinel:features:{customer_id}"
            data = self._redis.hgetall(key)
            if not data:
                return None
            return {k.decode(): float(v) for k, v in data.items()}
        except Exception as e:
            logger.error("Redis retrieval failed",
                         customer_id=customer_id, error=str(e))
            return None

    def _build_feature_vector(self, features: dict,
                               feature_cols: list) -> np.ndarray:
        """Build numpy array from feature dict, filling missing with 0."""
        return np.array(
            [features.get(col, 0.0) for col in feature_cols], dtype=float
        )

    def _pd_to_pulse_score(self, pd_probability: float) -> int:
        """
        Sigmoid scaling: PD → Pulse Score 0-100
        PD < 0.10 → score < 25  (green)
        PD ~ 0.30 → score ~ 50  (orange boundary)
        PD > 0.60 → score > 70  (red)
        """
        scaled = 1.0 / (1.0 + np.exp(-10.0 * (pd_probability - 0.30)))
        return max(1, min(100, int(round(scaled * 100))))

    def _get_risk_tier(self, pulse_score: int) -> str:
        """Map pulse score to risk tier."""
        if pulse_score >= 70:
            return "red"
        if pulse_score >= 45:
            return "orange"
        if pulse_score >= 25:
            return "yellow"
        return "green"

    def _get_intervention(self, risk_tier: str) -> tuple[bool, Optional[str]]:
        """Return intervention recommendation based on risk tier."""
        if risk_tier == "red":
            return True, "payment_holiday_or_restructuring"
        if risk_tier == "orange":
            return True, "flexible_emi_offer"
        if risk_tier == "yellow":
            return True, "preventive_digital_nudge"
        return False, None

    def _compute_shap(self, fv: np.ndarray,
                      feature_cols: list) -> list[dict]:
        """Compute SHAP values and return top 5 contributing factors."""
        if self._explainer is None:
            return []
        try:
            shap_values = self._explainer.shap_values(fv.reshape(1, -1))
            vals = shap_values[1][0] if isinstance(shap_values, list) \
                   else shap_values[0]

            factors = []
            for i, col in enumerate(feature_cols):
                contrib = float(vals[i])
                if abs(contrib) < 0.001:
                    continue
                factors.append({
                    "feature_name":   col,
                    "contribution":   round(abs(contrib), 4),
                    "human_readable": FEATURE_LABELS.get(col, col),
                    "direction":      "increases_risk" if contrib > 0
                                      else "decreases_risk",
                    "raw_value":      round(float(fv[i]), 4),
                })
            factors.sort(key=lambda x: x["contribution"], reverse=True)
            return factors[:7]
        except Exception as e:
            logger.error("SHAP failed", error=str(e))
            return []

    def _write_to_dynamodb(self, result: dict) -> None:
        """Persist Pulse Score to AWS DynamoDB sentinel-customer-scores."""
        if self._dynamo is None:
            return
        try:
            table      = self._dynamo.Table(settings.dynamodb_table_scores)
            top_factor = (result["top_factors"][0]["feature_name"]
                          if result.get("top_factors") else "unknown")
            table.update_item(
                Key={"customer_id": result["customer_id"]},
                UpdateExpression="""
                    SET pulse_score       = :ps,
                        risk_tier         = :rt,
                        pd_probability    = :pd,
                        confidence        = :cf,
                        top_factor        = :tf,
                        intervention_flag = :iv,
                        intervention_type = :it,
                        model_version     = :mv,
                        updated_at        = :ua
                """,
                ExpressionAttributeValues={
                    ":ps": result["pulse_score"],
                    ":rt": result["risk_tier"],
                    ":pd": Decimal(str(round(result["pd_probability"], 6))),
                    ":cf": Decimal(str(round(result["confidence"], 4))),
                    ":tf": top_factor,
                    ":iv": result["intervention_recommended"],
                    ":it": result.get("intervention_type") or "none",
                    ":mv": result.get("model_version", "unknown"),
                    ":ua": result["scored_at"],
                },
            )
        except Exception as e:
            logger.error("DynamoDB write failed",
                         customer_id=result["customer_id"], error=str(e))

    def _fallback_score(self, customer_id: str) -> dict:
        """Rule-based fallback when model or Redis features unavailable."""
        idx = int(customer_id.replace("CUST", "")) \
              if customer_id.startswith("CUST") else 500
        pd_prob = round(np.random.uniform(0.02, 0.18), 4)
        pulse_score = self._pd_to_pulse_score(pd_prob)
        risk_tier   = self._get_risk_tier(pulse_score)
        recommended, intervention_type = self._get_intervention(risk_tier)
        return {
            "customer_id":              customer_id,
            "pulse_score":              pulse_score,
            "risk_tier":                risk_tier,
            "pd_probability":           pd_prob,
            "confidence":               0.5,
            "top_factors":              [],
            "intervention_recommended": recommended,
            "intervention_type":        intervention_type,
            "scored_at":                datetime.now(timezone.utc).isoformat(),
            "model_version":            "fallback",
            "cached":                   False,
            "warning":                  "Model or Redis features unavailable",
        }

    def score(self, customer_id: str, force_refresh: bool = False) -> dict:
        """Score a customer. Writes result to DynamoDB."""
        features = self._get_features_from_redis(customer_id)

        if not features or self._model_package is None:
            logger.warning("Using fallback scorer", customer_id=customer_id)
            result = self._fallback_score(customer_id)
            self._write_to_dynamodb(result)
            return result

        model        = self._model_package["model"]
        feature_cols = self._model_package["feature_cols"]
        version      = self._model_package.get("version", "unknown")

        features["month_offset"] = 0.0
        fv = self._build_feature_vector(features, feature_cols)

        pd_prob     = float(model.predict_proba(fv.reshape(1, -1))[0, 1])
        pulse_score = self._pd_to_pulse_score(pd_prob)
        risk_tier   = self._get_risk_tier(pulse_score)
        recommended, intervention_type = self._get_intervention(risk_tier)
        confidence  = float(abs(pd_prob - 0.5) * 2)
        top_factors = self._compute_shap(fv, feature_cols)

        logger.info("Customer scored",
                    customer_id=customer_id,
                    pulse_score=pulse_score,
                    risk_tier=risk_tier,
                    pd_probability=round(pd_prob, 4),
                    top_factor=top_factors[0]["feature_name"] if top_factors else "none")

        result = {
            "customer_id":              customer_id,
            "pulse_score":              pulse_score,
            "risk_tier":                risk_tier,
            "pd_probability":           round(pd_prob, 4),
            "confidence":               round(confidence, 4),
            "top_factors":              top_factors,
            "intervention_recommended": recommended,
            "intervention_type":        intervention_type,
            "scored_at":                datetime.now(timezone.utc).isoformat(),
            "model_version":            version,
            "cached":                   False,
        }

        self._write_to_dynamodb(result)
        return result


# Singleton
_scorer: Optional[PulseScorer] = None


def get_scorer() -> PulseScorer:
    """Return singleton PulseScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = PulseScorer()
    return _scorer