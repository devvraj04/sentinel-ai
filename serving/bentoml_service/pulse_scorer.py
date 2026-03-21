"""
serving/bentoml_service/pulse_scorer.py
──────────────────────────────────────────────────────────────────────────────
Real-time Pulse Scorer.

Two modes — selected automatically based on environment:
  LOCAL MODE:     SAGEMAKER_ENDPOINT_NAME not set → uses local lgbm_model.joblib
  SAGEMAKER MODE: SAGEMAKER_ENDPOINT_NAME is set  → calls AWS SageMaker endpoint

Both modes:
  - Read behavioral features from Redis
  - Write Pulse Score to AWS DynamoDB
  - Return full SHAP factor contributions
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3
import joblib
import numpy as np
import psycopg2
from psycopg2 import pool as _pgpool
import redis
from botocore.config import Config

from config.settings import get_settings
from config.logging_config import get_logger
from serving.bentoml_service.scoring_utils import (
    pd_to_pulse_score, pulse_score_to_tier, get_intervention, set_sigmoid_center
)

settings = get_settings()
logger   = get_logger(__name__)

_pool = _pgpool.SimpleConnectionPool(1, 10, settings.database_url)

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
    "credit_enquiries_3m":       "Credit enquiries (approximate proxy via failed auto-debits)",
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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_dynamodb_resource():
    """Always connects to real AWS DynamoDB — never localhost."""
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.dynamodb_endpoint,
        config=Config(connect_timeout=5, read_timeout=5,
                      retries={"max_attempts": 1}),
    )








def _connect_redis_safe() -> Optional[redis.Redis]:
    try:
        r = redis.from_url(settings.redis_url, socket_connect_timeout=3)
        r.ping()
        logger.info("Redis connected", url=settings.redis_url)
        return r
    except Exception as e:
        logger.warning("Redis unavailable", error=str(e))
        return None


def _connect_dynamo_safe() -> Optional[object]:
    result = {"dynamo": None}

    def _try():
        try:
            db = _get_dynamodb_resource()
            db.meta.client.list_tables(Limit=1)
            result["dynamo"] = db
            logger.info("DynamoDB connected", region=settings.aws_region)
        except Exception as e:
            logger.warning("DynamoDB unavailable", error=str(e))

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=8)
    if t.is_alive():
        logger.warning("DynamoDB connection timed out")
    return result["dynamo"]


def _get_features_from_redis(
    r: Optional[redis.Redis], customer_id: str
) -> Optional[dict]:
    if not r:
        return None
    try:
        key  = f"sentinel:features:{customer_id}"
        data = r.hgetall(key)
        if not data:
            return None
        features = {}
        for k, v in data.items():
            key_str = k.decode()
            if key_str == "computed_at":   # timestamp string — skip, not a model feature
                continue
            try:
                features[key_str] = float(v)
            except (ValueError, TypeError):
                continue   # skip any other non-numeric field silently
        return features if features else None
    except Exception as e:
        logger.error("Redis retrieval failed", customer_id=customer_id, error=str(e))
        return None


def _write_score_to_dynamodb(dynamo, result: dict) -> None:
    """Persist Pulse Score to AWS DynamoDB sentinel-customer-scores table."""
    if dynamo is None:
        return
    try:
        table      = dynamo.Table(settings.dynamodb_table_scores)
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


def _write_score_to_postgres(result: dict) -> None:
    """
    Insert one row into pulse_score_history (PostgreSQL) so the score
    timeline is permanently recorded with a timestamp.

    Called immediately after _write_score_to_dynamodb() using the SAME
    result dict, guaranteeing both stores carry the exact same score,
    tier, PD, and factors at the exact same scored_at timestamp.

    Every call produces a NEW history row. DynamoDB keeps the current
    score (latest wins); PostgreSQL keeps every score ever computed.
    """
    conn = None
    try:
        import json as _json
        top_factors = result.get("top_factors", [])
        f1 = top_factors[0]["feature_name"] if len(top_factors) > 0 else None
        f2 = top_factors[1]["feature_name"] if len(top_factors) > 1 else None
        f3 = top_factors[2]["feature_name"] if len(top_factors) > 2 else None
        shap_json = _json.dumps([
            {k: v for k, v in f.items() if k != "raw_value"}
            for f in top_factors
        ]) if top_factors else None

        conn = _pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pulse_score_history (
                    customer_id, pulse_score, risk_tier, pd_probability,
                    confidence, top_factor_1, top_factor_2, top_factor_3,
                    shap_values, model_version,
                    intervention_flag, intervention_type,
                    scored_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s
                )
            """, (
                result["customer_id"],
                result["pulse_score"],
                result["risk_tier"],
                round(result["pd_probability"], 6),
                round(result.get("confidence", 0.0), 4),
                f1, f2, f3,
                shap_json,
                result.get("model_version", "unknown"),
                result.get("intervention_recommended", False),
                result.get("intervention_type") or "none",
                result["scored_at"],
            ))
        conn.commit()
    except Exception as e:
        logger.error("PostgreSQL pulse_score_history write failed",
                     customer_id=result.get("customer_id"), error=str(e))
    finally:
        if conn:
            _pool.putconn(conn)


def _write_score_to_both(dynamo, result: dict) -> None:
    """
    Single call-site that writes to DynamoDB (current score) AND
    PostgreSQL (history row) using the same result dict.
    Both stores always carry the exact same values.
    """
    _write_score_to_dynamodb(dynamo, result)
    _write_score_to_postgres(result)


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL SCORER — uses lgbm_model.joblib + shap_explainer.joblib on disk
# ══════════════════════════════════════════════════════════════════════════════
class PulseScorer:
    """
    Local model scorer.
    Loads LightGBM + SHAP from disk.
    Used when SAGEMAKER_ENDPOINT_NAME is NOT set.
    """

    def __init__(self):
        self._model_package = None
        self._explainer     = None
        self._redis         = None
        self._dynamo        = None
        self._load()

    def _load(self):
        # Model
        if os.path.exists(MODEL_PATH):
            self._model_package = joblib.load(MODEL_PATH)
            if "threshold" in self._model_package:
                set_sigmoid_center(self._model_package["threshold"])
            logger.info("LightGBM model loaded (local)",
                        version=self._model_package.get("version"),
                        cv_auc=self._model_package.get("cv_auc"))
        else:
            logger.warning("Model not found — rule-based fallback", path=MODEL_PATH)

        # SHAP
        if os.path.exists(SHAP_PATH):
            self._explainer = joblib.load(SHAP_PATH)
            logger.info("SHAP explainer loaded")

        # Redis
        self._redis = _connect_redis_safe()

        # DynamoDB
        self._dynamo = _connect_dynamo_safe()

    def _compute_shap(self, fv: np.ndarray, feature_cols: list) -> list[dict]:
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

    def _last_known_score(self, customer_id: str) -> dict | None:
        """
        Read the most recent real score for this customer from PostgreSQL
        pulse_score_history, then DynamoDB as a secondary fallback.
        Returns None only if the customer has never been scored at all.
        Never generates a random or hardcoded score.
        """
        # Primary: PostgreSQL pulse_score_history (full details)
        conn = None
        try:
            conn = _pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pulse_score, risk_tier, pd_probability,
                           confidence, top_factor_1, model_version, scored_at
                    FROM pulse_score_history
                    WHERE customer_id = %s
                    ORDER BY scored_at DESC
                    LIMIT 1
                """, (customer_id,))
                row = cur.fetchone()
            if row:
                score, tier, pd, conf, top_f, mv, scored_at = row
                recommended, iv_type = get_intervention(tier)
                return {
                    "customer_id":              customer_id,
                    "pulse_score":              int(score),
                    "risk_tier":                tier,
                    "pd_probability":           float(pd),
                    "confidence":               float(conf or 0),
                    "top_factors":              [{"feature_name": top_f,
                                                  "contribution": 0,
                                                  "human_readable": top_f or "",
                                                  "direction": "increases_risk"}]
                                               if top_f else [],
                    "intervention_recommended": recommended,
                    "intervention_type":        iv_type,
                    "scored_at":                scored_at.isoformat() if scored_at else datetime.now(timezone.utc).isoformat(),
                    "model_version":            mv or "cached",
                    "cached":                   True,
                    "warning":                  "Served from score history — Redis features unavailable",
                }
        except Exception as e:
            logger.warning("PostgreSQL fallback read failed", customer_id=customer_id, error=str(e))
        finally:
            if conn:
                _pool.putconn(conn)

        # Secondary: DynamoDB current score record
        try:
            if self._dynamo:
                table = self._dynamo.Table(settings.dynamodb_table_scores)
                resp  = table.get_item(Key={"customer_id": customer_id})
                item  = resp.get("Item")
                if item and item.get("pulse_score"):
                    score = int(item["pulse_score"])
                    tier  = item.get("risk_tier", pulse_score_to_tier(score))
                    pd    = float(item.get("pd_probability", 0))
                    recommended, iv_type = get_intervention(tier)
                    return {
                        "customer_id":              customer_id,
                        "pulse_score":              score,
                        "risk_tier":                tier,
                        "pd_probability":           pd,
                        "confidence":               float(item.get("confidence", 0)),
                        "top_factors":              [],
                        "intervention_recommended": recommended,
                        "intervention_type":        iv_type,
                        "scored_at":                item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                        "model_version":            item.get("model_version", "cached"),
                        "cached":                   True,
                        "warning":                  "Served from DynamoDB cache — Redis features unavailable",
                    }
        except Exception as e:
            logger.warning("DynamoDB fallback read failed", customer_id=customer_id, error=str(e))

        return None  # customer has never been scored

    def score(self, customer_id: str, force_refresh: bool = False) -> dict:
        features = _get_features_from_redis(self._redis, customer_id)

        if not features or self._model_package is None:
            cached = self._last_known_score(customer_id)
            if cached:
                logger.info("Serving cached score — Redis features unavailable",
                            customer_id=customer_id,
                            pulse_score=cached["pulse_score"],
                            source="history")
                return cached   # do NOT write to history — score has not changed
            # Truly no data at all — log and return error signal without writing
            logger.warning("No score available — customer not yet scored",
                           customer_id=customer_id)
            return {
                "customer_id":   customer_id,
                "pulse_score":   None,
                "risk_tier":     None,
                "pd_probability":None,
                "confidence":    None,
                "top_factors":   [],
                "intervention_recommended": False,
                "intervention_type":        None,
                "scored_at":     datetime.now(timezone.utc).isoformat(),
                "model_version": "unscored",
                "cached":        False,
                "warning":       "Customer has not been scored yet. Run historical load first.",
            }

        model        = self._model_package["model"]
        feature_cols = self._model_package["feature_cols"]
        version      = self._model_package.get("version", "unknown")

        # No month_offset hack — features are used as-is from Redis
        fv = np.array([features.get(col, 0.0) for col in feature_cols], dtype=float)

        pd_prob     = float(model.predict_proba(fv.reshape(1, -1))[0, 1])
        pulse_score = pd_to_pulse_score(pd_prob)
        risk_tier   = pulse_score_to_tier(pulse_score)
        recommended, intervention_type = get_intervention(risk_tier)
        top_factors = self._compute_shap(fv, feature_cols)

        # SHAP-based confidence: total signal strength
        if top_factors:
            total_shap = sum(f["contribution"] for f in top_factors)
            confidence = float(min(total_shap / max(len(feature_cols) * 0.1, 1), 1.0))
        else:
            confidence = float(abs(pd_prob - 0.5) * 2)  # fallback

        logger.info("Customer scored (local)",
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

        _write_score_to_both(self._dynamo, result)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SAGEMAKER SCORER — calls AWS SageMaker endpoint
# Used when SAGEMAKER_ENDPOINT_NAME is set in .env
# ══════════════════════════════════════════════════════════════════════════════
class SageMakerScorer:
    """
    Calls the deployed SageMaker endpoint for inference.
    Drop-in replacement for PulseScorer.
    The model runs on AWS — not on your laptop.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint = endpoint_name
        self._redis    = None
        self._dynamo   = None

        # SageMaker runtime client
        self._sm_client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(connect_timeout=10, read_timeout=30,
                          retries={"max_attempts": 2}),
        )

        self._redis  = _connect_redis_safe()
        self._dynamo = _connect_dynamo_safe()

        logger.info("SageMaker scorer ready",
                    endpoint=endpoint_name,
                    region=settings.aws_region)

    def _last_known_score(self, customer_id: str) -> dict | None:
        """
        Read the most recent real score from PostgreSQL/DynamoDB.
        Same logic as PulseScorer — no random, no hardcoded values.
        """
        try:
            conn = psycopg2.connect(settings.database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pulse_score, risk_tier, pd_probability,
                           confidence, top_factor_1, model_version, scored_at
                    FROM pulse_score_history
                    WHERE customer_id = %s
                    ORDER BY scored_at DESC LIMIT 1
                """, (customer_id,))
                row = cur.fetchone()
            conn.close()
            if row:
                score, tier, pd, conf, top_f, mv, scored_at = row
                recommended, iv_type = get_intervention(tier)
                return {
                    "customer_id": customer_id, "pulse_score": int(score),
                    "risk_tier": tier, "pd_probability": float(pd),
                    "confidence": float(conf or 0), "top_factors": [],
                    "intervention_recommended": recommended,
                    "intervention_type": iv_type,
                    "scored_at": scored_at.isoformat() if scored_at else datetime.now(timezone.utc).isoformat(),
                    "model_version": mv or "cached", "cached": True,
                    "warning": "SageMaker unavailable — serving cached score",
                }
        except Exception as e:
            logger.warning("PostgreSQL fallback read failed", customer_id=customer_id, error=str(e))
        try:
            if self._dynamo:
                table = self._dynamo.Table(settings.dynamodb_table_scores)
                item = table.get_item(Key={"customer_id": customer_id}).get("Item")
                if item and item.get("pulse_score"):
                    score = int(item["pulse_score"])
                    tier  = item.get("risk_tier", pulse_score_to_tier(score))
                    recommended, iv_type = get_intervention(tier)
                    return {
                        "customer_id": customer_id, "pulse_score": score,
                        "risk_tier": tier, "pd_probability": float(item.get("pd_probability", 0)),
                        "confidence": float(item.get("confidence", 0)), "top_factors": [],
                        "intervention_recommended": recommended, "intervention_type": iv_type,
                        "scored_at": item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                        "model_version": item.get("model_version", "cached"), "cached": True,
                        "warning": "SageMaker unavailable — serving DynamoDB cache",
                    }
        except Exception as e:
            logger.warning("DynamoDB fallback read failed", customer_id=customer_id, error=str(e))
        return None

    def score(self, customer_id: str, force_refresh: bool = False) -> dict:
        # Get features from Redis — no synthetic features injected
        features = _get_features_from_redis(self._redis, customer_id) or {}

        try:
            # Call SageMaker endpoint
            payload  = json.dumps({"customer_id": customer_id, "features": features})
            response = self._sm_client.invoke_endpoint(
                EndpointName=self._endpoint,
                ContentType="application/json",
                Body=payload.encode("utf-8"),
            )
            result = json.loads(response["Body"].read().decode("utf-8"))

            # Enrich result with human-readable labels if not already present
            for factor in result.get("top_factors", []):
                if "human_readable" not in factor:
                    factor["human_readable"] = FEATURE_LABELS.get(
                        factor.get("feature_name", ""), factor.get("feature_name", "")
                    )

            result["cached"] = False
            result["scored_at"] = datetime.now(timezone.utc).isoformat()

            # Add intervention recommendation if not returned by endpoint
            if "intervention_recommended" not in result:
                tier = result.get("risk_tier", "green")
                recommended, intervention_type = get_intervention(tier)
                result["intervention_recommended"] = recommended
                result["intervention_type"]        = intervention_type

            logger.info("Customer scored (SageMaker)",
                        customer_id=customer_id,
                        pulse_score=result.get("pulse_score"),
                        risk_tier=result.get("risk_tier"),
                        endpoint=self._endpoint)

        except Exception as e:
            logger.error("SageMaker endpoint call failed",
                         customer_id=customer_id,
                         endpoint=self._endpoint,
                         error=str(e))
            cached = self._last_known_score(customer_id)
            if cached:
                logger.info("Serving cached score — SageMaker unavailable",
                            customer_id=customer_id, pulse_score=cached["pulse_score"])
                return cached  # do NOT write — score unchanged
            return {
                "customer_id": customer_id, "pulse_score": None, "risk_tier": None,
                "pd_probability": None, "confidence": None, "top_factors": [],
                "intervention_recommended": False, "intervention_type": None,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "unscored", "cached": False,
                "warning": "Customer has not been scored yet.",
            }

        _write_score_to_both(self._dynamo, result)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON — auto-selects local vs SageMaker based on .env
# ══════════════════════════════════════════════════════════════════════════════
_scorer: Optional[PulseScorer | SageMakerScorer] = None


def get_scorer() -> PulseScorer | SageMakerScorer:
    """
    Returns singleton scorer instance.

    Selection logic:
      If SAGEMAKER_ENDPOINT_NAME is set in .env → SageMakerScorer
      Otherwise                                 → PulseScorer (local)

    To switch to SageMaker after deploying the endpoint, just add to .env:
      SAGEMAKER_ENDPOINT_NAME=sentinel-pulse-scorer
    """
    global _scorer
    if _scorer is None:
        endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "").strip()
        if endpoint_name:
            logger.info("Using SageMaker scorer", endpoint=endpoint_name)
            _scorer = SageMakerScorer(endpoint_name)
        else:
            logger.info("Using local scorer", model_path=MODEL_PATH)
            _scorer = PulseScorer()
    return _scorer