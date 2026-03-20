"""
scripts/simulate_transactions.py
──────────────────────────────────────────────────────────────────────────────
Sentinel Transaction Simulator — infinite random-user mode.

DESIGN:
  - 100 customers, each starting with pulse_score = 0.
  - Runs an infinite loop: each iteration picks a RANDOM customer (uniform,
    no bias) and generates ONE transaction for them.
  - Every transaction is published to Kafka → consumed by feature_pipeline
    → features written to Redis → LightGBM scores → DynamoDB + PostgreSQL.
  - Score is updated by the model after EVERY transaction (positive, negative,
    or zero delta — all are shown).
  - Press Ctrl+C at any time to stop; a ranked summary is printed.

DATE CONFIGURATION:
  Real-time mode always uses datetime.now(timezone.utc) — no capping.
  To change the historical cutoff (used only for timestamp labels):
  Edit the line marked ← EDIT HERE.

HOW TO RUN:
  # Terminal 1 — Kafka + infra
  docker-compose up -d

  # Terminal 2 — Feature pipeline consumer (keep running)
  python -m ingestion.consumers.feature_pipeline

  # Terminal 3 — This simulator
  python -m scripts.simulate_transactions
  python -m scripts.simulate_transactions --customers 100 --delay 200
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
import random
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from botocore.config import Config
from faker import Faker

from config.settings import get_settings
from ingestion.producers.transaction_producer import TransactionProducer
from ingestion.schemas.transaction_event import (
    AccountType, MerchantCategory, PaymentStatus,
    TransactionEvent, TransactionType,
)
from serving.bentoml_service.scoring_utils import (
    pd_to_pulse_score, pulse_score_to_tier, get_intervention,
)

fake     = Faker("en_IN")
settings = get_settings()

IST = timezone(timedelta(hours=5, minutes=30))
CUTOFF_LABEL = datetime(2026, 3, 20, 23, 59, 59, tzinfo=IST)  # ← EDIT HERE (display only)

GEOGRAPHIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
               "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
SEGMENTS    = ["mass_retail", "mass_retail", "mass_retail", "affluent", "hni"]
CHANNELS    = ["mobile_app", "net_banking", "branch", "atm"]


# ── Connections ────────────────────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(settings.database_url)

def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(connect_timeout=10, read_timeout=10,
                      retries={"max_attempts": 2}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk(profile: dict) -> tuple[str, float]:
    income  = profile["monthly_income"]
    emi     = profile["emi_amount"]
    savings = profile["avg_savings_balance"]
    emp     = profile["employment_status"]
    shock   = profile["has_life_shock"]

    dti     = min(emi / max(income, 1), 1.0)
    sav_r   = min(savings / max(income * 3, 1), 1.0)
    emp_r   = {"salaried": 0.15, "self_employed": 0.45, "gig": 0.70}.get(emp, 0.35)
    shock_r = 0.55 if shock else 0.0
    raw = dti*0.35 + (1-sav_r)*0.25 + emp_r*0.20 + shock_r*0.20
    raw = max(0.0, min(1.0, raw + random.gauss(0, 0.07)))
    if raw >= 0.60: return "high_risk", raw
    if raw >= 0.35: return "at_risk",   raw
    return "healthy", raw


def make_customer(idx: int) -> dict[str, Any]:
    ir = {"low":(12_000,35_000), "mid":(35_000,90_000),
          "high":(90_000,200_000), "very_high":(200_000,500_000)}
    bracket = random.choices(list(ir), weights=[35,40,18,7])[0]
    income  = random.randint(*ir[bracket])
    emp     = random.choices(["salaried","self_employed","gig"], weights=[60,30,10])[0]
    emi_r   = random.choices([0.10,0.25,0.40,0.55,0.65], weights=[20,30,25,15,10])[0]
    emi     = income * emi_r * random.uniform(0.9, 1.1)
    sav_m   = random.choices([0.3,1.0,2.5,5.0,10.0], weights=[15,30,30,18,7])[0]
    savings = income * sav_m * random.uniform(0.8, 1.2)

    profile = {
        "customer_id":         f"CUST{idx:05d}",
        "full_name":           fake.name(),
        "email":               fake.email(),
        "phone":               fake.phone_number()[:20],
        "monthly_income":      income,
        "salary_day":          random.randint(1, 7),
        "emi_amount":          emi,
        "emi_due_day":         random.randint(5, 12),
        "avg_savings_balance": savings,
        "credit_limit":        income * random.uniform(1.5, 5.0),
        "segment":             random.choice(SEGMENTS),
        "geography":           random.choice(GEOGRAPHIES),
        "employment_status":   emp,
        "preferred_channel":   random.choice(CHANNELS),
        "product_mix":         random.choices(["loan_only","card_only","both"], weights=[25,20,55])[0],
        "has_life_shock":      random.random() < 0.15,
        "salary_irregularity": random.uniform(0,1) if emp != "salaried" else random.uniform(0,0.2),
    }
    profile["risk_level"], profile["stress_base"] = compute_risk(profile)
    return profile


def build_customers(n: int = 100) -> list[dict[str, Any]]:
    random.seed(42)
    customers = [make_customer(i) for i in range(1, n+1)]
    random.seed()
    counts = {"healthy":0, "at_risk":0, "high_risk":0}
    for c in customers: counts[c["risk_level"]] += 1
    print(f"  Customers: Healthy={counts['healthy']:,}  "
          f"At-Risk={counts['at_risk']:,}  High-Risk={counts['high_risk']:,}")
    return customers


# ══════════════════════════════════════════════════════════════════════════════
# FULL 51-FEATURE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(customer: dict, df: pd.DataFrame,
                         reference_date: datetime) -> dict[str, float]:
    income     = float(customer["monthly_income"])
    emi        = float(customer["emi_amount"])
    credit_lim = float(customer["credit_limit"])
    emp        = customer["employment_status"]
    segment    = customer["segment"]
    fv: dict[str, float] = {}

    now          = reference_date
    salary_start = now - timedelta(days=35)
    short_start  = now - timedelta(days=14)
    hist_start   = now - timedelta(days=90)

    if not df.empty:
        df = df.copy()
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)

    df_sal   = df[df["txn_timestamp"] >= salary_start] if not df.empty else pd.DataFrame()
    df_short = df[df["txn_timestamp"] >= short_start]  if not df.empty else pd.DataFrame()
    df_hist  = df[(df["txn_timestamp"] >= hist_start) &
                  (df["txn_timestamp"] < short_start)] if not df.empty else pd.DataFrame()
    df_90    = df[df["txn_timestamp"] >= hist_start]   if not df.empty else pd.DataFrame()

    # Salary
    salaries = df_sal[df_sal["txn_type"]=="salary_credit"] if not df_sal.empty else pd.DataFrame()
    if not salaries.empty:
        last = salaries.sort_values("txn_timestamp").iloc[-1]
        delay = max(0.0, float(last["txn_timestamp"].day - customer.get("salary_day", 3)))
        sal_drop = max(0.0, (income - salaries["amount"].mean()) / max(income, 1))
    else:
        delay = 5.0; sal_drop = 0.1
    fv["salary_delay_days"]      = delay
    fv["salary_amount_drop_pct"] = sal_drop

    # Balance flow
    if not df_short.empty and "account_type" in df_short.columns:
        sav = df_short[df_short["account_type"]=="savings"]
        if not sav.empty:
            credit_t = {"salary_credit","upi_credit","neft_rtgs","reversal"}
            credits  = sav[sav["txn_type"].isin(credit_t)]["amount"].sum()
            debits   = sav[~sav["txn_type"].isin(credit_t)]["amount"].sum()
            total    = credits + debits
            wow_drop = float(np.clip((debits-credits)/max(total, income*0.1), -1, 1))
        else:
            wow_drop = 0.0
    else:
        wow_drop = 0.0
    fv["balance_wow_drop_pct"]  = wow_drop
    fv["savings_runway_months"] = min(customer["avg_savings_balance"]/max(income,1), 24.0)

    # Lending UPI
    def csum(fr, col, val):
        if fr.empty or col not in fr.columns: return 0.0
        return float(fr[fr[col]==val]["amount"].sum())
    lend_s = csum(df_short, "is_lending_app_upi", True)
    lend_h = csum(df_hist,  "is_lending_app_upi", True)
    h_avg  = lend_h / (76/14) if lend_h > 0 else 0.0
    if lend_s == 0.0:
        lend_r = 1.0
    elif h_avg > 0:
        lend_r = float(np.clip(lend_s/h_avg, 1.0, 10.0))
    else:
        lend_r = float(np.clip(1.0 + lend_s/max(income*0.05,1), 1.0, 10.0))
    fv["upi_lending_spike_ratio"]  = lend_r
    fv["upi_lending_total_amount"] = float(lend_s)

    # Utility latency
    utils = df_short[df_short["txn_type"]=="utility_payment"] if not df_short.empty else pd.DataFrame()
    fv["utility_payment_latency"] = float(utils["txn_timestamp"].apply(lambda t: t.day).mean()) if not utils.empty else 5.0

    # Discretionary
    disc_c = {"dining","entertainment","travel","shopping"}
    disc_s = df_short[df_short["merchant_category"].isin(disc_c)]["amount"].sum() if not df_short.empty else 0.0
    disc_h = df_hist[df_hist["merchant_category"].isin(disc_c)]["amount"].sum()   if not df_hist.empty else 0.0
    disc_n = int(len(df_short[df_short["merchant_category"].isin(disc_c)]))        if not df_short.empty else 0
    if disc_h > 0:
        disc_r = float(np.clip(disc_s / max(disc_h/(76/14), 1), 0, 5))
    else:
        disc_r = float(np.clip(disc_s / max(income*0.05, 100), 0, 5))
    fv["discretionary_contraction"] = disc_r
    fv["discretionary_txn_count"]   = float(disc_n)

    # ATM
    atm_s = df_short[df_short["txn_type"]=="atm_withdrawal"]["amount"].sum() if not df_short.empty else 0.0
    atm_h = df_hist[df_hist["txn_type"]=="atm_withdrawal"]["amount"].sum()   if not df_hist.empty else 0.0
    if atm_h > 0:
        atm_sp = float(np.clip(atm_s/max(atm_h/(76/14),1), 1.0, 10.0))
    else:
        atm_sp = float(np.clip(1.0 + atm_s/max(income*0.05,2000), 1.0, 10.0))
    fv["atm_withdrawal_spike"] = atm_sp
    fv["atm_amount_spike"]     = float(atm_s)

    # Failed debits
    fd14 = df_short[(df_short["txn_type"]=="auto_debit")&(df_short["payment_status"]=="failed")] if not df_short.empty else pd.DataFrame()
    fd90 = df_90[(df_90["txn_type"]=="auto_debit")&(df_90["payment_status"]=="failed")]           if not df_90.empty   else pd.DataFrame()
    fu14 = df_short[(df_short["txn_type"]=="utility_payment")&(df_short["payment_status"]=="failed")] if not df_short.empty else pd.DataFrame()
    fv["failed_auto_debit_count"]  = float(len(fd14))
    fv["failed_auto_debit_amount"] = float(fd14["amount"].sum()) if not fd14.empty else 0.0
    fv["failed_utility_count"]     = float(len(fu14))
    fv["missed_emi_streak"]        = float(min(len(fd90), 3))
    fv["dpd_30_last_12m"]          = float(min(len(fd90)*30, 90))

    # Credit card
    cc_s = df_short[df_short["account_type"]=="credit_card"]["amount"].sum() if not df_short.empty else 0.0
    cc_h = df_hist[df_hist["account_type"]=="credit_card"]["amount"].sum()   if not df_hist.empty else 0.0
    cc_a = cc_h/(76/14) if cc_h > 0 else 0.0
    fv["credit_utilization_delta"] = float(np.clip((cc_s-cc_a)/max(credit_lim,1), -1, 1))
    fv["revolving_utilization"]    = float(np.clip(cc_s/max(credit_lim/14*30,1), 0, 1))

    fv["emi_to_income_ratio"]  = float(np.clip(emi/max(income,1), 0, 1))
    fv["total_txn_count"]      = float(len(df_short)) if not df_short.empty else 0.0
    fv["total_txn_amount"]     = float(df_short["amount"].sum()) if not df_short.empty else 0.0
    fv["income_coverage_ratio"]= float(np.clip(salaries["amount"].sum()/max(income,1),0,2)) if not salaries.empty else 0.5
    fv["monthly_income"]       = income
    fv["ead_estimate"]         = float(emi * 24)

    # Flags
    fv["flag_salary"]          = 1.0 if delay > 7 else 0.0
    fv["flag_balance"]         = 1.0 if wow_drop > 0.3 else 0.0
    fv["flag_lending"]         = 1.0 if lend_s > income*0.15 else 0.0
    fv["flag_utility"]         = 1.0 if fv["utility_payment_latency"] > 22 else 0.0
    fv["flag_discretionary"]   = 1.0 if disc_r < 0.3 else 0.0
    fv["flag_atm"]             = 1.0 if atm_sp > 3.0 else 0.0
    fv["flag_failed_debit"]    = 1.0 if len(fd14) > 0 else 0.0
    fv["flag_emi_burden"]      = 1.0 if emi/max(income,1) > 0.55 else 0.0
    fv["flag_high_utilization"]= 1.0 if fv["revolving_utilization"] > 0.85 else 0.0
    fv["total_stress_flags"]   = sum([fv[f"flag_{k}"] for k in
        ["salary","balance","lending","utility","discretionary","atm",
         "failed_debit","emi_burden","high_utilization"]])

    # Drift
    def drift(cur, norm, sens=0.3):
        return float(np.clip((cur-norm)/max(abs(norm)*sens,0.1), -5, 5))
    fv["drift_salary"]        = drift(delay, 2.0)
    fv["drift_balance"]       = drift(wow_drop, 0.0)
    fv["drift_lending"]       = drift(lend_r, 1.0)
    fv["drift_utility"]       = drift(fv["utility_payment_latency"], 8.0)
    fv["drift_discretionary"] = drift(disc_r, 1.0)
    fv["drift_atm"]           = drift(atm_sp, 1.0)
    fv["drift_auto_debit"]    = drift(float(len(fd14)), 0.0, 1.0)
    fv["drift_credit_card"]   = drift(fv["credit_utilization_delta"], 0.0)
    fv["composite_drift_score"] = float(np.mean([abs(fv[f"drift_{k}"])
        for k in ["salary","balance","lending","atm","auto_debit"]]))

    fv["p2p_transfer_spike"]        = 1.0
    fv["investment_redemption_pct"] = 0.0
    fv["credit_enquiries_3m"]       = 0.0
    fv["tenure_months"]             = 24.0
    fv["is_salaried"]      = 1.0 if emp == "salaried"       else 0.0
    fv["is_self_employed"] = 1.0 if emp == "self_employed"  else 0.0
    fv["is_mass_retail"]   = 1.0 if segment == "mass_retail"else 0.0
    fv["is_affluent"]      = 1.0 if segment in ("affluent","hni") else 0.0
    return fv


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — load once, score on demand
# ══════════════════════════════════════════════════════════════════════════════

import joblib as _jl
_PKG: Optional[dict] = None

def _model():
    global _PKG
    if _PKG is None:
        try:
            _PKG = _jl.load("models/lightgbm/lgbm_model.joblib")
            if not hasattr(_model, "_shap"):
                try:
                    _model._shap = _jl.load("models/lightgbm/shap_explainer.joblib")
                except Exception:
                    _model._shap = None
            print(f"  Model v{_PKG.get('version')}  AUC={_PKG.get('cv_auc',0):.4f}")
        except Exception as e:
            print(f"  WARNING: model not loaded — {e}")
    return _PKG


def score_customer(customer: dict, db_conn, dynamo_db,
                   ref: Optional[datetime] = None) -> Optional[dict]:
    pkg = _model()
    if pkg is None:
        return None

    model        = pkg["model"]
    feature_cols = pkg["feature_cols"]
    version      = pkg.get("version", "unknown")
    ref          = ref or datetime.now(timezone.utc)
    cid          = customer["customer_id"]

    # Read full transaction history from PostgreSQL
    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT txn_type, amount, merchant_category, payment_status,
                       account_type, txn_timestamp
                FROM transactions
                WHERE customer_id = %s
                ORDER BY txn_timestamp ASC
            """, (cid,))
            rows = cur.fetchall()
    except Exception:
        return None

    if rows:
        df = pd.DataFrame(rows, columns=[
            "txn_type","amount","merchant_category","payment_status",
            "account_type","txn_timestamp"])
        df["is_lending_app_upi"]   = df["merchant_category"] == "lending_app"
        df["is_auto_debit_failed"] = ((df["txn_type"]=="auto_debit") &
                                      (df["payment_status"]=="failed"))
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df = pd.DataFrame()

    fv_dict = build_feature_vector(customer, df, ref)
    fv      = np.array([fv_dict.get(col, 0.0) for col in feature_cols], dtype=float)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd_prob = float(model.predict_proba(fv.reshape(1,-1))[0,1])

    pulse_score = pd_to_pulse_score(pd_prob)
    risk_tier   = pulse_score_to_tier(pulse_score)
    recommended, iv_type = get_intervention(risk_tier)
    confidence  = float(abs(pd_prob - 0.5) * 2)

    # SHAP
    top_factors = []
    if _model._shap:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = _model._shap.shap_values(fv.reshape(1,-1))
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
            factors = [{"feature_name":col,"contribution":round(abs(float(vals[i])),4),
                        "human_readable":col.replace("_"," ").title(),
                        "direction":"increases_risk" if vals[i]>0 else "decreases_risk",
                        "raw_value":round(float(fv[i]),4)}
                       for i,col in enumerate(feature_cols) if abs(vals[i])>0.001]
            top_factors = sorted(factors, key=lambda x: x["contribution"], reverse=True)[:7]
        except Exception:
            pass

    scored_at = ref.isoformat()
    result = {
        "customer_id":              cid,
        "pulse_score":              pulse_score,
        "risk_tier":                risk_tier,
        "pd_probability":           round(pd_prob, 6),
        "confidence":               round(confidence, 4),
        "top_factors":              top_factors,
        "intervention_recommended": recommended,
        "intervention_type":        iv_type or "none",
        "scored_at":                scored_at,
        "model_version":            version,
        "cached":                   False,
    }

    _write_dynamodb(result, customer, dynamo_db)
    _write_history(result, db_conn)
    return result


def _write_dynamodb(result: dict, customer: dict, dynamo_db) -> None:
    if dynamo_db is None: return
    try:
        top = result.get("top_factors", [])
        dynamo_db.Table(settings.dynamodb_table_scores).put_item(Item={
            "customer_id":         result["customer_id"],
            "full_name":           customer.get("full_name",""),
            "segment":             customer.get("segment",""),
            "geography":           customer.get("geography",""),
            "employment_status":   customer.get("employment_status",""),
            "preferred_channel":   customer.get("preferred_channel",""),
            "monthly_income":      Decimal(str(customer.get("monthly_income",0))),
            "credit_limit":        Decimal(str(round(customer.get("credit_limit",0),2))),
            "outstanding_balance": Decimal(str(round(customer.get("emi_amount",0)*24,2))),
            "credit_utilization":  Decimal(str(round(float(result.get("pd_probability",0))*0.8,4))),
            "days_past_due":       0,
            "pulse_score":         result["pulse_score"],
            "risk_tier":           result["risk_tier"],
            "pd_probability":      Decimal(str(result["pd_probability"])),
            "confidence":          Decimal(str(result["confidence"])),
            "top_factor":          top[0]["feature_name"] if top else "unknown",
            "intervention_flag":   result["intervention_recommended"],
            "intervention_type":   result["intervention_type"],
            "model_version":       result["model_version"],
            "updated_at":          result["scored_at"],
        })
    except Exception:
        pass


def _write_history(result: dict, db_conn) -> None:
    if db_conn is None: return
    try:
        import json as _json
        tf = result.get("top_factors", [])
        shap_json = _json.dumps([{k:v for k,v in f.items() if k!="raw_value"}
                                  for f in tf]) if tf else None
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pulse_score_history (
                    customer_id, pulse_score, risk_tier, pd_probability,
                    confidence, top_factor_1, top_factor_2, top_factor_3,
                    shap_values, model_version,
                    intervention_flag, intervention_type, scored_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                result["customer_id"], result["pulse_score"], result["risk_tier"],
                result["pd_probability"], result["confidence"],
                tf[0]["feature_name"] if len(tf)>0 else None,
                tf[1]["feature_name"] if len(tf)>1 else None,
                tf[2]["feature_name"] if len(tf)>2 else None,
                shap_json, result["model_version"],
                result["intervention_recommended"],
                result["intervention_type"], result["scored_at"],
            ))
        db_conn.commit()
    except Exception:
        try: db_conn.rollback()
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRES HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_customer_postgres(customer: dict, conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO customers
                (customer_id, full_name, email, phone, segment,
                 geography, employment_status, monthly_income)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (customer_id) DO NOTHING
        """, (customer["customer_id"], customer["full_name"],
              customer["email"], customer["phone"], customer["segment"],
              customer["geography"], customer["employment_status"],
              customer["monthly_income"]))
    conn.commit()


def insert_transaction(evt: TransactionEvent, conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transactions
                (customer_id, account_id, txn_type, amount,
                 merchant_category, payment_channel, payment_status, txn_timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            evt.customer_id,
            getattr(evt, "account_id", None),
            evt.txn_type.value if hasattr(evt.txn_type,"value") else str(evt.txn_type),
            float(evt.amount),
            evt.merchant_category.value if hasattr(evt.merchant_category,"value") else str(evt.merchant_category),
            evt.payment_channel,
            evt.payment_status.value if hasattr(evt.payment_status,"value") else str(evt.payment_status),
            evt.txn_timestamp,
        ))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION TYPES — weighted by customer stress level
# ══════════════════════════════════════════════════════════════════════════════

def next_transaction(customer: dict, seq: int, now: datetime) -> TransactionEvent:
    """
    Generate the next transaction for a customer.
    seq = global transaction number (0-based) — used only for variety, not order.
    """
    cid    = customer["customer_id"]
    income = customer["monthly_income"]
    emi    = customer["emi_amount"]
    stress = customer["stress_base"]
    credit = customer["credit_limit"]

    weights = {
        "salary_credit":     8,
        "utility_payment":   10,
        "auto_debit":        8,
        "upi_dining":        max(2, int(15 * (1 - stress*0.7))),
        "upi_groceries":     max(3, int(12 * (1 - stress*0.4))),
        "upi_shopping":      max(2, int(10 * (1 - stress*0.6))),
        "atm_small":         max(3, int(8  * (1 - stress*0.3))),
        "atm_large":         max(1, int(10 * stress)),
        "savings_drain":     max(1, int(12 * stress)),
        "lending_app":       max(1, int(15 * stress)),
        "utility_fail":      max(1, int(8  * stress)),
        "credit_card_spend": 8,
    }

    kind = random.choices(list(weights), weights=list(weights.values()))[0]
    fail_prob = min(0.9, customer.get("stress_base", 0) * 0.8 +
                    (0.2 if customer.get("has_life_shock") else 0))

    if kind == "salary_credit":
        delay = random.randint(0, max(0, int(8*stress))) if stress > 0.2 else 0
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SALARY_CREDIT,
            amount=round(income * random.uniform(0.88, 1.03) * (1-0.12*stress)),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT",
            txn_timestamp=now - timedelta(days=delay),
        )
    elif kind == "utility_payment":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 5000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        )
    elif kind == "auto_debit":
        failed = random.random() < fail_prob
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi),
            payment_channel="ECS",
            payment_status=PaymentStatus.FAILED if failed else PaymentStatus.SUCCESS,
            txn_timestamp=now,
            is_auto_debit_failed=failed,
        )
    elif kind == "upi_dining":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(80, income*0.04), 2),
            merchant_category=MerchantCategory.DINING,
            payment_channel="UPI", txn_timestamp=now,
        )
    elif kind == "upi_groceries":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(200, income*0.05), 2),
            merchant_category=MerchantCategory.GROCERIES,
            payment_channel="UPI", txn_timestamp=now,
        )
    elif kind == "upi_shopping":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(500, income*0.08), 2),
            merchant_category=MerchantCategory.SHOPPING,
            payment_channel="UPI", txn_timestamp=now,
        )
    elif kind == "atm_small":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=random.choice([500, 1000, 2000]),
            payment_channel="ATM", txn_timestamp=now,
        )
    elif kind == "atm_large":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=random.choice([5000, 10000]),
            payment_channel="ATM", txn_timestamp=now,
        )
    elif kind == "savings_drain":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(1000, customer["avg_savings_balance"] *
                             random.uniform(0.05, 0.25) * stress), 2),
            payment_channel="NEFT", txn_timestamp=now,
        )
    elif kind == "lending_app":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(2000, min(30000, income*0.6)), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI",
            txn_timestamp=now,
            is_lending_app_upi=True,
        )
    elif kind == "utility_fail":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 3000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.FAILED,
            txn_timestamp=now,
        )
    else:  # credit_card_spend
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.CREDIT_CARD,
            txn_type=TransactionType.CREDIT_CARD_PAYMENT,
            amount=round(random.uniform(500, credit*0.12), 2),
            merchant_category=random.choice([
                MerchantCategory.GROCERIES, MerchantCategory.DINING,
                MerchantCategory.SHOPPING]),
            payment_channel="POS", txn_timestamp=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STRESS SIGNAL DETECTION (for console display)
# ══════════════════════════════════════════════════════════════════════════════

def is_stress(evt: TransactionEvent) -> tuple[bool, str]:
    tt  = evt.txn_type.value if hasattr(evt.txn_type,"value") else str(evt.txn_type)
    st  = evt.payment_status.value if hasattr(evt.payment_status,"value") else str(evt.payment_status)
    cat = evt.merchant_category.value if hasattr(evt.merchant_category,"value") else str(evt.merchant_category)
    amt = float(evt.amount)

    if getattr(evt,"is_auto_debit_failed",False) or (tt=="auto_debit" and st=="failed"):
        return True, "FAILED EMI"
    if getattr(evt,"is_lending_app_upi",False) or cat=="lending_app":
        return True, "LENDING APP"
    if tt=="utility_payment" and st=="failed":
        return True, "FAILED UTILITY"
    if tt=="atm_withdrawal" and amt >= 5000:
        return True, "LARGE ATM"
    if tt=="savings_withdrawal" and amt >= 10000:
        return True, "SAVINGS DRAIN"
    return False, ""


def _print_summary(customers: list, current_scores: dict,
                   txn_counts: dict, total_txns: int) -> None:
    """Print a ranked summary table when the loop exits."""
    print(f"\n\n{'='*72}")
    print("SENTINEL — Session Summary  (Ctrl+C detected)")
    print(f"  Total transactions processed : {total_txns:,}")
    print(f"  Customers active             : {sum(1 for v in txn_counts.values() if v > 0)}")
    print(f"{'='*72}")
    print(f"  {'Rank':<5} {'Customer ID':<14} {'Txns':>5} {'Score':>6} "
          f"{'Tier':<10} {'Risk':<12}  {'Name'}")
    print(f"  {'─'*70}")

    cust_map = {c["customer_id"]: c for c in customers}
    ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (cid, score) in enumerate(ranked[:20], 1):
        c    = cust_map.get(cid, {})
        tier = pulse_score_to_tier(score)
        mark = {"green":"●","yellow":"◆","orange":"▲","red":"■"}.get(tier, "?")
        print(f"  {rank:<5} {cid:<14} {txn_counts[cid]:>5} {score:>6} "
              f"{mark} {tier:<8} {c.get('risk_level','?'):<12}  {c.get('full_name','')}")

    tier_counts = {"green":0, "yellow":0, "orange":0, "red":0}
    for s in current_scores.values():
        tier_counts[pulse_score_to_tier(s)] += 1
    print(f"\n  Tier distribution: "
          f"green={tier_counts['green']}  yellow={tier_counts['yellow']}  "
          f"orange={tier_counts['orange']}  red={tier_counts['red']}")

    # Bias check — min/max txn counts
    counts = list(txn_counts.values())
    if counts:
        print(f"\n  Transaction spread  min={min(counts)}  "
              f"max={max(counts)}  avg={sum(counts)/len(counts):.1f}")
    print(f"{'='*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — infinite random-user loop, score every transaction
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    n_customers: int = 100,
    delay_ms: float  = 0,       # ms between transactions (0 = as fast as possible)
    db_conn          = None,
    dynamo_db        = None,
) -> None:
    """
    Infinite loop:
      1. Build n_customers profiles (once).
      2. Register all customers in PostgreSQL (once).
      3. Loop forever:
         a. Pick a UNIFORMLY random customer — no bias.
         b. Generate ONE transaction for them using the current timestamp.
         c. Publish to Kafka, insert into PostgreSQL.
         d. Score via LightGBM — EVERY transaction, no skipping.
         e. Print live delta (positive / negative / zero).
      4. Ctrl+C → print summary and exit cleanly.
    """
    # ── Build customers ──────────────────────────────────────────────────────
    random.seed(42)
    customers = [make_customer(i) for i in range(1, n_customers + 1)]
    random.seed()   # unseed — all randomness from here is truly random
    cust_ids  = [c["customer_id"] for c in customers]
    cust_map  = {c["customer_id"]: c for c in customers}

    # ── Live state ───────────────────────────────────────────────────────────
    current_scores: dict[str, int] = {cid: 0 for cid in cust_ids}
    txn_counts:     dict[str, int] = {cid: 0 for cid in cust_ids}
    total_txns = 0

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"SENTINEL — Infinite Random Transaction Pipeline")
    print(f"  Customers  : {n_customers}")
    print(f"  User pick  : uniform random — no bias")
    print(f"  Scoring    : every transaction (model-only authority)")
    print(f"  Kafka      : {settings.kafka_bootstrap_servers}")
    print(f"  Scoring    : LightGBM v{_model().get('version','?') if _model() else '?'}")
    print(f"  Stop       : Ctrl+C")
    print(f"{'='*72}")
    print(f"  {'#':>7}  {'Customer':<12} {'Type':<22} {'Amount':>10}  "
          f"{'Signal':<16} {'Score':>6}  {'Δ':>5}  Tier")
    print(f"  {'─'*78}")

    # ── Register all customers in PostgreSQL upfront ─────────────────────────
    print("  Registering customers in PostgreSQL...", end="", flush=True)
    for c in customers:
        ensure_customer_postgres(c, db_conn)
    print(f" ✓ ({n_customers} customers ready)\n  {'─'*78}")

    # ── Infinite loop ─────────────────────────────────────────────────────────
    with TransactionProducer() as producer:
        try:
            while True:
                # ① Pick a uniformly random customer — pure random.choice,
                #    no weighting, no round-robin → zero systematic bias
                customer = random.choice(customers)
                cid      = customer["customer_id"]
                txn_ts   = datetime.now(timezone.utc)

                # ② Generate transaction
                evt = next_transaction(customer, txn_counts[cid], txn_ts)

                # ③ Publish to Kafka
                producer.publish(evt)

                # ④ Insert into PostgreSQL
                insert_transaction(evt, db_conn)

                txn_counts[cid] += 1
                total_txns       += 1

                # ⑤ Score via LightGBM — every single transaction
                result    = score_customer(customer, db_conn, dynamo_db, ref=txn_ts)
                new_score = result["pulse_score"] if result else current_scores[cid]
                delta     = new_score - current_scores[cid]
                tier      = pulse_score_to_tier(new_score)
                current_scores[cid] = new_score

                # ⑥ Console output
                stress_flag, stress_label = is_stress(evt)
                tier_mark  = {"green":"●","yellow":"◆","orange":"▲","red":"■"}.get(tier, "?")
                delta_str  = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "  0")
                signal_str = f"[{stress_label}]" if stress_flag else ""
                txn_label  = (
                    evt.txn_type.value if hasattr(evt.txn_type, "value") else str(evt.txn_type)
                ).replace("_", " ").upper()[:22]

                print(f"  {total_txns:>7}  {cid:<12} {txn_label:<22} "
                      f"₹{float(evt.amount):>9,.0f}  "
                      f"{signal_str:<16} "
                      f"{new_score:>6}  "
                      f"{delta_str:>5}  "
                      f"{tier_mark} {tier}")

                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

        except KeyboardInterrupt:
            _print_summary(customers, current_scores, txn_counts, total_txns)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Sentinel — Infinite Random Transaction Simulator"
    )
    parser.add_argument("--customers", type=int,   default=100,
                        help="Number of customers (default 100)")
    parser.add_argument("--delay",     type=float, default=0,
                        help="Delay between transactions in ms (default 0)")
    args = parser.parse_args()

    print("="*72)
    print("SENTINEL — Infinite Random Transaction Simulator")
    print(f"  Customers : {args.customers}")
    print(f"  Delay     : {args.delay} ms")
    print(f"  Mode      : Random user, score every txn, run until Ctrl+C")
    print("="*72 + "\n")

    print("Connecting to PostgreSQL...", end="", flush=True)
    conn = get_db_conn()
    print(" ✓")

    print("Connecting to DynamoDB...", end="", flush=True)
    try:
        dynamo = get_dynamodb()
        dynamo.meta.client.list_tables(Limit=1)
        print(f" ✓ ({settings.aws_region})")
    except Exception as e:
        print(f" ✗ ({e})")
        dynamo = None

    _model()  # pre-load model before loop starts

    try:
        run_pipeline(
            n_customers = args.customers,
            delay_ms    = args.delay,
            db_conn     = conn,
            dynamo_db   = dynamo,
        )
    finally:
        conn.close()
        print("Connection closed. Done.")