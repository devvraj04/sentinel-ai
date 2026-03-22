"""
scripts/simulate_transactions.py
──────────────────────────────────────────────────────────────────────────────
Sentinel Transaction Simulator — infinite random-user mode.

SCORING SEMANTICS:
  Pulse Score = PD probability mapped through a sigmoid.
  HIGHER score = MORE likely to default = WORSE for the customer.
  LOWER  score = LESS likely to default = BETTER for the customer.

  Transaction → feature impact → score direction:
    ✓ Salary on time         → income_coverage↑, salary_delay↓      → score ↓
    ✓ EMI succeeds           → failed_debit_count=0, streak=0        → score ↓
    ✓ CC bill paid           → cc utilization untouched              → score ↓
    ✓ Utility paid on time   → utility_latency low                   → score ↓
    ✓ Savings growing        → savings_runway_months↑ (live balance) → score ↓
    ✗ EMI fails              → missed_emi_streak↑, flag_failed=1     → score ↑
    ✗ Lending app UPI        → upi_lending_spike_ratio↑              → score ↑↑
    ✗ Large ATM              → atm_withdrawal_spike↑                 → score ↑
    ✗ Savings drain          → balance_wow_drop↑                     → score ↑
    ✗ Utility fails          → failed_utility_count↑                 → score ↑

SUSTAINED GOOD BEHAVIOR — SCORE STABILIZATION:
  Each customer carries a live `good_streak` counter (resets on any stress event).
  The streak drives two mechanisms:

  1. WEIGHT BOOST (transaction generation):
     streak  0–9  : normal healthy/stress mix (~80/20)
     streak 10–19 : healthy weights × 1.25 → fewer stress events generated
     streak 20–34 : healthy weights × 1.50 → even more dominated by healthy txns
     streak ≥ 35  : healthy weights × 1.75 → near-complete healthy dominance

  2. LIVE BALANCE (feature vector accuracy):
     `estimated_balance` is tracked per customer and updated after every
     transaction (salary credit adds, every spend subtracts). This feeds
     directly into `savings_runway_months` in the feature vector so the
     model sees the customer's ACTUAL growing savings — not a frozen initial
     value. A customer consistently spending less than they earn will have
     a rising balance → rising runway → falling PD → falling score.

CREDIT CARD BILL PAYMENT:
  CC SPEND   → account_type=CREDIT_CARD → counted in cc_s → cc utilization ↑
  CC PAYMENT → account_type=SAVINGS, txn_type=TRANSFER_OUT → NOT in cc_s
               → paying the bill does NOT raise utilization → score ↓

HOW TO RUN:
  python -m scripts.simulate_transactions                  # 100 customers, fast
  python -m scripts.simulate_transactions --delay 300      # 300 ms between txns
  python -m scripts.simulate_transactions --customers 50   # fewer customers
  Press Ctrl+C to stop and print a ranked summary.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
import numpy as np
import pandas as pd
import psycopg2
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

GEOGRAPHIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
               "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
SEGMENTS    = ["mass_retail", "mass_retail", "mass_retail", "affluent", "hni"]
CHANNELS    = ["mobile_app", "net_banking", "branch", "atm"]


# ══════════════════════════════════════════════════════════════════════════════
# PER-CUSTOMER LIVE STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CustomerState:
    """
    Mutable live state tracked per customer throughout the simulation.

    estimated_balance:
        Running savings balance updated after every transaction.
        Salary credits add; every spend/payment subtracts.
        Fed into build_feature_vector() so savings_runway_months reflects
        the customer's ACTUAL current balance — not the frozen initial value.

    good_streak:
        Consecutive healthy transactions since the last stress event.
        Drives _build_weights() to generate fewer and fewer stress events
        the longer a customer stays on a positive trajectory.
        Resets to 0 on ANY stress transaction (failed EMI, lending app, etc.).

    total_healthy / total_stress:
        Cumulative counters — used in the summary table.

    salary_on_time_streak:
        Consecutive salary credits that arrived without delay.
        Not currently used in weights but logged in the summary.
    """
    estimated_balance:    float
    good_streak:          int   = 0
    total_healthy:        int   = 0
    total_stress:         int   = 0
    salary_on_time_streak: int  = 0


def make_state(customer: dict) -> CustomerState:
    """Initialise state from the customer profile."""
    return CustomerState(
        estimated_balance=customer["avg_savings_balance"]
    )


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
        "product_mix":         random.choices(["loan_only","card_only","both"],
                                              weights=[25,20,55])[0],
        "has_life_shock":      random.random() < 0.15,
        "salary_irregularity": (random.uniform(0, 1)
                                if emp != "salaried"
                                else random.uniform(0, 0.2)),
    }
    profile["risk_level"], profile["stress_base"] = compute_risk(profile)
    return profile


def build_customers(n: int = 100) -> list[dict[str, Any]]:
    random.seed(42)
    customers = [make_customer(i) for i in range(1, n + 1)]
    random.seed()
    counts = {"healthy": 0, "at_risk": 0, "high_risk": 0}
    for c in customers:
        counts[c["risk_level"]] += 1
    print(f"  Customers: Healthy={counts['healthy']:,}  "
          f"At-Risk={counts['at_risk']:,}  High-Risk={counts['high_risk']:,}")
    return customers


# ══════════════════════════════════════════════════════════════════════════════
# FULL 51-FEATURE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(customer: dict, df: pd.DataFrame,
                         reference_date: datetime,
                         estimated_balance: Optional[float] = None) -> dict[str, float]:
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

    # ── Salary signals ────────────────────────────────────────────────────
    salaries = (df_sal[df_sal["txn_type"] == "salary_credit"]
                if not df_sal.empty else pd.DataFrame())
    if not salaries.empty:
        last     = salaries.sort_values("txn_timestamp").iloc[-1]
        delay    = max(0.0, float(last["txn_timestamp"].day
                                  - customer.get("salary_day", 3)))
        sal_drop = max(0.0, (income - salaries["amount"].mean()) / max(income, 1))
    else:
        delay = 5.0; sal_drop = 0.1
    fv["salary_delay_days"]      = delay
    fv["salary_amount_drop_pct"] = sal_drop

    # ── Balance flow (savings account only) ───────────────────────────────
    # credit_t = transaction types that represent INFLOWS to savings.
    # TRANSFER_OUT is deliberately excluded — credit card bill payments
    # use TRANSFER_OUT and must count as outflows (correct behaviour).
    if not df_short.empty and "account_type" in df_short.columns:
        sav = df_short[df_short["account_type"] == "savings"]
        if not sav.empty:
            credit_t = {"salary_credit", "upi_credit", "neft_rtgs", "reversal"}
            credits  = sav[sav["txn_type"].isin(credit_t)]["amount"].sum()
            debits   = sav[~sav["txn_type"].isin(credit_t)]["amount"].sum()
            total    = credits + debits
            wow_drop = float(np.clip(
                (debits - credits) / max(total, income * 0.1), -1, 1))
        else:
            wow_drop = 0.0
    else:
        wow_drop = 0.0
    fv["balance_wow_drop_pct"]  = wow_drop
    # Use the live estimated_balance if provided (tracks real savings growth/drain
    # over the course of the simulation).  Falls back to the initial profile value
    # only when called outside the simulator (e.g. unit tests).
    live_balance = estimated_balance if estimated_balance is not None \
                   else customer["avg_savings_balance"]
    fv["savings_runway_months"] = min(live_balance / max(income, 1), 24.0)

    # ── Lending-app UPI ── inferred from merchant_category, NOT pre-label ──
    def csum_cat(fr, cat_val):
        if fr.empty or "merchant_category" not in fr.columns:
            return 0.0
        return float(fr[fr["merchant_category"] == cat_val]["amount"].sum())

    lend_s = csum_cat(df_short, "lending_app")
    lend_h = csum_cat(df_hist,  "lending_app")
    h_avg  = lend_h / (76 / 14) if lend_h > 0 else 0.0
    if lend_s == 0.0:
        lend_r = 1.0
    elif h_avg > 0:
        lend_r = float(np.clip(lend_s / h_avg, 1.0, 10.0))
    else:
        lend_r = float(np.clip(1.0 + lend_s / max(income * 0.05, 1), 1.0, 10.0))
    fv["upi_lending_spike_ratio"]  = lend_r
    fv["upi_lending_total_amount"] = float(lend_s)

    # ── Utility latency ───────────────────────────────────────────────────
    utils = (df_short[df_short["txn_type"] == "utility_payment"]
             if not df_short.empty else pd.DataFrame())
    fv["utility_payment_latency"] = (
        float(utils["txn_timestamp"].apply(lambda t: t.day).mean())
        if not utils.empty else 5.0
    )

    # ── Discretionary spend ───────────────────────────────────────────────
    disc_c = {"dining", "entertainment", "travel", "shopping"}
    disc_s = (df_short[df_short["merchant_category"].isin(disc_c)]["amount"].sum()
              if not df_short.empty else 0.0)
    disc_h = (df_hist[df_hist["merchant_category"].isin(disc_c)]["amount"].sum()
              if not df_hist.empty else 0.0)
    disc_n = (int(len(df_short[df_short["merchant_category"].isin(disc_c)]))
              if not df_short.empty else 0)
    if disc_h > 0:
        disc_r = float(np.clip(disc_s / max(disc_h / (76 / 14), 1), 0, 5))
    else:
        disc_r = float(np.clip(disc_s / max(income * 0.05, 100), 0, 5))
    fv["discretionary_contraction"] = disc_r
    fv["discretionary_txn_count"]   = float(disc_n)

    # ── ATM ───────────────────────────────────────────────────────────────
    atm_s = (df_short[df_short["txn_type"] == "atm_withdrawal"]["amount"].sum()
             if not df_short.empty else 0.0)
    atm_h = (df_hist[df_hist["txn_type"] == "atm_withdrawal"]["amount"].sum()
             if not df_hist.empty else 0.0)
    if atm_h > 0:
        atm_sp = float(np.clip(atm_s / max(atm_h / (76 / 14), 1), 1.0, 10.0))
    else:
        atm_sp = float(np.clip(1.0 + atm_s / max(income * 0.05, 2000), 1.0, 10.0))
    fv["atm_withdrawal_spike"] = atm_sp
    fv["atm_amount_spike"]     = float(atm_s)

    # ── Failed debits ─────────────────────────────────────────────────────
    fd14 = (df_short[(df_short["txn_type"] == "auto_debit") &
                     (df_short["payment_status"] == "failed")]
            if not df_short.empty else pd.DataFrame())
    fd90 = (df_90[(df_90["txn_type"] == "auto_debit") &
                  (df_90["payment_status"] == "failed")]
            if not df_90.empty else pd.DataFrame())
    fu14 = (df_short[(df_short["txn_type"] == "utility_payment") &
                     (df_short["payment_status"] == "failed")]
            if not df_short.empty else pd.DataFrame())
    fv["failed_auto_debit_count"]  = float(len(fd14))
    fv["failed_auto_debit_amount"] = float(fd14["amount"].sum()) if not fd14.empty else 0.0
    fv["failed_utility_count"]     = float(len(fu14))
    fv["missed_emi_streak"]        = float(min(len(fd90), 3))
    fv["dpd_30_last_12m"]          = float(min(len(fd90) * 30, 90))

    # ── Credit card utilization ───────────────────────────────────────────
    # Only transactions with account_type=CREDIT_CARD count as spend.
    # Credit card bill payments use account_type=SAVINGS + txn_type=TRANSFER_OUT
    # so they are invisible here — paying the bill does NOT raise utilization.
    cc_s = (df_short[df_short["account_type"] == "credit_card"]["amount"].sum()
            if not df_short.empty else 0.0)
    cc_h = (df_hist[df_hist["account_type"] == "credit_card"]["amount"].sum()
            if not df_hist.empty else 0.0)
    cc_a = cc_h / (76 / 14) if cc_h > 0 else 0.0
    fv["credit_utilization_delta"] = float(
        np.clip((cc_s - cc_a) / max(credit_lim, 1), -1, 1))
    fv["revolving_utilization"] = float(
        np.clip(cc_s / max(credit_lim / 14 * 30, 1), 0, 1))

    # ── Ratios & totals ───────────────────────────────────────────────────
    fv["emi_to_income_ratio"]   = float(np.clip(emi / max(income, 1), 0, 1))
    fv["total_txn_count"]       = float(len(df_short)) if not df_short.empty else 0.0
    fv["total_txn_amount"]      = float(df_short["amount"].sum()) if not df_short.empty else 0.0
    fv["income_coverage_ratio"] = (
        float(np.clip(salaries["amount"].sum() / max(income, 1), 0, 2))
        if not salaries.empty else 0.5
    )
    fv["monthly_income"] = income
    fv["ead_estimate"]   = float(emi * 24)

    # ── Binary stress flags ───────────────────────────────────────────────
    fv["flag_salary"]           = 1.0 if delay > 7 else 0.0
    fv["flag_balance"]          = 1.0 if wow_drop > 0.3 else 0.0
    fv["flag_lending"]          = 1.0 if lend_s > income * 0.15 else 0.0
    fv["flag_utility"]          = 1.0 if fv["utility_payment_latency"] > 22 else 0.0
    fv["flag_discretionary"]    = 1.0 if disc_r < 0.3 else 0.0
    fv["flag_atm"]              = 1.0 if atm_sp > 3.0 else 0.0
    fv["flag_failed_debit"]     = 1.0 if len(fd14) > 0 else 0.0
    fv["flag_emi_burden"]       = 1.0 if emi / max(income, 1) > 0.55 else 0.0
    fv["flag_high_utilization"] = 1.0 if fv["revolving_utilization"] > 0.85 else 0.0
    fv["total_stress_flags"]    = sum(fv[f"flag_{k}"] for k in [
        "salary", "balance", "lending", "utility", "discretionary",
        "atm", "failed_debit", "emi_burden", "high_utilization"])

    # ── Drift signals ─────────────────────────────────────────────────────
    def drift(cur, norm, sens=0.3):
        return float(np.clip((cur - norm) / max(abs(norm) * sens, 0.1), -5, 5))

    fv["drift_salary"]          = drift(delay, 2.0)
    fv["drift_balance"]         = drift(wow_drop, 0.0)
    fv["drift_lending"]         = drift(lend_r, 1.0)
    fv["drift_utility"]         = drift(fv["utility_payment_latency"], 8.0)
    fv["drift_discretionary"]   = drift(disc_r, 1.0)
    fv["drift_atm"]             = drift(atm_sp, 1.0)
    fv["drift_auto_debit"]      = drift(float(len(fd14)), 0.0, 1.0)
    fv["drift_credit_card"]     = drift(fv["credit_utilization_delta"], 0.0)
    fv["composite_drift_score"] = float(np.mean([
        abs(fv[f"drift_{k}"])
        for k in ["salary", "balance", "lending", "atm", "auto_debit"]
    ]))

    fv["p2p_transfer_spike"]        = 1.0
    fv["investment_redemption_pct"] = 0.0
    fv["credit_enquiries_3m"]       = 0.0
    fv["tenure_months"]             = 24.0
    fv["is_salaried"]      = 1.0 if emp == "salaried"              else 0.0
    fv["is_self_employed"] = 1.0 if emp == "self_employed"         else 0.0
    fv["is_mass_retail"]   = 1.0 if segment == "mass_retail"       else 0.0
    fv["is_affluent"]      = 1.0 if segment in ("affluent", "hni") else 0.0

    # ── Z-SCORE FEATURES (using population defaults in simulator) ───────
    fv["balance_zscore"]        = float(np.clip(wow_drop / 0.3, -5, 5))  # approximate Z from drop
    fv["salary_delay_zscore"]   = float(np.clip((delay - 3.0) / max(2.0, 0.5), -5, 5))
    fv["atm_spend_zscore"]      = float(np.clip((atm_sp - 1.0) / 0.3, -5, 5)) if atm_sp > 1.0 else 0.0
    fv["lending_spend_zscore"]  = float(np.clip(lend_s / max(income * 0.05, 100.0), -5, 5))
    fv["emi_reliability_score"] = 1.0 if len(fd14) == 0 else max(0.0, 1.0 - len(fd14) * 0.2)

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
            print(f"  Model v{_PKG.get('version')}  AUC={_PKG.get('cv_auc', 0):.4f}")
        except Exception as e:
            print(f"  WARNING: model not loaded — {e}")
    return _PKG


def score_customer(customer: dict, db_conn, dynamo_db,
                   ref: Optional[datetime] = None,
                   estimated_balance: Optional[float] = None) -> Optional[dict]:
    pkg = _model()
    if pkg is None:
        return None

    model        = pkg["model"]
    feature_cols = pkg["feature_cols"]
    version      = pkg.get("version", "unknown")
    ref          = ref or datetime.now(timezone.utc)
    cid          = customer["customer_id"]

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
            "txn_type", "amount", "merchant_category",
            "payment_status", "account_type", "txn_timestamp"])
        # NOTE: is_lending_app_upi and is_auto_debit_failed REMOVED (label leakage)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df = pd.DataFrame()

    fv_dict = build_feature_vector(customer, df, ref,
                                   estimated_balance=estimated_balance)
    fv      = np.array([fv_dict.get(col, 0.0) for col in feature_cols], dtype=float)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd_prob = float(model.predict_proba(fv.reshape(1, -1))[0, 1])

    pulse_score = pd_to_pulse_score(pd_prob)
    risk_tier   = pulse_score_to_tier(pulse_score)
    recommended, iv_type = get_intervention(risk_tier)
    confidence  = float(abs(pd_prob - 0.5) * 2)

    top_factors = []
    if getattr(_model, "_shap", None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv   = _model._shap.shap_values(fv.reshape(1, -1))
                vals = sv[1][0] if isinstance(sv, list) else sv[0]
                factors = [
                    {"feature_name": col,
                     "contribution": round(abs(float(vals[i])), 4),
                     "human_readable": col.replace("_", " ").title(),
                     "direction": "increases_risk" if vals[i] > 0 else "decreases_risk",
                     "raw_value": round(float(fv[i]), 4)}
                    for i, col in enumerate(feature_cols) if abs(vals[i]) > 0.001
                ]
                top_factors = sorted(factors,
                                     key=lambda x: x["contribution"],
                                     reverse=True)[:7]
        except Exception:
            pass

    result = {
        "customer_id":              cid,
        "pulse_score":              pulse_score,
        "risk_tier":                risk_tier,
        "pd_probability":           round(pd_prob, 6),
        "confidence":               round(confidence, 4),
        "top_factors":              top_factors,
        "intervention_recommended": recommended,
        "intervention_type":        iv_type or "none",
        "scored_at":                ref.isoformat(),
        "model_version":            version,
        "cached":                   False,
    }

    _write_dynamodb(result, customer, dynamo_db)
    _write_history(result, db_conn)
    return result


def _write_dynamodb(result: dict, customer: dict, dynamo_db) -> None:
    if dynamo_db is None:
        return
    try:
        top = result.get("top_factors", [])
        dynamo_db.Table(settings.dynamodb_table_scores).put_item(Item={
            "customer_id":         result["customer_id"],
            "full_name":           customer.get("full_name", ""),
            "segment":             customer.get("segment", ""),
            "geography":           customer.get("geography", ""),
            "employment_status":   customer.get("employment_status", ""),
            "preferred_channel":   customer.get("preferred_channel", ""),
            "monthly_income":      Decimal(str(customer.get("monthly_income", 0))),
            "credit_limit":        Decimal(str(round(customer.get("credit_limit", 0), 2))),
            "outstanding_balance": Decimal(str(round(customer.get("emi_amount", 0) * 24, 2))),
            "credit_utilization":  Decimal(str(round(
                float(result.get("pd_probability", 0)) * 0.8, 4))),
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
    if db_conn is None:
        return
    try:
        import json as _json
        tf        = result.get("top_factors", [])
        shap_json = _json.dumps([{k: v for k, v in f.items() if k != "raw_value"}
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
                tf[0]["feature_name"] if len(tf) > 0 else None,
                tf[1]["feature_name"] if len(tf) > 1 else None,
                tf[2]["feature_name"] if len(tf) > 2 else None,
                shap_json, result["model_version"],
                result["intervention_recommended"],
                result["intervention_type"], result["scored_at"],
            ))
        db_conn.commit()
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRES HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_customer_postgres(customer: dict, conn) -> None:
    """
    Upsert customer profile including all financial fields required
    by build_feature_vector() — emi_amount, credit_limit, etc.
    Uses DO UPDATE so re-runs always refresh the values.
    Requires migration 002 to have been applied first.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO customers
                (customer_id, full_name, email, phone, segment,
                 geography, employment_status, monthly_income,
                 emi_amount, credit_limit, avg_savings_balance,
                 tenure_months, expected_salary_day, preferred_channel,
                 product_mix)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (customer_id) DO UPDATE SET
                full_name           = EXCLUDED.full_name,
                monthly_income      = EXCLUDED.monthly_income,
                emi_amount          = EXCLUDED.emi_amount,
                credit_limit        = EXCLUDED.credit_limit,
                avg_savings_balance = EXCLUDED.avg_savings_balance,
                tenure_months       = EXCLUDED.tenure_months,
                expected_salary_day = EXCLUDED.expected_salary_day,
                preferred_channel   = EXCLUDED.preferred_channel,
                product_mix         = EXCLUDED.product_mix,
                updated_at          = NOW()
        """, (
            customer["customer_id"],
            customer["full_name"],
            customer["email"],
            customer["phone"],
            customer["segment"],
            customer["geography"],
            customer["employment_status"],
            float(customer["monthly_income"]),
            float(customer["emi_amount"]),
            float(customer["credit_limit"]),
            float(customer["avg_savings_balance"]),
            int(customer.get("tenure_months", 24)),
            int(customer.get("salary_day", 3)),
            customer.get("preferred_channel", "UPI"),
            customer.get("product_mix", "both"),
        ))
    conn.commit()


def insert_transaction(evt: TransactionEvent, conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transactions
                (customer_id, account_id, txn_type, amount,
                 merchant_category, payment_channel, payment_status,
                 txn_timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            evt.customer_id,
            getattr(evt, "account_id", None),
            evt.txn_type.value      if hasattr(evt.txn_type, "value")           else str(evt.txn_type),
            float(evt.amount),
            evt.merchant_category.value if hasattr(evt.merchant_category, "value") else str(evt.merchant_category),
            evt.payment_channel,
            evt.payment_status.value if hasattr(evt.payment_status, "value")    else str(evt.payment_status),
            evt.txn_timestamp,
        ))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION WEIGHT TABLES
# ══════════════════════════════════════════════════════════════════════════════
#
# Two tables: HEALTHY (score stays flat or falls) and STRESS (score rises).
#
# HEALTHY weights are FIXED — identical for every customer.
# STRESS weights are scaled by stress_base — but the healthy pool is always
# so large that even the most stressed customer produces ≤ ~25% stress events.
# Good streaks further boost the healthy pool via _build_weights().
#
# lending_app is hard-capped at 3 weight units regardless of stress_base or streak.
#
# Score impact reference:
#   salary_credit        → ↓ income_coverage↑, salary_delay↓
#   auto_debit_ok        → ↓ failed_debit_count stays 0
#   utility_ok           → ↓ utility_latency stays low
#   credit_card_bill_pay → ↓ SAVINGS+TRANSFER_OUT → cc utilization untouched
#   upi_groceries        → ≈ essential spend, neutral
#   upi_dining/shopping  → ≈ light discretionary, very minor upward pressure
#   atm_small            → ≈ small cash, neutral
#   credit_card_spend    → ↑ credit_utilization_delta rises (kept low weight)
#   savings_drain        → ↑ balance_wow_drop rises
#   atm_large            → ↑ atm_withdrawal_spike rises
#   utility_fail         → ↑ failed_utility_count rises
#   auto_debit_fail      → ↑↑ missed_emi_streak rises, flag_failed_debit = 1
#   lending_app          → ↑↑ upi_lending_spike_ratio spikes (hardest signal)

_HEALTHY_WEIGHTS: dict[str, int] = {
    "salary_credit":          12,
    "utility_ok":             11,
    "auto_debit_ok":          10,
    "upi_groceries":          10,
    "credit_card_bill_pay":    9,
    "atm_small":               8,
    "upi_dining":              7,
    "upi_shopping":            5,
    "credit_card_spend":       4,
}
# Total healthy base = 76

_STRESS_BASE_WEIGHTS: dict[str, int] = {
    "savings_drain":   3,
    "atm_large":       3,
    "utility_fail":    2,
    "auto_debit_fail": 2,
    "lending_app":     1,   # hard-capped below — never exceeds 3 weight units
}

# Which kinds count as "healthy" for streak tracking
_HEALTHY_KINDS: frozenset[str] = frozenset(_HEALTHY_WEIGHTS.keys())
_STRESS_KINDS:  frozenset[str] = frozenset(_STRESS_BASE_WEIGHTS.keys())


def _build_weights(stress: float,
                   good_streak: int = 0) -> tuple[list[str], list[float]]:
    """
    Merge healthy + scaled stress weights into a single distribution.

    Healthy weights receive a streak multiplier so customers on a sustained
    positive trajectory naturally generate progressively fewer stress events:

      streak  0–9  : multiplier = 1.00  (baseline ~80% healthy)
      streak 10–19 : multiplier = 1.25  (~83% healthy)
      streak 20–34 : multiplier = 1.50  (~86% healthy)
      streak ≥ 35  : multiplier = 1.75  (~88% healthy — financially stable)

    Stress weights still scale with stress_base so high-risk customers always
    have some stress events, but their good streaks gradually dampen them.
    lending_app is hard-capped at 3 weight units regardless of anything.
    """
    if good_streak >= 35:
        streak_mult = 1.75
    elif good_streak >= 20:
        streak_mult = 1.50
    elif good_streak >= 10:
        streak_mult = 1.25
    else:
        streak_mult = 1.00

    weights: dict[str, float] = {
        k: float(v) * streak_mult for k, v in _HEALTHY_WEIGHTS.items()
    }
    scale = 1.0 + stress * 1.5          # range: 1.0 (stress=0) … 2.5 (stress=1)
    for kind, base in _STRESS_BASE_WEIGHTS.items():
        if kind == "lending_app":
            weights[kind] = min(base * scale, 3.0)   # hard cap
        else:
            weights[kind] = base * scale

    return list(weights.keys()), list(weights.values())


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def next_transaction(customer: dict, now: datetime,
                     good_streak: int = 0) -> tuple[TransactionEvent, str]:
    """
    Generate exactly one transaction for *customer* at timestamp *now*.
    Returns (TransactionEvent, kind) so the caller can update the streak.

    Stress fraction WITH streak=0:
      stress_base = 0.0 (healthy)  → ~0%  stress events
      stress_base = 0.5 (at-risk)  → ~12% stress events
      stress_base = 1.0 (high-risk)→ ~25% stress events

    With streak=35+ those fractions are further halved by the healthy multiplier.
    """
    cid    = customer["customer_id"]
    income = customer["monthly_income"]
    emi    = customer["emi_amount"]
    stress = customer["stress_base"]
    credit = customer["credit_limit"]

    kinds, weights = _build_weights(stress, good_streak=good_streak)
    kind = random.choices(kinds, weights=weights, k=1)[0]

    # ── HEALTHY ──────────────────────────────────────────────────────────

    if kind == "salary_credit":
        # On-time for healthy customers; mild delay possible for stressed.
        # → income_coverage_ratio↑, salary_delay_days↓ → PD↓ → score↓
        delay_days = (0 if stress < 0.3
                      else random.randint(0, max(1, int(4 * stress))))
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SALARY_CREDIT,
            amount=round(income * random.uniform(0.92, 1.02)),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now - timedelta(days=delay_days),
        ), kind

    elif kind == "utility_ok":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 4000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "auto_debit_ok":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi),
            payment_channel="ECS",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "upi_groceries":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(200, min(income * 0.05, 3000)), 2),
            merchant_category=MerchantCategory.GROCERIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "credit_card_bill_pay":
        # CRITICAL: bill payment → SAVINGS + TRANSFER_OUT → cc_s NOT touched
        # → credit_utilization_delta NOT raised → score ↓
        bill = round(random.uniform(
            credit * 0.03,
            min(credit * 0.20, income * 0.25),
        ), 2)
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.TRANSFER_OUT,
            amount=bill,
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "atm_small":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=random.choice([500, 1000, 2000]),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="ATM",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "upi_dining":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(80, min(income * 0.03, 1500)), 2),
            merchant_category=MerchantCategory.DINING,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "upi_shopping":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(300, min(income * 0.06, 5000)), 2),
            merchant_category=MerchantCategory.SHOPPING,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "credit_card_spend":
        # Spending ON the card → cc_s↑ → utilization↑ → score↑ slightly
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.CREDIT_CARD,
            txn_type=TransactionType.CREDIT_CARD_PAYMENT,
            amount=round(random.uniform(300, min(credit * 0.08, income * 0.10)), 2),
            merchant_category=random.choice([
                MerchantCategory.GROCERIES,
                MerchantCategory.DINING,
                MerchantCategory.SHOPPING,
            ]),
            payment_channel="POS",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    # ── STRESS ───────────────────────────────────────────────────────────

    elif kind == "savings_drain":
        drain_pct = random.uniform(0.03, max(0.04, 0.12 * stress))
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(500, customer["avg_savings_balance"] * drain_pct), 2),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "atm_large":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=random.choice([5000, 10000]),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="ATM",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind

    elif kind == "utility_fail":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 3000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.FAILED,
            txn_timestamp=now,
        ), kind

    elif kind == "auto_debit_fail":
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi),
            payment_channel="ECS",
            payment_status=PaymentStatus.FAILED,
            txn_timestamp=now,
        ), kind

    else:  # lending_app
        return TransactionEvent(
            customer_id=cid,
            account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(2000, min(15000, income * 0.4)), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI",
            payment_status=PaymentStatus.SUCCESS,
            txn_timestamp=now,
        ), kind


# ══════════════════════════════════════════════════════════════════════════════
# STRESS SIGNAL DETECTION  (console display only — does not affect scoring)
# ══════════════════════════════════════════════════════════════════════════════

def is_stress(evt: TransactionEvent) -> tuple[bool, str]:
    tt  = evt.txn_type.value      if hasattr(evt.txn_type, "value")           else str(evt.txn_type)
    st  = evt.payment_status.value if hasattr(evt.payment_status, "value")    else str(evt.payment_status)
    cat = evt.merchant_category.value if hasattr(evt.merchant_category, "value") else str(evt.merchant_category)
    amt = float(evt.amount)

    if tt == "auto_debit" and st == "failed":
        return True, "FAILED EMI"
    if cat == "lending_app":
        return True, "LENDING APP"
    if tt == "utility_payment" and st == "failed":
        return True, "FAILED UTILITY"
    if tt == "atm_withdrawal" and amt >= 5000:
        return True, "LARGE ATM"
    if tt == "savings_withdrawal" and amt >= 8000:
        return True, "SAVINGS DRAIN"
    return False, ""


def _txn_label(evt: TransactionEvent) -> str:
    """Human-readable label — distinguishes CC spend from CC bill payment."""
    if (evt.txn_type == TransactionType.TRANSFER_OUT and
            evt.account_type == AccountType.SAVINGS):
        return "CC BILL PAYMENT"
    if (evt.txn_type == TransactionType.CREDIT_CARD_PAYMENT and
            evt.account_type == AccountType.CREDIT_CARD):
        return "CC SPEND"
    return (evt.txn_type.value if hasattr(evt.txn_type, "value")
            else str(evt.txn_type)).replace("_", " ").upper()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# STATE UPDATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _update_state(state: CustomerState, kind: str) -> None:
    """
    Update good_streak and counters after each transaction.

    Healthy kinds → streak increments.
    Stress kinds  → streak resets to 0 (any single bad event breaks the run).
    """
    if kind in _HEALTHY_KINDS:
        state.good_streak   += 1
        state.total_healthy += 1
    elif kind in _STRESS_KINDS:
        state.good_streak  = 0      # reset — sustained good behavior interrupted
        state.total_stress += 1


def _update_balance(state: CustomerState,
                    evt: TransactionEvent) -> None:
    """
    Update estimated_balance based on the transaction.

    Rules:
      - FAILED transactions move no money.
      - Salary credits / UPI credits / reversals → INFLOWS → balance ↑
      - All SAVINGS account debits (spend, bill pay, ATM, EMI, etc.) → balance ↓
      - CREDIT_CARD account transactions don't touch the savings balance directly
        (the bill payment is already handled as SAVINGS TRANSFER_OUT).
      - Balance is floored at 0 — we don't model overdraft.
    """
    st  = (evt.payment_status.value if hasattr(evt.payment_status, "value")
           else str(evt.payment_status))
    tt  = (evt.txn_type.value if hasattr(evt.txn_type, "value")
           else str(evt.txn_type))
    amt = float(evt.amount)

    if st == "failed":
        return   # no money moves on failed transactions

    # Inflows to savings
    if tt in ("salary_credit", "upi_credit", "neft_rtgs", "reversal"):
        state.estimated_balance += amt

    # Outflows from savings account
    elif (hasattr(evt.account_type, "value") and
          evt.account_type.value == "savings") or str(evt.account_type) == "savings":
        state.estimated_balance = max(0.0, state.estimated_balance - amt)

    # LOAN / CREDIT_CARD account transactions don't affect estimated savings balance




def _print_summary(customers: list, current_scores: dict,
                   txn_counts: dict, sim_states: dict,
                   total_txns: int) -> None:
    cust_map = {c["customer_id"]: c for c in customers}
    print(f"\n\n{'='*80}")
    print("SENTINEL — Session Summary  (stopped by Ctrl+C)")
    print(f"  Total transactions : {total_txns:,}")
    print(f"  Customers active   : "
          f"{sum(1 for v in txn_counts.values() if v > 0)}/{len(customers)}")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Customer':<12} {'Txns':>5} {'Score':>6} "
          f"{'Tier':<12} {'Streak':>6} {'Balance':>12}  Risk")
    print(f"  {'─'*76}")

    ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (cid, score) in enumerate(ranked[:20], 1):
        c     = cust_map.get(cid, {})
        tier  = pulse_score_to_tier(score)
        mark  = {"green": "●", "yellow": "◆",
                 "orange": "▲", "red": "■"}.get(tier, "?")
        state = sim_states[cid]
        print(f"  {rank:<5} {cid:<12} {txn_counts[cid]:>5} {score:>6} "
              f"{mark} {tier:<10} {state.good_streak:>6} "
              f"₹{state.estimated_balance:>10,.0f}  "
              f"{c.get('risk_level', '?')}")

    tier_counts = {"green": 0, "yellow": 0, "orange": 0, "red": 0}
    for s in current_scores.values():
        tier_counts[pulse_score_to_tier(s)] += 1
    print(f"\n  Tier distribution : "
          f"● green={tier_counts['green']}  "
          f"◆ yellow={tier_counts['yellow']}  "
          f"▲ orange={tier_counts['orange']}  "
          f"■ red={tier_counts['red']}")

    counts = list(txn_counts.values())
    if counts:
        print(f"  Txn spread (bias) : "
              f"min={min(counts)}  max={max(counts)}  "
              f"avg={sum(counts) / len(counts):.1f}")

    # Streak stats
    streaks = [s.good_streak for s in sim_states.values()]
    print(f"  Good streaks      : "
          f"max={max(streaks)}  avg={sum(streaks)/len(streaks):.1f}  "
          f"customers@10+={sum(1 for s in streaks if s >= 10)}  "
          f"customers@35+={sum(1 for s in streaks if s >= 35)}")
    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    n_customers: int = 100,
    delay_ms: float  = 0,
    db_conn          = None,
    dynamo_db        = None,
) -> None:
    """
    Infinite loop — runs until Ctrl+C.

    Per-customer live state (CustomerState) is maintained across the entire
    session:
      • estimated_balance  — grows with salary, shrinks with every spend.
                             Fed into savings_runway_months so the model sees
                             the real balance, not the frozen initial value.
      • good_streak        — consecutive healthy transactions.
                             Boosts healthy weights so sustained good behaviour
                             produces even more healthy transactions.
                             Resets to 0 on ANY stress event.
    """
    random.seed(42)
    customers = [make_customer(i) for i in range(1, n_customers + 1)]
    random.seed()

    cust_ids       = [c["customer_id"] for c in customers]
    current_scores = {cid: 0   for cid in cust_ids}
    txn_counts     = {cid: 0   for cid in cust_ids}
    sim_states     = {c["customer_id"]: make_state(c) for c in customers}
    total_txns     = 0

    risk_counts = {"healthy": 0, "at_risk": 0, "high_risk": 0}
    for c in customers:
        risk_counts[c["risk_level"]] += 1

    print(f"\n{'='*80}")
    print("SENTINEL — Infinite Random Transaction Pipeline")
    print(f"  Customers   : {n_customers}  "
          f"(● healthy={risk_counts['healthy']}  "
          f"◆ at_risk={risk_counts['at_risk']}  "
          f"■ high_risk={risk_counts['high_risk']})")
    print(f"  User pick   : uniform random — zero bias")
    print(f"  Scoring     : LightGBM after every single transaction")
    print(f"  Score dir   : HIGHER = more likely to default (worse)")
    print(f"  Live state  : balance + good_streak tracked per customer")
    print(f"  Streak boost: ×1.25@10  ×1.50@20  ×1.75@35 healthy txns")
    print(f"  Stress cap  : lending_app ≤3%  |  total stress ≤25%")
    print(f"  Kafka       : {settings.kafka_bootstrap_servers}")
    model_pkg = _model()
    if model_pkg:
        print(f"  Model       : LightGBM v{model_pkg.get('version', '?')}  "
              f"AUC={model_pkg.get('cv_auc', 0):.4f}")
    print(f"  Stop        : Ctrl+C")
    print(f"{'='*80}")
    print(f"  {'#':>7}  {'Customer':<12} {'Transaction':<16} {'Amount':>10}  "
          f"{'Signal':<16} {'Score':>6}  {'Δ':>5}  {'Streak':>6}  Tier")
    print(f"  {'─'*82}")

    print("  Registering customers in PostgreSQL...", end="", flush=True)
    for c in customers:
        ensure_customer_postgres(c, db_conn)
    print(f" ✓\n  {'─'*82}")

    with TransactionProducer() as producer:
        try:
            while True:
                # ① Uniform random pick
                customer = random.choice(customers)
                cid      = customer["customer_id"]
                state    = sim_states[cid]
                txn_ts   = datetime.now(timezone.utc)

                # ② Generate transaction — pass good_streak so weights are adjusted
                evt, kind = next_transaction(customer, txn_ts,
                                             good_streak=state.good_streak)

                # ③ Publish to Kafka, persist to PostgreSQL
                producer.publish(evt)
                insert_transaction(evt, db_conn)
                txn_counts[cid] += 1
                total_txns      += 1

                # ④ Update live state BEFORE scoring so the feature vector
                #    sees the correct balance and the model benefits immediately
                _update_balance(state, evt)
                _update_state(state, kind)

                # ⑤ Score via LightGBM — pass live balance into feature vector
                result    = score_customer(customer, db_conn, dynamo_db,
                                           ref=txn_ts,
                                           estimated_balance=state.estimated_balance)
                new_score = result["pulse_score"] if result else current_scores[cid]
                delta     = new_score - current_scores[cid]
                tier      = pulse_score_to_tier(new_score)
                current_scores[cid] = new_score

                # ⑥ Console output
                stress_flag, stress_label = is_stress(evt)

                # Show [STABLE ↓] when customer has a long healthy streak
                if state.good_streak >= 35 and delta <= 0:
                    signal_str = "[STABLE ↓]"
                elif state.good_streak >= 20 and delta <= 0:
                    signal_str = "[HEALTHY ↓]"
                elif stress_flag:
                    signal_str = f"[{stress_label}]"
                else:
                    signal_str = ""

                tier_mark = {"green": "●", "yellow": "◆",
                             "orange": "▲", "red": "■"}.get(tier, "?")
                delta_str = (f"+{delta}" if delta > 0
                             else (str(delta) if delta < 0 else "  0"))

                print(f"  {total_txns:>7}  {cid:<12} {_txn_label(evt):<16} "
                      f"₹{float(evt.amount):>9,.0f}  "
                      f"{signal_str:<16} "
                      f"{new_score:>6}  "
                      f"{delta_str:>5}  "
                      f"{state.good_streak:>6}  "
                      f"{tier_mark} {tier}")

                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

        except KeyboardInterrupt:
            _print_summary(customers, current_scores, txn_counts,
                           sim_states, total_txns)


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

    print("=" * 76)
    print("SENTINEL — Infinite Random Transaction Simulator")
    print(f"  Customers : {args.customers}")
    print(f"  Delay     : {args.delay} ms")
    print(f"  Mode      : uniform random user | score every txn | Ctrl+C to stop")
    print("=" * 76 + "\n")

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

    _model()  # pre-load before loop starts

    try:
        run_pipeline(
            n_customers = args.customers,
            delay_ms    = args.delay,
            db_conn     = conn,
            dynamo_db   = dynamo,
        )
    finally:
        conn.close()
        print("PostgreSQL connection closed.")