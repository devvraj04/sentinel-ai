# Sentinel AI — Per-Customer Baseline & Blind Anomaly Detection
## Complete Implementation Plan

---

## Executive Summary

This document describes the structural changes required to transform Sentinel from a
system that uses **population-level hardcoded thresholds** into one that detects stress
relative to **each customer's own personal financial baseline** — and does so without
ever being told which transactions are stress signals.

**Three core problems this plan solves:**

1. `is_lending_app_upi=True` and `is_auto_debit_failed=True` are boolean labels being
   injected directly onto the TransactionEvent. The model is essentially being told
   "this is a bad transaction" before it even scores. That is label leakage.

2. Flag thresholds like `delay > 7 days`, `wow_drop > 0.3`, `atm_spike > 3.0` are
   population-level magic numbers. A customer whose salary always arrives on the 10th is
   not stressed if it arrives on the 15th. A customer who normally keeps ₹500 is in a
   crisis if their balance drops to ₹100 — a customer who normally keeps ₹50,000 is not.

3. There is no `customer_baseline` table. `avg_savings_balance` is a static value set
   at customer creation and never updated from actual transaction history.

---

## Current State Audit

### What the model is currently told (it shouldn't be)

**`ingestion/schemas/transaction_event.py`** — lines that pre-label stress:
```python
is_lending_app_upi:   bool = False   # ← tells model "this is a bad transaction"
is_auto_debit_failed: bool = False   # ← tells model "this EMI bounced"
is_p2p_transfer:      bool = False
is_investment_txn:    bool = False
```

**`sagemaker/inference.py`** — `build_feature_vector()` reads these labels:
```python
df["is_lending_app_upi"]   = df["merchant_category"] == "lending_app"  # pre-labeled
df["is_auto_debit_failed"] = (df["txn_type"] == "auto_debit") & (df["payment_status"] == "failed")
lend_s = csum(df_short, "is_lending_app_upi", True)  # uses the pre-labeled column
```

**`sagemaker/inference.py`** — hardcoded population thresholds:
```python
fv["flag_salary"]          = 1.0 if delay > 7 else 0.0        # same for everyone
fv["flag_balance"]         = 1.0 if wow_drop > 0.3 else 0.0   # same for everyone
fv["flag_utility"]         = 1.0 if latency > 22 else 0.0     # same for everyone
fv["flag_atm"]             = 1.0 if atm_sp > 3.0 else 0.0     # same for everyone
```

These are not wrong features — the problem is the thresholds are static.

---

## Architecture of the Fix

```
PHASE 1 — Historical Seed (90 days, run once)
  simulate_transactions.py --mode seed --days 90
    ↓
  Transactions written to PostgreSQL (with balance_before, balance_after)
    ↓
  scripts/build_baselines.py  ← NEW SCRIPT
    ↓
  customer_baseline table populated (per-customer statistics)

PHASE 2 — Real-Time (ongoing)
  simulate_transactions.py --mode realtime
    ↓
  Raw transaction → Kafka (NO is_stress, NO is_lending_app_upi label)
    ↓
  feature_pipeline reads transaction + baseline
    ↓
  Z-score features computed relative to that customer's own baseline
    ↓
  LightGBM scores — model detects anomalies itself
```

---

## Step 1 — New Database Table: `customer_baseline`

**File to create:** `data_warehouse/schemas/migrate_003_customer_baseline.sql`

```sql
-- ── Customer statistical baseline ──────────────────────────────────────────────
-- Computed from first 90 days of transaction history per customer.
-- Updated every 30 days via scripts/build_baselines.py.
-- All feature computations in build_feature_vector() use this table
-- instead of hardcoded thresholds.

CREATE TABLE IF NOT EXISTS customer_baseline (
    customer_id             VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id),
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    observation_days        INTEGER NOT NULL DEFAULT 0,  -- how many days of history used

    -- ── Balance statistics ────────────────────────────────────────────────────
    balance_mean            NUMERIC(15,2) DEFAULT 0,   -- rolling 90-day mean balance
    balance_std             NUMERIC(15,2) DEFAULT 1,   -- std deviation of balance
    balance_min_90d         NUMERIC(15,2) DEFAULT 0,   -- lowest balance in 90 days
    balance_max_90d         NUMERIC(15,2) DEFAULT 0,   -- peak balance in 90 days
    balance_trend_slope     NUMERIC(8,4)  DEFAULT 0,   -- positive=growing, negative=eroding

    -- ── Salary statistics ─────────────────────────────────────────────────────
    salary_day_mean         NUMERIC(5,2) DEFAULT 3,    -- typical day-of-month salary arrives
    salary_day_std          NUMERIC(5,2) DEFAULT 1,    -- std deviation of salary day
    salary_amount_mean      NUMERIC(15,2) DEFAULT 0,
    salary_amount_std       NUMERIC(15,2) DEFAULT 1,
    salary_months_observed  INTEGER DEFAULT 0,

    -- ── EMI / auto-debit statistics ───────────────────────────────────────────
    emi_amount_mean         NUMERIC(15,2) DEFAULT 0,
    emi_day_mean            NUMERIC(5,2) DEFAULT 5,
    emi_success_rate        NUMERIC(5,4) DEFAULT 1.0,  -- 1.0 = never bounced
    emi_attempts_90d        INTEGER DEFAULT 0,
    emi_failures_90d        INTEGER DEFAULT 0,

    -- ── Spending behaviour statistics ────────────────────────────────────────
    monthly_spend_mean      NUMERIC(15,2) DEFAULT 0,
    monthly_spend_std       NUMERIC(15,2) DEFAULT 1,
    discretionary_mean      NUMERIC(15,2) DEFAULT 0,   -- dining/entertainment/travel
    discretionary_std       NUMERIC(15,2) DEFAULT 1,
    atm_monthly_mean        NUMERIC(15,2) DEFAULT 0,
    atm_monthly_std         NUMERIC(15,2) DEFAULT 1,
    utility_day_mean        NUMERIC(5,2) DEFAULT 10,   -- typical day utility bills paid
    utility_day_std         NUMERIC(5,2) DEFAULT 3,

    -- ── Credit behaviour statistics ───────────────────────────────────────────
    cc_utilization_mean     NUMERIC(5,4) DEFAULT 0.3,
    cc_utilization_std      NUMERIC(5,4) DEFAULT 0.1,
    upi_to_lending_mean     NUMERIC(15,2) DEFAULT 0,   -- monthly UPI to lending apps (₹)
    upi_to_lending_std      NUMERIC(15,2) DEFAULT 1,

    -- ── P2P / informal lending ────────────────────────────────────────────────
    p2p_outflow_mean        NUMERIC(15,2) DEFAULT 0,
    p2p_outflow_std         NUMERIC(15,2) DEFAULT 1,

    -- ── Anomaly sensitivity thresholds (per customer, not population) ─────────
    -- Computed as: threshold = mean + (z_multiplier * std)
    -- z_multiplier = 2.0 for most signals (flags at 2 std deviations from normal)
    balance_drop_threshold  NUMERIC(8,4) DEFAULT 0.30, -- fraction drop that is alarming
    salary_delay_threshold  NUMERIC(5,2) DEFAULT 7,    -- days late that is alarming
    atm_spike_threshold     NUMERIC(5,2) DEFAULT 3.0,  -- ratio vs baseline that is alarming
    lending_spike_threshold NUMERIC(5,2) DEFAULT 2.5,  -- ratio vs baseline

    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_baseline_customer
    ON customer_baseline(customer_id);

CREATE INDEX IF NOT EXISTS idx_baseline_computed
    ON customer_baseline(computed_at DESC);

-- ── Baseline computation log (audit trail) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS baseline_computation_log (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    txn_count       INTEGER,
    observation_days INTEGER,
    notes           TEXT
);
```

**Apply:**
```powershell
docker cp data_warehouse\schemas\migrate_003_customer_baseline.sql sentinel-postgres:/tmp/migrate_003.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/migrate_003.sql
```

---

## Step 2 — New Script: `scripts/build_baselines.py`

This script reads transaction history for each customer and computes their personal
baseline statistics. Run it once after the historical seed, then re-run periodically
to update baselines as the customer's financial behaviour evolves.

**File:** `scripts/build_baselines.py`

```python
"""
scripts/build_baselines.py
──────────────────────────────────────────────────────────────────────────────
Computes per-customer statistical baselines from transaction history.

Run ONCE after historical seeding, then periodically (e.g. every 30 days).

Usage:
    python -m scripts.build_baselines                    # all customers
    python -m scripts.build_baselines --customer CUST00001  # one customer
    python -m scripts.build_baselines --days 90          # use last 90 days

What it computes:
    For each customer independently:
    - Balance mean, std, min, max, trend over observation window
    - Salary: typical day-of-month, typical amount, variability
    - EMI: typical amount, day, historical success rate
    - Spending: monthly spend mean/std, discretionary, ATM, utility latency
    - Credit: CC utilisation mean/std
    - UPI to lending apps: historical mean/std
    - Per-customer anomaly thresholds (mean + 2.0 * std)

These baselines replace ALL hardcoded population thresholds in
build_feature_vector(). The model sees Z-scores relative to this customer's
own history, not arbitrary population-level cutoffs.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from config.settings import get_settings

settings = get_settings()
logger   = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

Z_MULTIPLIER = 2.0   # flag at 2 standard deviations from the customer's own mean
MIN_OBS_DAYS = 14    # minimum days of history required to compute a meaningful baseline


def get_conn():
    return psycopg2.connect(settings.database_url,
                            cursor_factory=psycopg2.extras.RealDictCursor)


def load_transactions(conn, customer_id: str, days: int) -> pd.DataFrame:
    """Load transaction history for a customer over the observation window."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                txn_type, amount, merchant_category, payment_status,
                account_type, txn_timestamp, balance_before, balance_after,
                counterparty_id, counterparty_name, platform
            FROM transactions
            WHERE customer_id = %s
              AND txn_timestamp >= %s
            ORDER BY txn_timestamp ASC
        """, (customer_id, since))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    return df


def compute_balance_stats(df: pd.DataFrame) -> dict:
    """
    Compute balance statistics from balance_after on savings/current transactions.
    Uses the actual running balance stored per transaction — not inferred from flows.
    """
    bal_rows = df[df["account_type"].isin(["savings", "current"]) &
                  df["balance_after"].notna()].copy()
    if len(bal_rows) < 3:
        return {
            "balance_mean": 0.0, "balance_std": 1.0,
            "balance_min_90d": 0.0, "balance_max_90d": 0.0,
            "balance_trend_slope": 0.0, "balance_drop_threshold": 0.30,
        }

    balances = bal_rows["balance_after"].astype(float).values
    mean_bal = float(np.mean(balances))
    std_bal  = float(np.std(balances)) if np.std(balances) > 0 else max(mean_bal * 0.1, 100.0)

    # Linear trend over time (positive = growing balance, negative = eroding)
    x = np.arange(len(balances), dtype=float)
    slope = float(np.polyfit(x, balances, 1)[0]) if len(balances) > 2 else 0.0

    # Personalised drop threshold: 2 std deviations below mean as fraction of mean
    # e.g. mean=500, std=80 → threshold = (500 - 2*80) / 500 = 0.68 (68% drop is alarming)
    # e.g. mean=50000, std=5000 → threshold = (50000 - 2*5000) / 50000 = 0.80
    drop_thresh = max((std_bal * Z_MULTIPLIER) / max(mean_bal, 1), 0.10)
    drop_thresh = min(drop_thresh, 0.90)  # cap at 90%

    return {
        "balance_mean":          round(mean_bal, 2),
        "balance_std":           round(std_bal, 2),
        "balance_min_90d":       round(float(np.min(balances)), 2),
        "balance_max_90d":       round(float(np.max(balances)), 2),
        "balance_trend_slope":   round(slope, 4),
        "balance_drop_threshold": round(drop_thresh, 4),
    }


def compute_salary_stats(df: pd.DataFrame) -> dict:
    """Compute personal salary timing and amount statistics."""
    salaries = df[df["txn_type"] == "salary_credit"].copy()
    if salaries.empty:
        return {
            "salary_day_mean": 3.0, "salary_day_std": 1.0,
            "salary_amount_mean": 0.0, "salary_amount_std": 1.0,
            "salary_months_observed": 0, "salary_delay_threshold": 7.0,
        }

    days   = salaries["txn_timestamp"].dt.day.astype(float).values
    amts   = salaries["amount"].astype(float).values
    day_m  = float(np.mean(days))
    day_s  = float(np.std(days)) if np.std(days) > 0 else 2.0
    amt_m  = float(np.mean(amts))
    amt_s  = float(np.std(amts)) if np.std(amts) > 0 else max(amt_m * 0.05, 1.0)

    # Personalised delay threshold: mean arrival day + 2 std deviations
    delay_thresh = round(day_m + Z_MULTIPLIER * day_s, 1)
    delay_thresh = max(delay_thresh - day_m, 3.0)  # convert to delay days from mean

    return {
        "salary_day_mean":         round(day_m, 2),
        "salary_day_std":          round(day_s, 2),
        "salary_amount_mean":      round(amt_m, 2),
        "salary_amount_std":       round(amt_s, 2),
        "salary_months_observed":  len(salaries),
        "salary_delay_threshold":  round(delay_thresh, 1),
    }


def compute_emi_stats(df: pd.DataFrame) -> dict:
    """Compute EMI payment behaviour statistics."""
    emis = df[df["txn_type"] == "auto_debit"].copy()
    if emis.empty:
        return {
            "emi_amount_mean": 0.0, "emi_day_mean": 5.0,
            "emi_success_rate": 1.0, "emi_attempts_90d": 0, "emi_failures_90d": 0,
        }

    attempts = len(emis)
    failures = len(emis[emis["payment_status"] == "failed"])
    success_rate = float((attempts - failures) / max(attempts, 1))

    successful = emis[emis["payment_status"] == "success"]
    amt_mean = float(successful["amount"].mean()) if not successful.empty else 0.0
    day_mean = float(emis["txn_timestamp"].dt.day.mean())

    return {
        "emi_amount_mean":    round(amt_mean, 2),
        "emi_day_mean":       round(day_mean, 1),
        "emi_success_rate":   round(success_rate, 4),
        "emi_attempts_90d":   attempts,
        "emi_failures_90d":   failures,
    }


def compute_spending_stats(df: pd.DataFrame, monthly_income: float) -> dict:
    """Compute monthly spending pattern statistics."""
    # Group debits by calendar month
    debits = df[~df["txn_type"].isin([
        "salary_credit", "upi_credit", "neft_rtgs", "reversal", "loan_disbursement"
    ])].copy()
    debits["month"] = debits["txn_timestamp"].dt.to_period("M")

    monthly = debits.groupby("month")["amount"].sum()
    spend_m = float(monthly.mean()) if len(monthly) > 0 else 0.0
    spend_s = float(monthly.std()) if len(monthly) > 1 else max(spend_m * 0.15, 1.0)

    # Discretionary
    disc_cats = {"dining", "entertainment", "travel", "shopping"}
    disc = df[df["merchant_category"].isin(disc_cats)]
    disc_monthly = disc.groupby(disc["txn_timestamp"].dt.to_period("M"))["amount"].sum()
    disc_m = float(disc_monthly.mean()) if len(disc_monthly) > 0 else 0.0
    disc_s = float(disc_monthly.std()) if len(disc_monthly) > 1 else max(disc_m * 0.20, 1.0)

    # ATM
    atm = df[df["txn_type"] == "atm_withdrawal"]
    atm_monthly = atm.groupby(atm["txn_timestamp"].dt.to_period("M"))["amount"].sum()
    atm_m = float(atm_monthly.mean()) if len(atm_monthly) > 0 else 0.0
    atm_s = float(atm_monthly.std()) if len(atm_monthly) > 1 else max(atm_m * 0.30, 100.0)
    atm_spike_thresh = round((atm_m + Z_MULTIPLIER * atm_s) / max(atm_m, 1), 1)
    atm_spike_thresh = max(atm_spike_thresh, 1.5)

    # Utility latency
    utils = df[df["txn_type"] == "utility_payment"]
    util_days = utils["txn_timestamp"].dt.day.astype(float)
    util_m = float(util_days.mean()) if len(util_days) > 0 else 10.0
    util_s = float(util_days.std()) if len(util_days) > 1 else 3.0

    return {
        "monthly_spend_mean":    round(spend_m, 2),
        "monthly_spend_std":     round(spend_s, 2),
        "discretionary_mean":    round(disc_m, 2),
        "discretionary_std":     round(disc_s, 2),
        "atm_monthly_mean":      round(atm_m, 2),
        "atm_monthly_std":       round(atm_s, 2),
        "utility_day_mean":      round(util_m, 1),
        "utility_day_std":       round(util_s, 1),
        "atm_spike_threshold":   round(atm_spike_thresh, 2),
    }


def compute_credit_stats(df: pd.DataFrame, credit_limit: float) -> dict:
    """Compute credit card utilisation and UPI-to-lending statistics."""
    cc = df[df["account_type"] == "credit_card"]
    cc_monthly = cc.groupby(cc["txn_timestamp"].dt.to_period("M"))["amount"].sum()
    cc_util_vals = (cc_monthly / max(credit_limit, 1)).clip(0, 1)
    cc_m = float(cc_util_vals.mean()) if len(cc_util_vals) > 0 else 0.3
    cc_s = float(cc_util_vals.std()) if len(cc_util_vals) > 1 else 0.1

    # UPI to lending apps
    lending = df[df["merchant_category"] == "lending_app"]
    lend_monthly = lending.groupby(lending["txn_timestamp"].dt.to_period("M"))["amount"].sum()
    lend_m = float(lend_monthly.mean()) if len(lend_monthly) > 0 else 0.0
    lend_s = float(lend_monthly.std()) if len(lend_monthly) > 1 else max(lend_m * 0.30, 100.0)
    lend_spike_thresh = round((lend_m + Z_MULTIPLIER * lend_s) / max(lend_m, 100.0), 1)
    lend_spike_thresh = max(lend_spike_thresh, 2.0)

    # P2P
    p2p = df[df["is_p2p_transfer"] == True] if "is_p2p_transfer" in df.columns else df[
        (df["txn_type"] == "upi_debit") &
        (~df["merchant_category"].isin(["lending_app", "dining", "shopping",
                                         "groceries", "utilities", "fuel"]))
    ]
    p2p_monthly = p2p.groupby(p2p["txn_timestamp"].dt.to_period("M"))["amount"].sum()
    p2p_m = float(p2p_monthly.mean()) if len(p2p_monthly) > 0 else 0.0
    p2p_s = float(p2p_monthly.std()) if len(p2p_monthly) > 1 else max(p2p_m * 0.30, 100.0)

    return {
        "cc_utilization_mean":      round(cc_m, 4),
        "cc_utilization_std":       round(cc_s, 4),
        "upi_to_lending_mean":      round(lend_m, 2),
        "upi_to_lending_std":       round(lend_s, 2),
        "p2p_outflow_mean":         round(p2p_m, 2),
        "p2p_outflow_std":          round(p2p_s, 2),
        "lending_spike_threshold":  round(lend_spike_thresh, 2),
    }


def compute_baseline_for_customer(
    conn, customer_id: str, days: int
) -> dict | None:
    """Compute the full baseline for a single customer."""
    df = load_transactions(conn, customer_id, days)
    if df.empty or len(df) < 5:
        logger.debug("Not enough transactions for %s (%d rows)", customer_id, len(df))
        return None

    observation_days = min(
        days,
        int((df["txn_timestamp"].max() - df["txn_timestamp"].min()).days) + 1
    )
    if observation_days < MIN_OBS_DAYS:
        logger.debug("Only %d days of history for %s", observation_days, customer_id)
        return None

    # Fetch customer profile for income/credit_limit
    with conn.cursor() as cur:
        cur.execute(
            "SELECT monthly_income, credit_limit, emi_amount FROM customers "
            "WHERE customer_id = %s",
            (customer_id,)
        )
        profile = cur.fetchone()
    income       = float(profile["monthly_income"] or 0) if profile else 0.0
    credit_limit = float(profile["credit_limit"] or 0)   if profile else 0.0

    baseline = {"customer_id": customer_id, "observation_days": observation_days}
    baseline.update(compute_balance_stats(df))
    baseline.update(compute_salary_stats(df))
    baseline.update(compute_emi_stats(df))
    baseline.update(compute_spending_stats(df, income))
    baseline.update(compute_credit_stats(df, credit_limit))
    return baseline


def upsert_baseline(conn, baseline: dict) -> None:
    """Write or update customer baseline in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO customer_baseline (
                customer_id, computed_at, observation_days,
                balance_mean, balance_std, balance_min_90d, balance_max_90d,
                balance_trend_slope, balance_drop_threshold,
                salary_day_mean, salary_day_std, salary_amount_mean, salary_amount_std,
                salary_months_observed, salary_delay_threshold,
                emi_amount_mean, emi_day_mean, emi_success_rate,
                emi_attempts_90d, emi_failures_90d,
                monthly_spend_mean, monthly_spend_std,
                discretionary_mean, discretionary_std,
                atm_monthly_mean, atm_monthly_std, utility_day_mean, utility_day_std,
                atm_spike_threshold,
                cc_utilization_mean, cc_utilization_std,
                upi_to_lending_mean, upi_to_lending_std,
                p2p_outflow_mean, p2p_outflow_std, lending_spike_threshold,
                updated_at
            ) VALUES (
                %(customer_id)s, NOW(), %(observation_days)s,
                %(balance_mean)s, %(balance_std)s, %(balance_min_90d)s, %(balance_max_90d)s,
                %(balance_trend_slope)s, %(balance_drop_threshold)s,
                %(salary_day_mean)s, %(salary_day_std)s,
                %(salary_amount_mean)s, %(salary_amount_std)s,
                %(salary_months_observed)s, %(salary_delay_threshold)s,
                %(emi_amount_mean)s, %(emi_day_mean)s, %(emi_success_rate)s,
                %(emi_attempts_90d)s, %(emi_failures_90d)s,
                %(monthly_spend_mean)s, %(monthly_spend_std)s,
                %(discretionary_mean)s, %(discretionary_std)s,
                %(atm_monthly_mean)s, %(atm_monthly_std)s,
                %(utility_day_mean)s, %(utility_day_std)s,
                %(atm_spike_threshold)s,
                %(cc_utilization_mean)s, %(cc_utilization_std)s,
                %(upi_to_lending_mean)s, %(upi_to_lending_std)s,
                %(p2p_outflow_mean)s, %(p2p_outflow_std)s,
                %(lending_spike_threshold)s,
                NOW()
            )
            ON CONFLICT (customer_id) DO UPDATE SET
                computed_at           = EXCLUDED.computed_at,
                observation_days      = EXCLUDED.observation_days,
                balance_mean          = EXCLUDED.balance_mean,
                balance_std           = EXCLUDED.balance_std,
                balance_min_90d       = EXCLUDED.balance_min_90d,
                balance_max_90d       = EXCLUDED.balance_max_90d,
                balance_trend_slope   = EXCLUDED.balance_trend_slope,
                balance_drop_threshold= EXCLUDED.balance_drop_threshold,
                salary_day_mean       = EXCLUDED.salary_day_mean,
                salary_day_std        = EXCLUDED.salary_day_std,
                salary_amount_mean    = EXCLUDED.salary_amount_mean,
                salary_amount_std     = EXCLUDED.salary_amount_std,
                salary_months_observed= EXCLUDED.salary_months_observed,
                salary_delay_threshold= EXCLUDED.salary_delay_threshold,
                emi_amount_mean       = EXCLUDED.emi_amount_mean,
                emi_day_mean          = EXCLUDED.emi_day_mean,
                emi_success_rate      = EXCLUDED.emi_success_rate,
                emi_attempts_90d      = EXCLUDED.emi_attempts_90d,
                emi_failures_90d      = EXCLUDED.emi_failures_90d,
                monthly_spend_mean    = EXCLUDED.monthly_spend_mean,
                monthly_spend_std     = EXCLUDED.monthly_spend_std,
                discretionary_mean    = EXCLUDED.discretionary_mean,
                discretionary_std     = EXCLUDED.discretionary_std,
                atm_monthly_mean      = EXCLUDED.atm_monthly_mean,
                atm_monthly_std       = EXCLUDED.atm_monthly_std,
                utility_day_mean      = EXCLUDED.utility_day_mean,
                utility_day_std       = EXCLUDED.utility_day_std,
                atm_spike_threshold   = EXCLUDED.atm_spike_threshold,
                cc_utilization_mean   = EXCLUDED.cc_utilization_mean,
                cc_utilization_std    = EXCLUDED.cc_utilization_std,
                upi_to_lending_mean   = EXCLUDED.upi_to_lending_mean,
                upi_to_lending_std    = EXCLUDED.upi_to_lending_std,
                p2p_outflow_mean      = EXCLUDED.p2p_outflow_mean,
                p2p_outflow_std       = EXCLUDED.p2p_outflow_std,
                lending_spike_threshold = EXCLUDED.lending_spike_threshold,
                updated_at            = NOW()
        """, baseline)
    conn.commit()


def run(customer_id: str | None = None, days: int = 90) -> None:
    conn = get_conn()
    if customer_id:
        customers = [customer_id]
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT customer_id FROM customers ORDER BY customer_id")
            customers = [r["customer_id"] for r in cur.fetchall()]

    logger.info("Computing baselines for %d customers (window=%d days)", len(customers), days)
    computed = skipped = 0

    for cid in customers:
        baseline = compute_baseline_for_customer(conn, cid, days)
        if baseline:
            upsert_baseline(conn, baseline)
            computed += 1
            logger.debug("Baseline written: %s (bal_mean=%.0f, sal_day=%.1f±%.1f)",
                         cid, baseline["balance_mean"],
                         baseline["salary_day_mean"], baseline["salary_day_std"])
        else:
            skipped += 1

    conn.close()
    logger.info("Done. Computed: %d  Skipped (insufficient data): %d", computed, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-customer baselines")
    parser.add_argument("--customer", default=None, help="Single customer ID (default: all)")
    parser.add_argument("--days",     type=int, default=90, help="Observation window in days")
    args = parser.parse_args()
    run(customer_id=args.customer, days=args.days)
```

---

## Step 3 — Remove Label Leakage from TransactionEvent

**File:** `ingestion/schemas/transaction_event.py`

Remove the pre-labelling fields that tell the model what kind of stress this is:

```python
# REMOVE these fields entirely:
is_lending_app_upi:   bool = False   # ← DELETE — model infers this from counterparty + amount
is_auto_debit_failed: bool = False   # ← DELETE — model infers this from txn_type + payment_status
is_p2p_transfer:      bool = False   # ← DELETE — model infers from platform + counterparty
is_investment_txn:    bool = False   # ← DELETE — model infers from counterparty keywords
```

**Why this matters:** When `is_lending_app_upi=True` is on the transaction, the feature
pipeline does `lend_s = csum(df, "is_lending_app_upi", True)` — it is directly
summing a pre-labelled column. The model never has to learn that lending-app UPIs are
stress signals — it is told. With the field removed, the model must infer it from:
counterparty_id patterns, amount relative to income, and frequency changes vs baseline.

The enrichment classifier (`ingestion/enrichment/transaction_classifier.py`) that
currently sets these fields can continue running to classify transaction types
(`txn_type`, `merchant_category`, `platform`) — those are legitimate inferences about
**what happened**, not labels about **whether it is stress**.

---

## Step 4 — Rewrite Flag Features to Use Per-Customer Baseline

**File:** `sagemaker/inference.py` — `build_feature_vector()` function

Replace all hardcoded threshold flags with Z-score based personalised flags:

```python
# ── OLD (population-level hardcoded) ──────────────────────────────────────────
fv["flag_salary"]  = 1.0 if delay > 7 else 0.0
fv["flag_balance"] = 1.0 if wow_drop > 0.3 else 0.0
fv["flag_atm"]     = 1.0 if atm_sp > 3.0 else 0.0

# ── NEW (per-customer personalised) ──────────────────────────────────────────
# Load baseline once at the top of build_feature_vector():
baseline = _load_customer_baseline(customer_id)

# Salary delay — personalised threshold from this customer's history
# e.g. if salary always comes on 10th ± 2 days, flag only if arrives after 14th
sal_z = (delay - baseline["salary_day_std"] * 0) / max(baseline["salary_day_std"], 0.5)
fv["flag_salary"] = 1.0 if delay > baseline["salary_delay_threshold"] else 0.0

# Balance drop — personalised threshold
# if balance is 2+ std deviations below this customer's personal mean, flag it
current_balance = estimated_balance or baseline["balance_mean"]
balance_z = (baseline["balance_mean"] - current_balance) / max(baseline["balance_std"], 1.0)
fv["flag_balance"] = 1.0 if balance_z > Z_MULTIPLIER else 0.0

# ATM spike — personalised threshold
fv["flag_atm"] = 1.0 if atm_sp > baseline["atm_spike_threshold"] else 0.0

# Lending spike — personalised threshold
fv["flag_lending"] = 1.0 if lend_s > (baseline["upi_to_lending_mean"] +
                              Z_MULTIPLIER * baseline["upi_to_lending_std"]) else 0.0
```

Add a baseline loader function:

```python
def _load_customer_baseline(conn, customer_id: str) -> dict:
    """
    Load per-customer baseline from PostgreSQL.
    Returns population-level defaults if no baseline exists yet
    (i.e. customer is new with < 14 days of history).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM customer_baseline WHERE customer_id = %s",
            (customer_id,)
        )
        row = cur.fetchone()
    if row:
        return dict(row)
    # No baseline yet — use conservative population defaults
    return {
        "balance_mean":            0.0,
        "balance_std":             1.0,
        "balance_drop_threshold":  0.30,
        "salary_day_mean":         3.0,
        "salary_day_std":          2.0,
        "salary_delay_threshold":  7.0,
        "atm_spike_threshold":     3.0,
        "lending_spike_threshold": 2.5,
        "upi_to_lending_mean":     0.0,
        "upi_to_lending_std":      100.0,
        "emi_success_rate":        1.0,
        "cc_utilization_mean":     0.3,
        "cc_utilization_std":      0.1,
    }
```

---

## Step 5 — Add Z-Score Features to the Feature Vector

The model should receive the raw Z-scores (how many standard deviations from this
customer's normal) as additional features alongside the ratios. This gives the model
richer signal than just binary flags.

**Add to `build_feature_vector()` in `sagemaker/inference.py`:**

```python
# ── Z-SCORE FEATURES (personalised deviation from baseline) ─────────────────
# These tell the model HOW FAR this customer is from their own normal,
# not whether they crossed an arbitrary population threshold.

baseline = _load_customer_baseline(conn, customer_id)

# Balance Z-score: positive means balance has dropped below customer's mean
if baseline["balance_std"] > 0 and baseline["balance_mean"] > 0:
    current_bal = estimated_balance if estimated_balance else balance_last
    fv["balance_zscore"] = float(np.clip(
        (baseline["balance_mean"] - current_bal) / baseline["balance_std"],
        -5, 5
    ))
else:
    fv["balance_zscore"] = 0.0

# Salary delay Z-score
fv["salary_delay_zscore"] = float(np.clip(
    (delay - baseline["salary_day_mean"]) / max(baseline["salary_day_std"], 0.5),
    -5, 5
))

# ATM spending Z-score (current month vs baseline mean)
atm_this_month = float(df_short[df_short["txn_type"] == "atm_withdrawal"]["amount"].sum())
if baseline["atm_monthly_std"] > 0 and baseline["atm_monthly_mean"] > 0:
    fv["atm_spend_zscore"] = float(np.clip(
        (atm_this_month - baseline["atm_monthly_mean"]) / baseline["atm_monthly_std"],
        -5, 5
    ))
else:
    fv["atm_spend_zscore"] = 0.0

# Lending UPI Z-score
lend_this_month = float(df_short[df_short["merchant_category"] == "lending_app"]["amount"].sum())
if baseline["upi_to_lending_std"] > 0:
    fv["lending_spend_zscore"] = float(np.clip(
        (lend_this_month - baseline["upi_to_lending_mean"]) / baseline["upi_to_lending_std"],
        -5, 5
    ))
else:
    fv["lending_spend_zscore"] = 0.0

# EMI reliability Z-score
# Use the historical success rate deviation as a running stress signal
fv["emi_reliability_score"] = float(baseline["emi_success_rate"])  # 1.0=perfect, <1.0=bounces
```

**Add these new features to `FEATURE_COLS` in `models/lightgbm/train_lgbm.py`:**
```python
"balance_zscore",        # personalised balance deviation
"salary_delay_zscore",   # personalised salary timing deviation
"atm_spend_zscore",      # personalised ATM withdrawal deviation
"lending_spend_zscore",  # personalised lending app spend deviation
"emi_reliability_score", # historical EMI success rate
```

---

## Step 6 — Update Training Data Generator

**File:** `models/training_pipelines/build_training_data.py`

The synthetic training data must include the new Z-score features. Since these are
Z-scores (standard normal distributed by construction), they are easy to simulate:

```python
# ── Z-SCORE FEATURES (normally distributed, customer-relative) ───────────────
# These simulate what the personalised features look like in training:
# - positive Z-score = customer is above their personal mean (stressed)
# - negative Z-score = customer is below their personal mean (healthier)

# Correlated with the existing signals — stressed customers have high positive Z-scores
def simulate_zscores(core: pd.DataFrame, new_sig: pd.DataFrame, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    # Balance Z-score: correlated with balance_wow_drop (same underlying reality)
    df["balance_zscore"] = (
        core["balance_wow_drop_pct"] / 16.0 +          # normalise to ~N(0,1)
        RNG.normal(0, 0.3, n)                           # add noise
    ).clip(-5, 5)

    # Salary delay Z-score: correlated with salary_delay_days
    df["salary_delay_zscore"] = (
        (core["salary_delay_days"] - 1.2) / 2.5 +     # normalise by population mean/std
        RNG.normal(0, 0.2, n)
    ).clip(-5, 5)

    # ATM Z-score: correlated with atm_withdrawal_spike
    df["atm_spend_zscore"] = (
        np.log1p(core["atm_withdrawal_spike"].values) +
        RNG.normal(0, 0.25, n)
    ).clip(-5, 5)

    # Lending Z-score: correlated with upi_lending_spike_ratio
    df["lending_spend_zscore"] = (
        np.log1p(core["upi_lending_spike_ratio"].values) +
        RNG.normal(0, 0.25, n)
    ).clip(-5, 5)

    # EMI reliability: inversely correlated with failed_auto_debit_count
    df["emi_reliability_score"] = np.clip(
        1.0 - (core["failed_auto_debit_count"] * 0.15), 0.0, 1.0
    )
    return df
```

---

## Step 7 — Revised Run Order

```powershell
# ── SETUP (once) ─────────────────────────────────────────────────────────────

# 1. Start infrastructure
docker compose up -d

# 2. Initialise Kafka + DynamoDB
python -m scripts.init_kafka_topics
python -m scripts.init_dynamodb

# 3. Apply all migrations (including the new baseline table)
docker cp data_warehouse\schemas\migrate_001_pulse_history.sql sentinel-postgres:/tmp/m1.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m1.sql

docker cp data_warehouse\schemas\migrate_002_txn_redesign.sql sentinel-postgres:/tmp/m2.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m2.sql

docker cp data_warehouse\schemas\migrate_003_customer_baseline.sql sentinel-postgres:/tmp/m3.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m3.sql

# ── PHASE 1: HISTORICAL SEED ─────────────────────────────────────────────────

# 4. Generate and retrain model to include new Z-score features
python -m models.training_pipelines.build_training_data
python -m models.lightgbm.train_lgbm

# 5. Start feature pipeline
python -m ingestion.consumers.feature_pipeline

# 6. Start API
uvicorn api.main:app --reload --port 8001

# 7. Run historical simulation (fast, no delay — build 90 days of baseline data)
python -m scripts.simulate_transactions --customers 100 --delay 0
# Let it run until ~5,000–10,000 transactions (Ctrl+C when done)

# 8. Compute per-customer baselines from the seeded history
python -m scripts.build_baselines --days 90
# Expected output:
#   Computing baselines for 100 customers (window=90 days)
#   Done. Computed: 100  Skipped: 0

# Verify a specific customer's baseline:
# docker exec sentinel-postgres psql -U sentinel -d sentinel_db -c
#   "SELECT customer_id, balance_mean, balance_std, salary_day_mean, salary_delay_threshold
#    FROM customer_baseline ORDER BY customer_id LIMIT 5;"

# ── PHASE 2: REAL-TIME ────────────────────────────────────────────────────────

# 9. Run real-time simulation — model now detects anomalies itself
python -m scripts.simulate_transactions --customers 100 --delay 300
```

---

## Step 8 — What Changes in the Console Output

After this change, the simulator console should still show the `[FAILED EMI]` and
`[LENDING APP]` labels — but these are now **only for human display**. They are NOT
passed to the model. The `is_stress()` function in `simulate_transactions.py` only
produces labels for the console print; `score_customer()` receives only raw transaction
data and the customer profile.

**Old flow (label leakage):**
```
Transaction generated → is_lending_app_upi=True set on event
                      → Pipeline reads flag → feature column pre-labelled
                      → Model told "this customer did a lending app UPI"
```

**New flow (blind detection):**
```
Transaction generated → counterparty=slice@upi, amount=15000, merchant_category=lending_app
                      → Pipeline computes lending_spend_zscore from baseline
                      → lending_spend_zscore = (15000 - baseline.mean) / baseline.std = +4.2
                      → Model sees a Z-score of 4.2 and high upi_lending_spike_ratio
                      → Model decides this is stress
```

---

## What Is NOT Changed

| Component | Status | Reason |
|-----------|--------|--------|
| `merchant_category = "lending_app"` | Keep | This is **what happened** (classification), not a stress label |
| `txn_type = "auto_debit"` + `payment_status = "failed"` | Keep | Raw facts about the transaction |
| `payment_status = "failed"` | Keep | Factual outcome, not a stress label |
| `counterparty_id`, `counterparty_name` | Keep | The model can learn patterns from these |
| `balance_before`, `balance_after` | Keep | Factual balance tracking |
| Feature pipeline Z-score computation | New | Core of this change |

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `data_warehouse/schemas/migrate_003_customer_baseline.sql` | **NEW** — baseline table |
| `scripts/build_baselines.py` | **NEW** — baseline computation script |
| `ingestion/schemas/transaction_event.py` | Remove `is_lending_app_upi`, `is_auto_debit_failed`, `is_p2p_transfer`, `is_investment_txn` |
| `sagemaker/inference.py` | Replace hardcoded flag thresholds with per-customer Z-scores |
| `ingestion/consumers/feature_pipeline.py` | Load baseline and pass to `build_feature_vector()` |
| `models/training_pipelines/build_training_data.py` | Add Z-score feature simulation |
| `models/lightgbm/train_lgbm.py` | Add new feature names to `FEATURE_COLS` |

---

*This plan was written by treating each customer as their own reference population of one —
the only meaningful benchmark for personal financial stress detection.*
