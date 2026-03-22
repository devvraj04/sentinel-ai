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

    # P2P — infer from UPI debits that aren't to known merchants
    p2p = df[
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
