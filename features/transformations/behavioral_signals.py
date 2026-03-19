"""
features/transformations/behavioral_signals.py
──────────────────────────────────────────────────────────────────────────────
Computes the 8 core behavioral stress signals from raw transaction history.
 
KEY DESIGN PRINCIPLE:
  Signals are computed as deviations from each customer's OWN historical
  baseline — not from population averages. If a customer normally keeps
  ₹500 in savings, ₹200 IS alarming. If they keep ₹50,000, ₹20,000 IS
  alarming at a different scale.
 
Usage:
  engine = BehavioralSignalEngine()
  signals = engine.compute(customer_id, transactions_df)
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
 
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
 
 
@dataclass
class BehavioralSignals:
    """Computed behavioral signals for one customer at one point in time."""
    customer_id:                    str
    computed_at:                    datetime
 
    # ── The 8 Core Signals ────────────────────────────────────────────────────
    salary_delay_days:              float = 0.0   # Days delayed vs expected
    balance_wow_drop_pct:           float = 0.0   # Week-over-week savings drop %
    upi_lending_spike_ratio:        float = 1.0   # vs 4-week avg (1.0 = normal)
    utility_payment_latency:        float = 0.0   # Days into billing cycle
    discretionary_contraction:      float = 1.0   # vs 4-week avg (1.0 = normal)
    atm_withdrawal_spike:           float = 1.0   # vs 4-week avg
    failed_auto_debit_count:        int   = 0     # In last 14 days
    credit_utilization_delta:       float = 0.0   # Current - 30-day avg
 
    # ── Drift Score (composite early warning) ─────────────────────────────────
    drift_score:                    float = 0.0   # Standardized composite drift
 
    # ── Per-signal drift values (for SHAP explainability) ─────────────────────
    signal_drifts:                  dict  = field(default_factory=dict)
 
    def has_early_stress(self, threshold: float = 1.5) -> bool:
        return self.drift_score >= threshold
 
    def to_feature_vector(self) -> dict:
        """Convert to ML model input format."""
        return {
            "salary_delay_days":        self.salary_delay_days,
            "balance_wow_drop_pct":     self.balance_wow_drop_pct,
            "upi_lending_spike_ratio":  self.upi_lending_spike_ratio,
            "utility_payment_latency":  self.utility_payment_latency,
            "discretionary_contraction": self.discretionary_contraction,
            "atm_withdrawal_spike":     self.atm_withdrawal_spike,
            "failed_auto_debit_count":  float(self.failed_auto_debit_count),
            "credit_utilization_delta": self.credit_utilization_delta,
            "drift_score":              self.drift_score,
        }
 
 
class BehavioralSignalEngine:
    """
    Computes behavioral stress signals from a customer's transaction DataFrame.
 
    Expected DataFrame columns:
        txn_timestamp   (datetime, UTC)
        txn_type        (str)
        amount          (float)
        merchant_category (str)
        payment_status  (str)
        account_type    (str)
        is_lending_app_upi (bool)
        is_auto_debit_failed (bool)
    """
 
    # ── Configuration ─────────────────────────────────────────────────────────
    SHORT_WINDOW_DAYS: int = 14     # Recent behavior window
    HIST_WINDOW_DAYS: int  = 90     # Historical baseline window
    DRIFT_THETA: float     = 1.5    # Threshold: |drift| > theta = stress flag
 
    def compute(
        self,
        customer_id: str,
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> BehavioralSignals:
        """
        Main entry point. Compute all signals for a customer.
        reference_date defaults to now (UTC).
        """
        now = reference_date or datetime.now(timezone.utc)
        result = BehavioralSignals(customer_id=customer_id, computed_at=now)
 
        if df.empty:
            return result
 
        # Ensure timestamp is timezone-aware
        df = df.copy()
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
        df = df.sort_values("txn_timestamp")
 
        # Split into windows
        short_start = now - timedelta(days=self.SHORT_WINDOW_DAYS)
        hist_start  = now - timedelta(days=self.HIST_WINDOW_DAYS)
        df_short = df[df["txn_timestamp"] >= short_start]
        df_hist  = df[(df["txn_timestamp"] >= hist_start) & (df["txn_timestamp"] < short_start)]
 
        # ── Compute each signal ────────────────────────────────────────────────
        result.salary_delay_days       = self._salary_delay(df_short)
        result.balance_wow_drop_pct    = self._balance_drop(df_short)
        result.upi_lending_spike_ratio = self._upi_lending_spike(df_short, df_hist)
        result.utility_payment_latency = self._utility_latency(df_short)
        result.discretionary_contraction = self._discretionary_contraction(df_short, df_hist)
        result.atm_withdrawal_spike    = self._atm_spike(df_short, df_hist)
        result.failed_auto_debit_count = self._failed_auto_debits(df_short)
        result.credit_utilization_delta = self._credit_utilization_delta(df_short, df_hist)
 
        # ── Composite drift score ──────────────────────────────────────────────
        result.drift_score, result.signal_drifts = self._compute_drift(df_short, df_hist)
 
        return result
 
    # ── Signal implementations ─────────────────────────────────────────────────
 
    def _salary_delay(self, df_short: pd.DataFrame) -> float:
        """Days between expected salary date (1st-5th) and actual credit."""
        salaries = df_short[df_short["txn_type"] == "salary_credit"]
        if salaries.empty:
            return 5.0  # No salary received — worst case
 
        last_salary = salaries.iloc[-1]["txn_timestamp"]
        expected_day = 3  # Assume expected on 3rd of month
        actual_day = last_salary.day
        delay = max(0, actual_day - expected_day)
        return float(delay)
 
    def _balance_drop(self, df_short: pd.DataFrame) -> float:
        """Week-over-week savings balance decline as a fraction."""
        savings = df_short[df_short["account_type"] == "savings"].copy()
        if len(savings) < 2:
            return 0.0
 
        # Approximate balance from running sum of credits - debits
        savings["signed_amount"] = np.where(
            savings["txn_type"].isin(["salary_credit", "upi_credit", "neft_rtgs"]),
            savings["amount"],
            -savings["amount"],
        )
        savings = savings.sort_values("txn_timestamp")
        mid = len(savings) // 2
        week1_balance = savings.iloc[:mid]["signed_amount"].sum()
        week2_balance = savings.iloc[mid:]["signed_amount"].sum()
 
        if week1_balance == 0:
            return 0.0
        drop = (week1_balance - week2_balance) / abs(week1_balance)
        return float(np.clip(drop, -5.0, 5.0))
 
    def _upi_lending_spike(self, df_short: pd.DataFrame, df_hist: pd.DataFrame) -> float:
        """Ratio of UPI-to-lending-app spend this period vs historical average."""
        current = df_short[df_short["is_lending_app_upi"] == True]["amount"].sum()
        if df_hist.empty:
            return 1.0 if current == 0 else 3.0  # No baseline — treat spike as red flag
        historical_avg = df_hist[df_hist["is_lending_app_upi"] == True]["amount"].sum()
        historical_avg = historical_avg / (self.HIST_WINDOW_DAYS / self.SHORT_WINDOW_DAYS)
        if historical_avg < 1:
            return 1.0 if current < 100 else 5.0
        return float(np.clip(current / historical_avg, 0, 20))
 
    def _utility_latency(self, df_short: pd.DataFrame) -> float:
        """Average days into the month when utility bills are paid."""
        utilities = df_short[df_short["txn_type"] == "utility_payment"]
        if utilities.empty:
            return 0.0
        avg_day = utilities["txn_timestamp"].apply(lambda t: t.day).mean()
        return float(avg_day)
 
    def _discretionary_contraction(self, df_short: pd.DataFrame, df_hist: pd.DataFrame) -> float:
        """Ratio of discretionary spending now vs historical (lower = more stress)."""
        disc_cats = {"dining", "entertainment", "travel", "shopping"}
        current = df_short[df_short["merchant_category"].isin(disc_cats)]["amount"].sum()
        if df_hist.empty:
            return 1.0
        hist_sum = df_hist[df_hist["merchant_category"].isin(disc_cats)]["amount"].sum()
        hist_avg = hist_sum / (self.HIST_WINDOW_DAYS / self.SHORT_WINDOW_DAYS)
        if hist_avg < 1:
            return 1.0
        ratio = current / hist_avg
        return float(np.clip(ratio, 0, 5))
 
    def _atm_spike(self, df_short: pd.DataFrame, df_hist: pd.DataFrame) -> float:
        """Ratio of ATM withdrawals now vs historical avg."""
        current = df_short[df_short["txn_type"] == "atm_withdrawal"]["amount"].sum()
        if df_hist.empty:
            return 1.0
        hist_sum = df_hist[df_hist["txn_type"] == "atm_withdrawal"]["amount"].sum()
        hist_avg = hist_sum / (self.HIST_WINDOW_DAYS / self.SHORT_WINDOW_DAYS)
        if hist_avg < 1:
            return 1.0 if current < 500 else 3.0
        return float(np.clip(current / hist_avg, 0, 10))
 
    def _failed_auto_debits(self, df_short: pd.DataFrame) -> int:
        """Count of failed auto-debit attempts in the short window."""
        failed = df_short[
            (df_short["txn_type"] == "auto_debit") &
            (df_short["payment_status"] == "failed")
        ]
        return int(len(failed))
 
    def _credit_utilization_delta(self, df_short: pd.DataFrame, df_hist: pd.DataFrame) -> float:
        """
        Change in credit card utilization.
        Positive = utilization rising (stress signal).
        """
        cc_debits_now = df_short[df_short["account_type"] == "credit_card"]["amount"].sum()
        if df_hist.empty:
            return 0.0
        cc_hist = df_hist[df_hist["account_type"] == "credit_card"]["amount"].sum()
        cc_hist_avg = cc_hist / (self.HIST_WINDOW_DAYS / self.SHORT_WINDOW_DAYS)
        delta = cc_debits_now - cc_hist_avg
        # Normalize to a rough utilization delta (assuming ₹100k credit limit)
        assumed_limit = 100_000
        return float(np.clip(delta / assumed_limit, -1.0, 1.0))
 
    def _compute_drift(
        self,
        df_short: pd.DataFrame,
        df_hist: pd.DataFrame,
    ) -> tuple[float, dict]:
        """
        Compute standardized drift score using:
        Drift_f = (mu_short - mu_hist) / sigma_hist
        Composite = weighted average of |Drift_f| across all signals.
        """
        # Feature-level weekly amounts for drift computation
        feature_funcs = {
            "salary_delay":    lambda d: self._salary_delay(d),
            "balance_drop":    lambda d: max(0, self._balance_drop(d)),
            "upi_lending":     lambda d: self._upi_lending_spike(d, pd.DataFrame()),
            "atm_withdrawal":  lambda d: self._atm_spike(d, pd.DataFrame()),
            "util_latency":    lambda d: self._utility_latency(d),
        }
 
        # Split history into 14-day chunks for variance estimation
        now = df_hist["txn_timestamp"].max() if not df_hist.empty else datetime.now(timezone.utc)
        if pd.isna(now):
            now = datetime.now(timezone.utc)
 
        signal_drifts = {}
        drift_values = []
 
        for fname, func in feature_funcs.items():
            try:
                current_val = func(df_short)
                hist_val = func(df_hist) if not df_hist.empty else current_val
                # Simple drift: normalized deviation
                sigma = max(abs(hist_val) * 0.3, 0.1)  # Assume 30% std dev of mean
                drift = (current_val - hist_val) / sigma
                signal_drifts[fname] = float(drift)
                drift_values.append(abs(drift))
            except Exception:
                signal_drifts[fname] = 0.0
 
        # Composite: weighted mean of absolute drifts
        weights = [1.5, 1.5, 1.2, 1.0, 0.8]  # salary and balance weighted more
        composite = np.average(drift_values, weights=weights[:len(drift_values)]) if drift_values else 0.0
        return float(composite), signal_drifts
