"""
models/lightgbm/train_lgbm.py
──────────────────────────────────────────────────────────────────────────────
Implements ALL formulas from the Mathematical & Economic Framework:

PREDICTIVE MODEL:
  PD = 1 / (1 + e^(-z))   where z = w^T X + b   (Logistic Output Layer)
  L  = -Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]  (Binary Cross Entropy)

THRESHOLD SELECTION:
  ECL = PD × LGD × EAD  →  threshold minimizes ECL, not accuracy
  Priority_i = PD_i × EAD_i  →  high exposure + high risk = first intervention

DRIFT MONITORING:
  PSI = Σ (Actual_i - Expected_i) × ln(Actual_i / Expected_i)
  PSI > 0.25 → Model Recalibration Trigger

FAIRNESS METRIC:
  AIR = Approval_Rate_protected / Approval_Rate_reference
  AIR < 0.80 → Bias Review Required

SHAP EXPLAINABILITY:
  Every prediction includes SHAP contributions for all 7 signals
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report,
)
from sklearn.model_selection import StratifiedKFold

TRAINING_DATA = "models/training_data/training_dataset.parquet"
MODEL_OUT     = "models/lightgbm/lgbm_model.joblib"
SHAP_OUT      = "models/lightgbm/shap_explainer.joblib"

# LGD assumption (RBI norms for unsecured retail)
LGD = 0.45

FEATURE_COLS = [
    # 7 core signals
    "salary_delay_days",
    "salary_amount_drop_pct",
    "balance_wow_drop_pct",
    "upi_lending_spike_ratio",
    "upi_lending_total_amount",
    "utility_payment_latency",
    "discretionary_contraction",
    "discretionary_txn_count",
    "atm_withdrawal_spike",
    "atm_amount_spike",
    "failed_auto_debit_count",
    "failed_auto_debit_amount",
    "failed_utility_count",
    "credit_utilization_delta",
    # Drift scores (μ_short - μ_hist)/σ_hist
    "drift_salary",
    "drift_balance",
    "drift_lending",
    "drift_utility",
    "drift_discretionary",
    "drift_atm",
    "drift_auto_debit",
    "drift_credit_card",
    # Composite drift
    "composite_drift_score",
    # Early stress flags
    "flag_salary",
    "flag_balance",
    "flag_lending",
    "flag_utility",
    "flag_discretionary",
    "flag_atm",
    "flag_failed_debit",
    "total_stress_flags",
    # ECL
    "ead_estimate",
    # Context
    "total_txn_count",
    "total_txn_amount",
    "income_coverage_ratio",
    "monthly_income",
    "is_salaried",
    "is_self_employed",
    "is_mass_retail",
    "is_affluent",
]

# Human-readable signal names for SHAP output
SIGNAL_LABELS = {
    "salary_delay_days":         "Salary Delay (days)",
    "salary_amount_drop_pct":    "Salary Amount Drop (%)",
    "balance_wow_drop_pct":      "Savings Balance Decline",
    "upi_lending_spike_ratio":   "UPI Lending App Spike",
    "upi_lending_total_amount":  "Total Borrowed from Lending Apps",
    "utility_payment_latency":   "Utility Payment Latency (days)",
    "discretionary_contraction": "Discretionary Spend Contraction",
    "discretionary_txn_count":   "Discretionary Transaction Count",
    "atm_withdrawal_spike":      "ATM Withdrawal Frequency Spike",
    "atm_amount_spike":          "ATM Withdrawal Amount Spike",
    "failed_auto_debit_count":   "Failed Auto-Debit Attempts",
    "failed_auto_debit_amount":  "Failed Auto-Debit Amount",
    "failed_utility_count":      "Failed Utility Payments",
    "credit_utilization_delta":  "Credit Utilization Rise",
    "drift_salary":              "Salary Drift Score",
    "drift_balance":             "Balance Drift Score",
    "drift_lending":             "Lending App Drift Score",
    "drift_utility":             "Utility Drift Score",
    "drift_discretionary":       "Discretionary Drift Score",
    "drift_atm":                 "ATM Drift Score",
    "drift_auto_debit":          "Auto-Debit Drift Score",
    "drift_credit_card":         "Credit Card Drift Score",
    "composite_drift_score":     "Composite Behavioral Drift",
    "flag_salary":               "Early Flag: Salary",
    "flag_balance":              "Early Flag: Savings",
    "flag_lending":              "Early Flag: Lending UPI",
    "flag_utility":              "Early Flag: Utility",
    "flag_discretionary":        "Early Flag: Discretionary",
    "flag_atm":                  "Early Flag: ATM",
    "flag_failed_debit":         "Early Flag: Auto-Debit",
    "total_stress_flags":        "Total Stress Flags Triggered",
    "ead_estimate":              "Exposure at Default (EAD)",
    "total_txn_count":           "Transaction Volume",
    "total_txn_amount":          "Total Transaction Amount",
    "income_coverage_ratio":     "Income Coverage Ratio",
    "monthly_income":            "Monthly Income",
    "is_salaried":               "Employment: Salaried",
    "is_self_employed":          "Employment: Self-Employed",
    "is_mass_retail":            "Segment: Mass Retail",
    "is_affluent":               "Segment: Affluent",
}


# ══════════════════════════════════════════════════════════════════════════════
# ECL-based threshold selection
# ECL = PD × LGD × EAD  →  minimise total expected loss
# ══════════════════════════════════════════════════════════════════════════════
def select_ecl_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ead: np.ndarray,
    lgd: float = LGD,
) -> tuple[float, float]:
    """
    Selects classification threshold by minimising Expected Credit Loss:
      ECL = PD × LGD × EAD
      Priority_i = PD_i × EAD_i

    Instead of maximising F1 or accuracy, we find the threshold where
    total portfolio ECL is minimised — aligning ML objective to business.
    """
    best_t, best_ecl = 0.5, float("inf")

    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)

        # False Negatives = missed defaulters (they default, no intervention)
        fn_mask = (pred == 0) & (y_true == 1)
        # Expected loss from missed cases = full PD × LGD × EAD
        fn_ecl = float(np.sum(y_prob[fn_mask] * lgd * ead[fn_mask]))

        # False Positives = unnecessary interventions (opportunity cost ~5%)
        fp_mask = (pred == 1) & (y_true == 0)
        fp_cost = float(np.sum(ead[fp_mask] * 0.05))

        total_ecl = fn_ecl + fp_cost

        if total_ecl < best_ecl:
            best_ecl, best_t = total_ecl, t

    return best_t, best_ecl


# ══════════════════════════════════════════════════════════════════════════════
# PSI — Population Stability Index
# PSI = Σ (Actual_i - Expected_i) × ln(Actual_i / Expected_i)
# ══════════════════════════════════════════════════════════════════════════════
def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 0.001
    bins[-1] += 0.001
    exp_pct = np.histogram(expected, bins=bins)[0] / max(len(expected), 1)
    act_pct = np.histogram(actual,   bins=bins)[0] / max(len(actual), 1)
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ══════════════════════════════════════════════════════════════════════════════
# AIR — Adverse Impact Ratio (Fairness Metric)
# AIR = Approval_Rate_protected / Approval_Rate_reference
# AIR < 0.80 → Bias Review Required
# ══════════════════════════════════════════════════════════════════════════════
def compute_air(
    df: pd.DataFrame,
    preds: np.ndarray,
    protected_col: str,
    protected_val,
    reference_val,
) -> float:
    """
    AIR = Approval_Rate_protected / Approval_Rate_reference
    Approval = predicted NOT at risk (pred == 0)
    """
    if protected_col not in df.columns:
        return 1.0

    mask_prot = df[protected_col] == protected_val
    mask_ref  = df[protected_col] == reference_val

    if mask_prot.sum() == 0 or mask_ref.sum() == 0:
        return 1.0

    rate_prot = float((preds[mask_prot] == 0).mean())
    rate_ref  = float((preds[mask_ref]  == 0).mean())

    return rate_prot / max(rate_ref, 0.001)


# ══════════════════════════════════════════════════════════════════════════════
# Priority Score — Priority_i = PD_i × EAD_i
# High exposure + high risk → first intervention
# ══════════════════════════════════════════════════════════════════════════════
def compute_priority(pd_probs: np.ndarray, ead: np.ndarray) -> np.ndarray:
    """Priority_i = PD_i × EAD_i — higher = intervene first."""
    return pd_probs * ead


def train():
    print("=" * 65)
    print("LightGBM — Sentinel Pre-Delinquency Risk Engine")
    print("Formulas: PD=sigmoid | ECL threshold | PSI | AIR | SHAP")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading {TRAINING_DATA}...")
    df = pd.read_parquet(TRAINING_DATA)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X   = df[feature_cols].fillna(0).astype(float)
    y   = df["label"].astype(int)
    ead = df["ead_estimate"].fillna(0).values if "ead_estimate" in df.columns \
          else np.ones(len(df)) * df.get("monthly_income", pd.Series([50000])).mean() * 12

    pos = int(y.sum())
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)

    print(f"  Samples  : {len(df):,}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Positive : {pos:,} ({pos/len(y):.1%})")
    print(f"  Negative : {neg:,} ({neg/len(y):.1%})")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    )
    mlflow.set_experiment("sentinel-lgbm-v3")

    with mlflow.start_run(run_name="lgbm_ecl_threshold_shap_air"):

        params = {
            "objective":         "binary",
            "metric":            "binary_logloss",  # Binary Cross-Entropy
            "boosting_type":     "gbdt",
            "num_leaves":        63,
            "max_depth":         8,
            "learning_rate":     0.05,
            "n_estimators":      600,
            "scale_pos_weight":  scale_pos_weight,
            "min_child_samples": 10,
            "feature_fraction":  0.8,
            "bagging_fraction":  0.8,
            "bagging_freq":      5,
            "reg_alpha":         0.1,
            "reg_lambda":        0.2,
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        mlflow.log_params({**params, "lgd": LGD, "n_features": len(feature_cols)})

        # ── 5-fold CV ─────────────────────────────────────────────────────────
        print("\n5-Fold Cross-Validation (Binary Cross-Entropy loss):")
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
            prob = m.predict_proba(X.iloc[val_idx])[:, 1]
            auc  = roc_auc_score(y.iloc[val_idx], prob)
            aucs.append(auc)
            print(f"  Fold {fold+1}  AUC: {auc:.4f}")

        cv_auc = float(np.mean(aucs))
        print(f"\n  CV AUC: {cv_auc:.4f} ± {np.std(aucs):.4f}")
        mlflow.log_metric("cv_auc_mean", cv_auc)
        mlflow.log_metric("cv_auc_std", float(np.std(aucs)))

        # ── Final model ───────────────────────────────────────────────────────
        print("\nTraining final model on full dataset...")
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
        proba = model.predict_proba(X)[:, 1]

        # ── ECL-based threshold selection ─────────────────────────────────────
        print("\nSelecting threshold using ECL = PD × LGD × EAD ...")
        ecl_threshold, min_ecl = select_ecl_threshold(
            y.values, proba, ead, lgd=LGD
        )
        print(f"  ECL-optimal threshold : {ecl_threshold:.2f}")
        print(f"  Minimum portfolio ECL : ₹{min_ecl:,.0f}")
        mlflow.log_metric("ecl_threshold", ecl_threshold)
        mlflow.log_metric("min_portfolio_ecl", min_ecl)

        # Also compute F1-optimal threshold for comparison
        best_f1_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            f = f1_score(y, (proba >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_f1_t = f, t
        print(f"  F1-optimal threshold  : {best_f1_t:.2f}  (shown for comparison)")
        print(f"  Using ECL threshold   : {ecl_threshold:.2f}  ← business-aligned")

        preds    = (proba >= ecl_threshold).astype(int)
        auc_full = roc_auc_score(y, proba)
        prec     = precision_score(y, preds, zero_division=0)
        rec      = recall_score(y, preds, zero_division=0)
        f1       = f1_score(y, preds, zero_division=0)

        print(f"\nFinal Model Metrics (ECL threshold={ecl_threshold:.2f}):")
        print(f"  AUC-ROC   : {auc_full:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}  ← critical: catch all defaulters")
        print(f"  F1        : {f1:.4f}")
        print(f"\n{classification_report(y, preds, target_names=['Healthy', 'At-Risk'])}")

        mlflow.log_metrics({
            "auc_roc": auc_full, "precision": prec,
            "recall": rec, "f1": f1,
            "ecl_threshold": ecl_threshold,
        })

        # ── PSI Monitoring ────────────────────────────────────────────────────
        print("Drift Monitoring — PSI = Σ(Actual - Expected) × ln(Actual/Expected):")
        if "month_offset" in df.columns:
            early = proba[df["month_offset"].values <= -3]
            late  = proba[df["month_offset"].values >= -1]
            if len(early) > 10 and len(late) > 10:
                psi = compute_psi(early, late)
                status = ("⚠  RECALIBRATION TRIGGER" if psi > 0.25
                          else "⚡ Minor drift" if psi > 0.10 else "✓  Stable")
                print(f"  PSI (early vs recent): {psi:.4f}  {status}")
                mlflow.log_metric("psi", psi)

        # ── AIR — Fairness Check ──────────────────────────────────────────────
        print("\nFairness — AIR = Approval_Rate_protected / Approval_Rate_reference:")
        air_checks = [
            ("is_salaried",    1.0, 0.0, "Salaried vs Self-Employed"),
            ("is_mass_retail", 1.0, 0.0, "Mass Retail vs Affluent"),
        ]
        all_air_pass = True
        for col, prot_val, ref_val, label in air_checks:
            air = compute_air(df, preds, col, prot_val, ref_val)
            status = "✓ PASS" if air >= 0.80 else "✗ BIAS REVIEW REQUIRED"
            print(f"  {label:<30}  AIR={air:.3f}  {status}")
            mlflow.log_metric(f"air_{col}", air)
            if air < 0.80:
                all_air_pass = False

        if all_air_pass:
            print("  All AIR checks passed (≥ 0.80)")

        # ── Priority Score (PD × EAD) ─────────────────────────────────────────
        priority = compute_priority(proba, ead)
        top_priority = pd.DataFrame({
            "customer_idx": range(len(proba)),
            "pd_probability": np.round(proba, 4),
            "ead": np.round(ead, 0),
            "priority_score": np.round(priority, 2),
            "risk_flag": preds,
        }).sort_values("priority_score", ascending=False).head(10)

        print("\nTop 10 Priority Customers (Priority = PD × EAD):")
        print("  (High exposure + high risk = intervene first)")
        print(f"  {'Index':<8} {'PD':>6} {'EAD':>10} {'Priority':>12} {'Flag':>6}")
        print("  " + "-" * 46)
        for _, row in top_priority.iterrows():
            print(f"  {int(row['customer_idx']):<8} "
                  f"{row['pd_probability']:>6.3f} "
                  f"₹{row['ead']:>9,.0f} "
                  f"{row['priority_score']:>12,.1f} "
                  f"{'AT-RISK' if row['risk_flag'] else 'OK':>6}")

        # ── Feature Importance ────────────────────────────────────────────────
        imp = pd.DataFrame({
            "feature":   feature_cols,
            "importance":model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\nTop 12 Feature Importances:")
        print(f"  {'Feature':<35} {'Importance':>10}  Signal")
        print("  " + "-" * 65)
        for _, row in imp.head(12).iterrows():
            label = SIGNAL_LABELS.get(row["feature"], row["feature"])
            print(f"  {row['feature']:<35} {row['importance']:>10.0f}  {label}")

        # ── SHAP Explainability ───────────────────────────────────────────────
        print("\nBuilding SHAP explainer (TreeExplainer)...")
        explainer = shap.TreeExplainer(model)
        sample_X  = X.sample(min(300, len(X)), random_state=42)
        shap_vals = explainer.shap_values(sample_X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_mean = pd.DataFrame({
            "feature":       feature_cols,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        print("\nMean |SHAP| — Contribution of each signal to risk score:")
        print(f"  {'Feature':<35} {'|SHAP|':>8}  Bar")
        print("  " + "-" * 65)
        max_shap = shap_mean["mean_abs_shap"].max()
        for _, row in shap_mean.head(12).iterrows():
            bar_len = int(row["mean_abs_shap"] / max_shap * 30)
            bar     = "█" * bar_len
            label   = SIGNAL_LABELS.get(row["feature"], row["feature"])
            print(f"  {row['feature']:<35} {row['mean_abs_shap']:>8.4f}  {bar}")

        # ── Save ──────────────────────────────────────────────────────────────
        os.makedirs("models/lightgbm", exist_ok=True)
        package = {
            "model":             model,
            "threshold":         ecl_threshold,    # ECL-optimal, not F1
            "f1_threshold":      best_f1_t,
            "feature_cols":      feature_cols,
            "scale_pos_weight":  scale_pos_weight,
            "version":           "3.0.0",
            "cv_auc":            cv_auc,
            "lgd":               LGD,
            "min_ecl":           min_ecl,
            "signal_labels":     SIGNAL_LABELS,
            "drift_thresholds":  {
                "salary_days": 1.5, "balance": 1.5, "lending_upi": 2.0,
                "utility": 1.5, "discretionary": 1.5, "atm": 2.0, "auto_debit": 1.0,
            },
        }
        joblib.dump(package,   MODEL_OUT)
        joblib.dump(explainer, SHAP_OUT)
        mlflow.log_artifact(MODEL_OUT)

        print(f"\n  ✓ Model   → {MODEL_OUT}")
        print(f"  ✓ SHAP    → {SHAP_OUT}")
        print(f"  ✓ MLflow  → http://localhost:5001")

    print("\n" + "=" * 65)
    print(f"Training Complete")
    print(f"  CV AUC      : {cv_auc:.4f}")
    print(f"  ECL Thresh  : {ecl_threshold:.2f}  (minimises PD×LGD×EAD)")
    print(f"  AIR Check   : {'PASS' if all_air_pass else 'REVIEW NEEDED'}")
    print("=" * 65)
    return cv_auc


if __name__ == "__main__":
    train()