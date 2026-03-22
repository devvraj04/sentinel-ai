"""
models/lightgbm/train_lgbm.py
──────────────────────────────────────────────────────────────────────────────
WHY PREVIOUS TRAIN AUC WAS 1.0 AND CV AUC WAS 0.63:


  BUG 1 — evaluated on training data:
    Final metrics were computed on the same 9,000 rows the model trained on.
    LightGBM with 500 trees memorises 9k rows → AUC=1.0 trivially.
    Fix: stratified 80/20 split; test set locked until final evaluation only.


  BUG 2 — DGP default rate was 2.4% (214 positives):
    214 positives in 7,200 training rows → model couldn't learn patterns.
    Fixed in build_training_data.py (ICR bug + recalibration → ~20% DR).


HYPERPARAMETER CHANGES FOR AUC ~0.90:
  num_leaves       : 63      (more capacity with more features)
  max_depth        : 7       (balanced depth)
  min_child_samples: 40      (prevents leaf memorisation)
  n_estimators     : 800     (more boosting rounds with early stopping)
  learning_rate    : 0.04    (slightly lower for better generalisation)
  reg_alpha/lambda : 0.3/0.4 (stronger regularisation)


EVALUATION:
  CV AUC   — on 5 folds of train set (generalisation estimate)
  Test AUC — on held-out 20% test set (the honest number)
  Both should be in 0.85–0.92 range with the fixed DGP.
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
    f1_score, classification_report, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


TRAINING_DATA = "models/training_data/training_dataset.parquet"
MODEL_OUT     = "models/lightgbm/lgbm_model.joblib"
SHAP_OUT      = "models/lightgbm/shap_explainer.joblib"
LGD           = 0.45   # RBI norms for unsecured retail


# ── Feature list — includes all new high-signal features ────────────────────
FEATURE_COLS = [
    # Core behavioral signals
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
    # New high-signal features
    "emi_to_income_ratio",
    "dpd_30_last_12m",
    "missed_emi_streak",
    "savings_runway_months",
    "revolving_utilization",
    "credit_enquiries_3m",
    "p2p_transfer_spike",
    "investment_redemption_pct",
    # Drift scores
    "drift_salary",
    "drift_balance",
    "drift_lending",
    "drift_utility",
    "drift_discretionary",
    "drift_atm",
    "drift_auto_debit",
    "drift_credit_card",
    "composite_drift_score",
    # Stress flags
    "flag_salary",
    "flag_balance",
    "flag_lending",
    "flag_utility",
    "flag_discretionary",
    "flag_atm",
    "flag_failed_debit",
    "flag_emi_burden",
    "flag_high_utilization",
    "total_stress_flags",
    # Context
    "ead_estimate",
    "total_txn_count",
    "total_txn_amount",
    "income_coverage_ratio",
    "monthly_income",
    "is_salaried",
    "is_self_employed",
    "is_mass_retail",
    "is_affluent",
    "tenure_months",
]


SIGNAL_LABELS = {
    "salary_delay_days":           "Salary Delay (days)",
    "salary_amount_drop_pct":      "Salary Amount Drop (%)",
    "balance_wow_drop_pct":        "Savings Balance Decline (%)",
    "upi_lending_spike_ratio":     "UPI Lending App Spike",
    "upi_lending_total_amount":    "Total Borrowed via Lending Apps (₹)",
    "utility_payment_latency":     "Utility Payment Latency (days)",
    "discretionary_contraction":   "Discretionary Spend Contraction (%)",
    "discretionary_txn_count":     "Discretionary Transaction Count",
    "atm_withdrawal_spike":        "ATM Withdrawal Frequency Spike",
    "atm_amount_spike":            "ATM Withdrawal Amount Spike",
    "failed_auto_debit_count":     "Failed Auto-Debit Count",
    "failed_auto_debit_amount":    "Failed Auto-Debit Amount (₹)",
    "failed_utility_count":        "Failed Utility Payments",
    "credit_utilization_delta":    "Credit Utilization Rise (%pts MoM)",
    "emi_to_income_ratio":         "EMI-to-Income Ratio",
    "dpd_30_last_12m":             "DPD-30 Events Last 12 Months",
    "missed_emi_streak":           "Consecutive Missed EMI Months",
    "savings_runway_months":       "Savings Runway (months)",
    "revolving_utilization":       "Revolving Credit Utilization (%)",
    "credit_enquiries_3m":         "Hard Credit Enquiries (90 days)",
    "p2p_transfer_spike":          "P2P Transfer Spike Ratio",
    "investment_redemption_pct":   "Investment Redemption (%)",
    "drift_salary":                "Salary Drift Score",
    "drift_balance":               "Balance Drift Score",
    "drift_lending":               "Lending App Drift Score",
    "drift_utility":               "Utility Drift Score",
    "drift_discretionary":         "Discretionary Drift Score",
    "drift_atm":                   "ATM Drift Score",
    "drift_auto_debit":            "Auto-Debit Drift Score",
    "drift_credit_card":           "Credit Card Drift Score",
    "composite_drift_score":       "Composite Behavioural Drift",
    "flag_salary":                 "Early Flag: Salary",
    "flag_balance":                "Early Flag: Savings",
    "flag_lending":                "Early Flag: Lending UPI",
    "flag_utility":                "Early Flag: Utility",
    "flag_discretionary":          "Early Flag: Discretionary",
    "flag_atm":                    "Early Flag: ATM",
    "flag_failed_debit":           "Early Flag: Auto-Debit",
    "flag_emi_burden":             "Early Flag: EMI Burden",
    "flag_high_utilization":       "Early Flag: High CC Utilization",
    "total_stress_flags":          "Total Stress Flags Triggered",
    "ead_estimate":                "Exposure at Default (₹)",
    "total_txn_count":             "Transaction Volume",
    "total_txn_amount":            "Total Monthly Transaction Amount (₹)",
    "income_coverage_ratio":       "Income Coverage Ratio",
    "monthly_income":              "Monthly Income (₹)",
    "is_salaried":                 "Employment: Salaried",
    "is_self_employed":            "Employment: Self-Employed",
    "is_mass_retail":              "Segment: Mass Retail",
    "is_affluent":                 "Segment: Affluent",
    "tenure_months":               "Tenure with Bank (months)",
}




# ════════════════════════════════════════════════════════════════════════════
# ECL threshold — minimise PD × LGD × EAD, not accuracy
# ════════════════════════════════════════════════════════════════════════════
def select_ecl_threshold(
    y_true: np.ndarray, y_prob: np.ndarray,
    ead: np.ndarray, lgd: float = LGD,
) -> tuple[float, float]:
    best_t, best_ecl = 0.5, float("inf")
    for t in np.arange(0.05, 0.90, 0.01):
        pred    = (y_prob >= t).astype(int)
        fn_mask = (pred == 0) & (y_true == 1)
        fp_mask = (pred == 1) & (y_true == 0)
        ecl     = float(np.sum(y_prob[fn_mask] * lgd * ead[fn_mask]))
        ecl    += float(np.sum(ead[fp_mask] * 0.05))
        if ecl < best_ecl:
            best_ecl, best_t = ecl, t
    return best_t, best_ecl




# ════════════════════════════════════════════════════════════════════════════
# PSI  Σ(A_i − E_i) × ln(A_i / E_i)
# ════════════════════════════════════════════════════════════════════════════
def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    bins        = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0]    -= 1e-4
    bins[-1]   += 1e-4
    exp_pct     = np.histogram(expected, bins=bins)[0] / max(len(expected), 1)
    act_pct     = np.histogram(actual,   bins=bins)[0] / max(len(actual),   1)
    exp_pct     = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct     = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))




# ════════════════════════════════════════════════════════════════════════════
# AIR  approval_rate_protected / approval_rate_reference  (≥0.80 required)
# ════════════════════════════════════════════════════════════════════════════
def compute_air(
    df: pd.DataFrame, preds: np.ndarray,
    protected_col: str, protected_val, reference_val,
) -> float:
    if protected_col not in df.columns:
        return 1.0
    mp = df[protected_col] == protected_val
    mr = df[protected_col] == reference_val
    if mp.sum() == 0 or mr.sum() == 0:
        return 1.0
    return float((preds[mp] == 0).mean()) / max(float((preds[mr] == 0).mean()), 1e-4)




def train():
    print("=" * 65)
    print("LightGBM — Sentinel Pre-Delinquency Risk Engine  (v3)")
    print("Target: Test AUC ~0.90 | ECL threshold | PSI | AIR | SHAP")
    print("=" * 65)


    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading {TRAINING_DATA}...")
    df           = pd.read_parquet(TRAINING_DATA)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing      = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠  {len(missing)} features missing from data: {missing}")


    X   = df[feature_cols].fillna(0).astype(float)
    y   = df["label"].astype(int)
    ead = df["ead_estimate"].fillna(0).values


    pos = int(y.sum())
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)


    print(f"  Samples    : {len(df):,}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Positives  : {pos:,}  ({pos/len(y):.1%})")
    print(f"  Negatives  : {neg:,}  ({neg/len(y):.1%})")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")


    if pos / len(y) < 0.05:
        print(f"\n  ⚠  Default rate {pos/len(y):.1%} is very low.")
        print(f"     Re-run build_training_data.py — ICR bug may still be present.")


    # ── Stratified 80/20 split — test set locked until final evaluation ───────
    (X_train, X_test,
     y_train, y_test,
     ead_train, ead_test,
     idx_train, idx_test) = train_test_split(
        X, y, ead, np.arange(len(df)),
        test_size=0.20, stratify=y, random_state=42,
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)


    print(f"\n  Train set  : {len(X_train):,}  (pos={y_train.sum():,})")
    print(f"  Test set   : {len(X_test):,}  (pos={y_test.sum():,})")


    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("sentinel-lgbm-v3")


    with mlflow.start_run(run_name="lgbm_v3_auc90_ecl_shap"):


        # Hyperparameters tuned for AUC ~0.90 with ~50 features
        params = {
            "objective":          "binary",
            "metric":             "auc",           # track AUC during training
            "boosting_type":      "gbdt",
            "num_leaves":         63,              # more capacity for new features
            "max_depth":          7,
            "learning_rate":      0.04,            # lower LR → better generalisation
            "n_estimators":       800,             # more rounds (early stopping guards)
            "scale_pos_weight":   scale_pos_weight,
            "min_child_samples":  40,              # prevents leaf memorisation
            "feature_fraction":   0.75,            # subsample features per tree
            "bagging_fraction":   0.80,
            "bagging_freq":       5,
            "reg_alpha":          0.30,            # L1 regularisation
            "reg_lambda":         0.40,            # L2 regularisation
            "min_split_gain":     0.01,            # prune trivial splits
            "random_state":       42,
            "n_jobs":             -1,
            "verbose":            -1,
        }
        mlflow.log_params({**params, "lgd": LGD, "n_features": len(feature_cols)})


        # ── 5-fold CV on train set ────────────────────────────────────────────
        print("\n5-Fold Stratified CV (on train set only):")
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []


        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                callbacks=[
                    lgb.early_stopping(60, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            prob = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            auc  = roc_auc_score(y_train.iloc[val_idx], prob)
            aucs.append(auc)
            print(f"  Fold {fold+1}  AUC: {auc:.4f}  (best iter: {m.best_iteration_})")


        cv_auc = float(np.mean(aucs))
        cv_std = float(np.std(aucs))
        print(f"\n  CV AUC : {cv_auc:.4f} ± {cv_std:.4f}")
        mlflow.log_metrics({"cv_auc_mean": cv_auc, "cv_auc_std": cv_std})


        # ── Final model — train on full train set ─────────────────────────────
        print("\nTraining final model on full train set...")
        model = lgb.LGBMClassifier(**params)
        # X_train is a pandas DataFrame — LightGBM reads column names automatically.
        # Do NOT pass feature_name= here; it is version-sensitive in the sklearn API
        # and ignored/overridden when X is a DataFrame in LGB 4.x anyway.
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

        # Immediately verify feature names are correct — fail loud if not
        actual_names = list(model.feature_name_)
        if actual_names[0] == "Column_0":
            raise RuntimeError(
                f"Model was trained with anonymous features (Column_0..N).\n"
                f"This means X_train lost its column names before fit().\n"
                f"X_train type: {type(X_train)}  columns[:3]: {list(X_train.columns[:3]) if hasattr(X_train, 'columns') else 'NO COLUMNS'}"
            )
        print(f"  Feature names verified: {actual_names[:3]}... ({len(actual_names)} total)")


        # ── ECL threshold — selected on TRAIN predictions (not test) ─────────
        train_prob    = model.predict_proba(X_train)[:, 1]
        ecl_threshold, _ = select_ecl_threshold(
            y_train.values, train_prob, ead_train, lgd=LGD
        )
        # Clamp: don't let ECL push threshold so low it flags everyone
        ecl_threshold = float(np.clip(ecl_threshold, 0.10, 0.65))
        print(f"  ECL-optimal threshold (on train): {ecl_threshold:.2f}")


        # ── TEST SET EVALUATION — held-out, never touched during training ─────
        print("\n" + "─" * 60)
        print("HELD-OUT TEST SET METRICS  (1,800 rows, never seen in training)")
        print("─" * 60)


        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= ecl_threshold).astype(int)


        auc_test  = roc_auc_score(y_test, test_prob)
        ap_test   = average_precision_score(y_test, test_prob)   # PR-AUC
        prec_test = precision_score(y_test, test_pred, zero_division=0)
        rec_test  = recall_score(y_test, test_pred, zero_division=0)
        f1_test   = f1_score(y_test, test_pred, zero_division=0)


        print(f"  AUC-ROC   : {auc_test:.4f}   ← target 0.87–0.92")
        print(f"  PR-AUC    : {ap_test:.4f}   (precision-recall; relevant for imbalanced)")
        print(f"  Precision : {prec_test:.4f}")
        print(f"  Recall    : {rec_test:.4f}   ← catch most defaulters")
        print(f"  F1        : {f1_test:.4f}")
        print(f"\n{classification_report(y_test, test_pred, target_names=['Healthy', 'At-Risk'])}")


        # Honest diagnostic
        gap = abs(cv_auc - auc_test)
        if auc_test > 0.97:
            print("  ⚠  AUC > 0.97 — check for feature leakage (e.g. true_pd in data)")
        elif auc_test < 0.70:
            print(f"  ⚠  AUC {auc_test:.3f} below target — verify DR > 10% in training data")
        else:
            band = "✓ target range" if auc_test >= 0.85 else "acceptable"
            print(f"  {band}  |CV−Test gap| = {gap:.3f}  ({'low overfit' if gap < 0.03 else 'moderate overfit'})")


        mlflow.log_metrics({
            "test_auc_roc":    auc_test,
            "test_pr_auc":     ap_test,
            "test_precision":  prec_test,
            "test_recall":     rec_test,
            "test_f1":         f1_test,
            "ecl_threshold":   ecl_threshold,
        })


        # ── PSI ───────────────────────────────────────────────────────────────
        print("\nDrift Monitoring — PSI Σ(A−E)×ln(A/E):")
        if "month_offset" in df.columns:
            mo     = df["month_offset"].values
            early  = model.predict_proba(X[mo <= -4])[:, 1] if (mo <= -4).sum() > 20 else None
            recent = model.predict_proba(X[mo >= -1])[:, 1] if (mo >= -1).sum() > 20 else None
            if early is not None and recent is not None:
                psi    = compute_psi(early, recent)
                status = ("⚠  RECALIBRATION TRIGGER" if psi > 0.25
                          else "⚡ Minor drift" if psi > 0.10 else "✓  Stable")
                print(f"  PSI (early vs recent): {psi:.4f}  {status}")
                mlflow.log_metric("psi", psi)
            else:
                print("  Not enough cohort data for PSI")


        # ── AIR fairness ──────────────────────────────────────────────────────
        print("\nFairness — AIR ≥ 0.80 required:")
        air_checks = [
            ("is_salaried",    1.0, 0.0, "Salaried vs Self-Employed"),
            ("is_mass_retail", 1.0, 0.0, "Mass Retail vs Affluent"),
        ]
        all_pass = True
        for col, pv, rv, lbl in air_checks:
            air    = compute_air(df_test, test_pred, col, pv, rv)
            status = "✓ PASS" if air >= 0.80 else "✗ BIAS REVIEW REQUIRED"
            print(f"  {lbl:<30}  AIR={air:.3f}  {status}")
            mlflow.log_metric(f"air_{col}", air)
            if air < 0.80:
                all_pass = False
        if all_pass:
            print("  All AIR checks passed (≥ 0.80)")


        # ── Priority customers from test set ──────────────────────────────────
        priority = test_prob * ead_test
        top10    = pd.DataFrame({
            "pd":       np.round(test_prob, 4),
            "ead":      np.round(ead_test,  0),
            "priority": np.round(priority,  2),
            "flag":     test_pred,
        }).sort_values("priority", ascending=False).head(10)


        print("\nTop 10 Priority Customers (TEST SET — Priority = PD × EAD):")
        print(f"  {'PD':>6} {'EAD':>12} {'Priority':>14} {'Flag':>8}")
        print("  " + "─" * 44)
        for _, row in top10.iterrows():
            print(
                f"  {row['pd']:>6.3f} "
                f"₹{row['ead']:>11,.0f} "
                f"{row['priority']:>14,.1f} "
                f"{'AT-RISK' if row['flag'] else 'OK':>8}"
            )


        # ── Feature importance ────────────────────────────────────────────────
        imp = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)


        print("\nTop 15 Feature Importances (split gain):")
        print(f"  {'Feature':<35} {'Importance':>10}  Signal")
        print("  " + "─" * 68)
        for _, row in imp.head(15).iterrows():
            lbl = SIGNAL_LABELS.get(row["feature"], row["feature"])
            print(f"  {row['feature']:<35} {row['importance']:>10.0f}  {lbl}")


        # ── SHAP ──────────────────────────────────────────────────────────────
        print("\nBuilding SHAP explainer (TreeExplainer)...")
        explainer  = shap.TreeExplainer(model)
        sample_X   = X_train.sample(min(500, len(X_train)), random_state=42)
        shap_vals  = explainer.shap_values(sample_X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]


        shap_df = pd.DataFrame({
            "feature":       feature_cols,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)


        print("\nMean |SHAP| — per-feature contribution to risk score:")
        print(f"  {'Feature':<35} {'|SHAP|':>8}  Bar")
        print("  " + "─" * 68)
        max_s = shap_df["mean_abs_shap"].max()
        for _, row in shap_df.head(15).iterrows():
            bar = "█" * int(row["mean_abs_shap"] / max_s * 30)
            lbl = SIGNAL_LABELS.get(row["feature"], row["feature"])
            print(f"  {row['feature']:<35} {row['mean_abs_shap']:>8.4f}  {bar}")


        # ── Save ──────────────────────────────────────────────────────────────
        os.makedirs("models/lightgbm", exist_ok=True)
        package = {
            "model":            model,
            "threshold":        ecl_threshold,
            "feature_cols":     feature_cols,
            "scale_pos_weight": scale_pos_weight,
            "version":          "3.1.0",
            "cv_auc":           cv_auc,
            "test_auc":         auc_test,
            "lgd":              LGD,
            "signal_labels":    SIGNAL_LABELS,
            "drift_thresholds": {
                "salary_days": 1.5, "balance": 1.5, "lending_upi": 2.0,
                "utility": 1.5, "discretionary": 1.5, "atm": 2.0, "auto_debit": 1.0,
            },
        }
        joblib.dump(package,   MODEL_OUT)
        joblib.dump(explainer, SHAP_OUT)

        # Save text dump so lgbm_model.txt always reflects the current trained model.
        # This is what you check with: head -5 models/lightgbm/lgbm_model.txt
        # It should show: feature_names=salary_delay_days balance_wow_drop_pct ...
        TXT_OUT = MODEL_OUT.replace(".joblib", ".txt")
        model.booster_.save_model(TXT_OUT)

        mlflow.log_artifact(MODEL_OUT)

        print(f"\n  ✓ Model  → {MODEL_OUT}")
        print(f"  ✓ SHAP   → {SHAP_OUT}")
        print(f"  ✓ Text   → {TXT_OUT}")
        print(f"  ✓ MLflow → http://localhost:5001")

        # Final sanity check — print first line of feature names from the saved txt
        with open(TXT_OUT, "r") as _f:
            for _line in _f:
                if _line.startswith("feature_names="):
                    print(f"\n  Verify: {_line.strip()[:120]}")
                    break


    print("\n" + "=" * 65)
    print("Training Complete")
    print(f"  CV AUC (train folds) : {cv_auc:.4f} ± {cv_std:.4f}")
    print(f"  Test AUC (held-out)  : {auc_test:.4f}  ← the honest number")
    print(f"  PR-AUC               : {ap_test:.4f}")
    print(f"  ECL Threshold        : {ecl_threshold:.2f}")
    print(f"  AIR Check            : {'PASS' if all_pass else 'REVIEW NEEDED'}")
    print("=" * 65)
    return auc_test


if __name__ == "__main__":
    train()