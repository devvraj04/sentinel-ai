"""
sagemaker/train.py
──────────────────────────────────────────────────────────────────────────────
This script runs INSIDE SageMaker's managed training container.
SageMaker automatically:
  - Downloads training data from S3 to /opt/ml/input/data/
  - Runs this script
  - Uploads everything saved to /opt/ml/model/ back to S3

Do NOT import config.settings here — this runs in a clean AWS container.
All config comes from environment variables or hyperparameters.
──────────────────────────────────────────────────────────────────────────────
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

# ── SageMaker paths (fixed — do not change these) ─────────────────────────────
INPUT_DATA_DIR = "/opt/ml/input/data/training"   # SageMaker downloads S3 data here
MODEL_DIR      = "/opt/ml/model"                 # SageMaker uploads this to S3
OUTPUT_DIR     = "/opt/ml/output"

# ── Hyperparameters (passed from launch script) ────────────────────────────────
def get_hyperparams():
    params_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            return json.load(f)
    return {}

hp = get_hyperparams()
LGD          = float(hp.get("lgd", 0.45))
N_ESTIMATORS = int(hp.get("n_estimators", 600))
NUM_LEAVES   = int(hp.get("num_leaves", 63))
LEARNING_RATE= float(hp.get("learning_rate", 0.05))

FEATURE_COLS = [
    "salary_delay_days", "salary_amount_drop_pct", "balance_wow_drop_pct",
    "upi_lending_spike_ratio", "upi_lending_total_amount",
    "utility_payment_latency", "discretionary_contraction",
    "discretionary_txn_count", "atm_withdrawal_spike", "atm_amount_spike",
    "failed_auto_debit_count", "failed_auto_debit_amount", "failed_utility_count",
    "credit_utilization_delta", "drift_salary", "drift_balance",
    "drift_lending", "drift_utility", "drift_discretionary",
    "drift_atm", "drift_auto_debit", "drift_credit_card",
    "composite_drift_score", "flag_salary", "flag_balance",
    "flag_lending", "flag_utility", "flag_discretionary",
    "flag_atm", "flag_failed_debit", "total_stress_flags",
    "ead_estimate", "total_txn_count", "total_txn_amount",
    "income_coverage_ratio", "monthly_income",
    "is_salaried", "is_self_employed", "is_mass_retail",
]

def select_ecl_threshold(y_true, y_prob, ead, lgd=LGD):
    best_t, best_ecl = 0.5, float("inf")
    for t in np.arange(0.05, 0.95, 0.01):
        pred   = (y_prob >= t).astype(int)
        fn_ecl = float(np.sum(y_prob[(pred==0)&(y_true==1)] * lgd * ead[(pred==0)&(y_true==1)]))
        fp_cost= float(np.sum(ead[(pred==1)&(y_true==0)] * 0.05))
        total  = fn_ecl + fp_cost
        if total < best_ecl:
            best_ecl, best_t = total, t
    return best_t, best_ecl

if __name__ == "__main__":
    print("=" * 60)
    print("Sentinel LightGBM Training — SageMaker")
    print("=" * 60)

    # Load data from SageMaker input path
    data_file = os.path.join(INPUT_DATA_DIR, "training_dataset.parquet")
    print(f"\nLoading: {data_file}")
    df = pd.read_parquet(data_file)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X   = df[feature_cols].fillna(0).astype(float)
    y   = df["label"].astype(int)
    ead = df["ead_estimate"].fillna(0).values

    pos = int(y.sum())
    neg = len(y) - pos
    spw = neg / max(pos, 1)

    print(f"  Samples  : {len(df):,}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Positive : {pos:,} ({pos/len(y):.1%})")
    print(f"  scale_pos_weight: {spw:.2f}")

    # Train
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "num_leaves": NUM_LEAVES,
        "max_depth": 8, "learning_rate": LEARNING_RATE,
        "n_estimators": N_ESTIMATORS, "scale_pos_weight": spw,
        "min_child_samples": 10, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.2,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }

    # 5-fold CV
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        auc = roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:,1])
        aucs.append(auc)
        print(f"  Fold {fold+1}  AUC: {auc:.4f}")

    cv_auc = float(np.mean(aucs))
    print(f"\n  CV AUC: {cv_auc:.4f}")

    # Final model
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    proba = model.predict_proba(X)[:, 1]

    ecl_threshold, min_ecl = select_ecl_threshold(y.values, proba, ead)
    print(f"  ECL threshold: {ecl_threshold:.2f}")
    print(f"  AUC-ROC: {roc_auc_score(y, proba):.4f}")

    # SHAP
    explainer  = shap.TreeExplainer(model)

    # Save model package to SageMaker model dir
    os.makedirs(MODEL_DIR, exist_ok=True)
    package = {
        "model": model, "threshold": ecl_threshold,
        "feature_cols": feature_cols, "scale_pos_weight": spw,
        "version": "3.0.0-sagemaker", "cv_auc": cv_auc,
        "lgd": LGD, "min_ecl": min_ecl,
    }
    joblib.dump(package,   os.path.join(MODEL_DIR, "lgbm_model.joblib"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.joblib"))
    print(f"\n  ✓ Model saved to {MODEL_DIR}")
    print("  SageMaker will upload this to S3 automatically.")