"""
models/lightgbm/train_lgbm.py
──────────────────────────────────────────────────────────────────────────────
Trains LightGBM model on all 7 behavioral stress signals.
Uses class weights to handle imbalance (200 high-risk / 1500 total).
Logs to MLflow. Outputs SHAP explanations.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

TRAINING_DATA_PATH = "models/training_data/training_dataset.parquet"
MODEL_OUTPUT_PATH  = "models/lightgbm/lgbm_model.joblib"
SHAP_OUTPUT_PATH   = "models/lightgbm/shap_explainer.joblib"

FEATURE_COLS = [
    # 7 core signals
    "salary_delay_days",
    "salary_amount_drop",
    "balance_wow_drop_pct",
    "upi_lending_spike_ratio",
    "upi_lending_amount",
    "utility_payment_latency",
    "discretionary_contraction",
    "discretionary_txn_count",
    "atm_withdrawal_spike",
    "atm_amount_spike",
    "failed_auto_debit_count",
    "failed_auto_debit_amount",
    "credit_utilization_delta",
    # Composite
    "drift_score",
    # Context
    "total_txn_count",
    "total_txn_amount",
    "income_coverage_ratio",
    "monthly_income",
    "is_salaried",
    "is_mass_retail",
    "is_affluent",
    "month_offset",
]


def train():
    print("=" * 60)
    print("LightGBM Training — Sentinel Pre-Delinquency Model")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading training data from {TRAINING_DATA_PATH}...")
    df = pd.read_parquet(TRAINING_DATA_PATH)
    print(f"  Samples: {len(df):,}  |  Features: {len(FEATURE_COLS)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Filter to only columns that exist
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)

    # ── Class weight (handles imbalance) ─────────────────────────────────────
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"\n  Class balance — Positive: {pos_count:,}  Negative: {neg_count:,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("sentinel-lgbm")

    with mlflow.start_run(run_name="lgbm_v2_all7signals"):

        # ── Model params ──────────────────────────────────────────────────────
        params = {
            "objective":        "binary",
            "metric":           "auc",
            "boosting_type":    "gbdt",
            "num_leaves":       63,
            "max_depth":        8,
            "learning_rate":    0.05,
            "n_estimators":     500,
            "scale_pos_weight": scale_pos_weight,
            "min_child_samples":10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "reg_alpha":        0.1,
            "reg_lambda":       0.1,
            "random_state":     42,
            "n_jobs":           -1,
            "verbose":          -1,
        }
        mlflow.log_params(params)

        # ── Cross-validation ──────────────────────────────────────────────────
        print("\nRunning 5-fold cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            cv_aucs.append(auc)
            print(f"  Fold {fold+1}/5 — AUC: {auc:.4f}")

        mean_auc = np.mean(cv_aucs)
        print(f"\n  CV AUC: {mean_auc:.4f} ± {np.std(cv_aucs):.4f}")
        mlflow.log_metric("cv_auc_mean", mean_auc)
        mlflow.log_metric("cv_auc_std", np.std(cv_aucs))

        # ── Final model on full data ───────────────────────────────────────────
        print("\nTraining final model on full dataset...")
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

        # ── Metrics on full data (train metrics for logging) ──────────────────
        proba_full = final_model.predict_proba(X)[:, 1]

        # Find optimal threshold using F1
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (proba_full >= t).astype(int)
            f = f1_score(y, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh = f, t

        preds_final = (proba_full >= best_thresh).astype(int)
        auc_final   = roc_auc_score(y, proba_full)
        prec_final  = precision_score(y, preds_final, zero_division=0)
        rec_final   = recall_score(y, preds_final, zero_division=0)
        f1_final    = f1_score(y, preds_final, zero_division=0)

        print(f"\n  Final model metrics (threshold={best_thresh:.2f}):")
        print(f"    AUC-ROC   : {auc_final:.4f}")
        print(f"    Precision : {prec_final:.4f}")
        print(f"    Recall    : {rec_final:.4f}")
        print(f"    F1        : {f1_final:.4f}")
        print(f"\n{classification_report(y, preds_final, target_names=['Healthy','High-Risk'])}")

        mlflow.log_metrics({
            "auc_roc": auc_final,
            "precision": prec_final,
            "recall": rec_final,
            "f1": f1_final,
            "best_threshold": best_thresh,
        })

        # ── Feature importance ────────────────────────────────────────────────
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 most important features:")
        print(importance.head(10).to_string(index=False))

        # ── SHAP explainer ────────────────────────────────────────────────────
        print("\nBuilding SHAP explainer...")
        explainer = shap.TreeExplainer(final_model)

        # ── Save model + threshold ────────────────────────────────────────────
        os.makedirs("models/lightgbm", exist_ok=True)
        model_package = {
            "model":         final_model,
            "threshold":     best_thresh,
            "feature_cols":  feature_cols,
            "scale_pos_weight": scale_pos_weight,
            "version":       "2.0.0",
            "cv_auc":        mean_auc,
        }
        joblib.dump(model_package, MODEL_OUTPUT_PATH)
        joblib.dump(explainer, SHAP_OUTPUT_PATH)

        mlflow.log_artifact(MODEL_OUTPUT_PATH)
        print(f"\n  ✓ Model saved to {MODEL_OUTPUT_PATH}")
        print(f"  ✓ SHAP explainer saved to {SHAP_OUTPUT_PATH}")
        print(f"\n  MLflow run logged at http://localhost:5001")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  CV AUC: {mean_auc:.4f} — {'GOOD' if mean_auc > 0.80 else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)


if __name__ == "__main__":
    train()