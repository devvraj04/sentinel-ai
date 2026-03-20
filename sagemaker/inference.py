"""
sagemaker/inference.py
Handles real-time inference requests on the SageMaker endpoint.
SageMaker calls model_fn() once on startup, then predict_fn() per request.
"""
import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    """
    Called ONCE when the endpoint starts.
    Loads model from /opt/ml/model/ (SageMaker downloads from S3 automatically).
    """
    package   = joblib.load(os.path.join(model_dir, "lgbm_model.joblib"))
    explainer = joblib.load(os.path.join(model_dir, "shap_explainer.joblib"))
    return {"package": package, "explainer": explainer}


def input_fn(request_body, content_type="application/json"):
    """Parse incoming request."""
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: dict, model: dict):
    """
    Called for every scoring request.
    input_data = {"customer_id": "CUST00123", "features": {...}}
    """
    package      = model["package"]
    explainer    = model["explainer"]
    lgbm_model   = package["model"]
    feature_cols = package["feature_cols"]
    threshold    = package["threshold"]
    version      = package.get("version", "unknown")

    features = input_data.get("features", {})
    features["month_offset"] = 0.0

    fv = np.array(
        [float(features.get(col, 0.0)) for col in feature_cols], dtype=float
    )

    pd_prob     = float(lgbm_model.predict_proba(fv.reshape(1, -1))[0, 1])
    scaled      = 1.0 / (1.0 + np.exp(-10.0 * (pd_prob - 0.30)))
    pulse_score = max(1, min(100, int(round(scaled * 100))))

    if pulse_score >= 70:   tier = "red"
    elif pulse_score >= 45: tier = "orange"
    elif pulse_score >= 25: tier = "yellow"
    else:                   tier = "green"

    # SHAP
    try:
        shap_vals = explainer.shap_values(fv.reshape(1, -1))
        vals      = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        factors   = sorted([
            {"feature_name": col, "contribution": round(abs(float(vals[i])), 4),
             "direction": "increases_risk" if vals[i] > 0 else "decreases_risk"}
            for i, col in enumerate(feature_cols) if abs(vals[i]) > 0.001
        ], key=lambda x: x["contribution"], reverse=True)[:5]
    except Exception:
        factors = []

    return {
        "customer_id":  input_data.get("customer_id", "unknown"),
        "pulse_score":  pulse_score,
        "risk_tier":    tier,
        "pd_probability": round(pd_prob, 4),
        "confidence":   round(abs(pd_prob - 0.5) * 2, 4),
        "top_factors":  factors,
        "model_version":version,
    }


def output_fn(prediction, accept="application/json"):
    """Serialize response."""
    return json.dumps(prediction), "application/json"