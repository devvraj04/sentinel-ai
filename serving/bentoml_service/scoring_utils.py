"""
serving/bentoml_service/scoring_utils.py
──────────────────────────────────────────────────────────────────────────────
SINGLE SOURCE OF TRUTH for all pulse score calculations.

Sigmoid calibration (why k=8, center=0.42):
  - The LightGBM model's own decision threshold is PD = 0.42.
  - We set center=0.42 so a customer exactly at the model threshold
    gets Pulse Score = 50 (mid-orange).
  - k=8 (was 10) gives a gentler curve so realistic PD distributions
    (0.05–0.80) map across the full 0–100 score range instead of
    clustering near the extremes.

Resulting tier boundaries:
  PD < 15%   → score < 25  → green   (safe, low risk)
  PD 15–28%  → score 25–44 → yellow  (watch)
  PD 28–55%  → score 45–69 → orange  (at risk)
  PD > 55%   → score 70+   → red     (critical)
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
from typing import Optional


# ── Tier thresholds — change here, takes effect everywhere ───────────────────
THRESHOLD_RED    = 70
THRESHOLD_ORANGE = 45
THRESHOLD_YELLOW = 25

# ── Sigmoid parameters — calibrated to model decision threshold ───────────────
# center = model threshold (0.42) → PD at threshold maps to score 50
# k      = steepness (8 gives realistic spread across full PD range)
_SIGMOID_CENTER = 0.42
_SIGMOID_K      = 8.0


def pd_to_pulse_score(pd_probability: float) -> int:
    """
    Convert PD probability → Pulse Score 0–100.

    Calibrated to the LightGBM model's threshold (0.42):
      PD = 0.42  → score = 50  (centre of orange tier)
      PD = 0.10  → score =  7  (green, low risk)
      PD = 0.60  → score = 81  (red, critical)
    """
    scaled = 1.0 / (1.0 + math.exp(-_SIGMOID_K * (float(pd_probability) - _SIGMOID_CENTER)))
    return max(1, min(100, int(round(scaled * 100))))


def pulse_score_to_tier(pulse_score: int) -> str:
    """Convert a Pulse Score (0–100) to a risk tier string."""
    if pulse_score >= THRESHOLD_RED:    return "red"
    if pulse_score >= THRESHOLD_ORANGE: return "orange"
    if pulse_score >= THRESHOLD_YELLOW: return "yellow"
    return "green"


def get_intervention(risk_tier: str) -> tuple[bool, Optional[str]]:
    """Return (intervention_recommended, intervention_type) for a risk tier."""
    if risk_tier == "red":    return True,  "payment_holiday_or_restructuring"
    if risk_tier == "orange": return True,  "flexible_emi_offer"
    if risk_tier == "yellow": return True,  "preventive_digital_nudge"
    return False, None


def tier_label(risk_tier: str) -> str:
    """Human-readable label for a risk tier."""
    return {
        "red":    "Critical",
        "orange": "At Risk",
        "yellow": "Watch",
        "green":  "Safe",
    }.get(risk_tier, risk_tier)