"""
serving/bentoml_service/schemas.py
Data models for the Pulse Score API request and response.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
 
 
class RiskTier(str, Enum):
    GREEN  = "green"    # Pulse 0-30:  Low risk — preventive nudge only
    YELLOW = "yellow"   # Pulse 30-50: Moderate risk — monitor closely
    ORANGE = "orange"   # Pulse 50-75: High risk — flexible EMI offer
    RED    = "red"      # Pulse 75+:   Critical — payment holiday / restructuring
 
 
class SHAPFactor(BaseModel):
    feature_name:    str
    contribution:    float    # SHAP value (positive = increases risk)
    human_readable:  str      # e.g. "Salary delayed by 5 days"
    direction:       str      # "increases_risk" | "reduces_risk"
 
 
class PulseScoreRequest(BaseModel):
    customer_id:  str = Field(..., min_length=1, max_length=50)
    force_refresh: bool = False   # If True, skip cache and recompute
 
 
class PulseScoreResponse(BaseModel):
    customer_id:    str
    pulse_score:    int = Field(..., ge=0, le=100)
    risk_tier:      RiskTier
    pd_probability: float = Field(..., ge=0.0, le=1.0)
    confidence:     float = Field(..., ge=0.0, le=1.0)
    top_factors:    list[SHAPFactor]
    intervention_recommended: bool
    intervention_type: Optional[str] = None   # "payment_holiday" | "flexible_emi" | "digital_nudge"
    scored_at:      datetime
    model_version:  str
    cached:         bool = False
 
 
def score_to_tier(pulse_score: int) -> RiskTier:
    if pulse_score >= 75: return RiskTier.RED
    if pulse_score >= 50: return RiskTier.ORANGE
    if pulse_score >= 30: return RiskTier.YELLOW
    return RiskTier.GREEN
 
 
def tier_to_intervention(tier: RiskTier) -> tuple[bool, Optional[str]]:
    """Return (should_intervene, intervention_type)."""
    if tier == RiskTier.RED:    return True,  "payment_holiday"
    if tier == RiskTier.ORANGE: return True,  "flexible_emi"
    if tier == RiskTier.YELLOW: return True,  "digital_nudge"
    return False, None
