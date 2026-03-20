"""
serving/bentoml_service/schemas.py
Data models for the Pulse Score API request and response.

Tier thresholds and scoring math live in scoring_utils.py — not here.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from serving.bentoml_service.scoring_utils import (
    pulse_score_to_tier, get_intervention,
    THRESHOLD_RED, THRESHOLD_ORANGE, THRESHOLD_YELLOW,
)


class RiskTier(str, Enum):
    GREEN  = "green"    # Pulse  0-24: Safe — preventive nudge only
    YELLOW = "yellow"   # Pulse 25-44: Watch — monitor closely
    ORANGE = "orange"   # Pulse 45-69: At Risk — flexible EMI offer
    RED    = "red"      # Pulse 70+:   Critical — payment holiday / restructuring


class SHAPFactor(BaseModel):
    feature_name:    str
    contribution:    float
    human_readable:  str
    direction:       str


class PulseScoreRequest(BaseModel):
    customer_id:   str = Field(..., min_length=1, max_length=50)
    force_refresh: bool = False


class PulseScoreResponse(BaseModel):
    customer_id:    str
    pulse_score:    int = Field(..., ge=0, le=100)
    risk_tier:      RiskTier
    pd_probability: float = Field(..., ge=0.0, le=1.0)
    confidence:     float = Field(..., ge=0.0, le=1.0)
    top_factors:    list[SHAPFactor]
    intervention_recommended: bool
    intervention_type: Optional[str] = None
    scored_at:      datetime
    model_version:  str
    cached:         bool = False


def score_to_tier(pulse_score: int) -> RiskTier:
    """Delegates to scoring_utils — canonical thresholds are 70/45/25."""
    return RiskTier(pulse_score_to_tier(pulse_score))


def tier_to_intervention(tier: RiskTier) -> tuple[bool, Optional[str]]:
    """Delegates to scoring_utils.get_intervention."""
    return get_intervention(tier.value)