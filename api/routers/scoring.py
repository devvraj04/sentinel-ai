"""
api/routers/scoring.py
Real-time scoring endpoint. Returns Pulse Score in < 50ms.
"""
from fastapi import APIRouter, HTTPException
from serving.bentoml_service.pulse_scorer import get_scorer
from serving.bentoml_service.schemas import PulseScoreRequest, PulseScoreResponse

router = APIRouter()


@router.post("/score")
async def score_customer(request: PulseScoreRequest):
    """
    Score a customer and return Pulse Score, risk tier, and top risk factors.
    Writes result to AWS DynamoDB automatically.
    """
    try:
        scorer = get_scorer()
        result = scorer.score(
            customer_id=request.customer_id,
            force_refresh=getattr(request, "force_refresh", False),
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(exc)}")