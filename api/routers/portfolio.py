"""
api/routers/portfolio.py
Portfolio-level aggregated risk metrics for Risk Manager Dashboard.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from boto3.dynamodb.conditions import Attr
import boto3
from config.settings import get_settings
 
settings = get_settings()
router = APIRouter()
 
 
class PortfolioMetrics(BaseModel):
    total_customers: int
    critical_risk_count: int
    at_risk_count: int
    safe_count: int
    total_portfolio_debt: float
    avg_credit_utilization: float
    npl_ratio: float
    avg_pd: float
    expected_loss: float
 
 
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        endpoint_url=settings.dynamodb_endpoint,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
 
 
@router.get("/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics():
    """Aggregate risk metrics across the entire portfolio."""
    try:
        db = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        response = table.scan()
        items = response.get("Items", [])
 
        if not items:
            return PortfolioMetrics(
                total_customers=0, critical_risk_count=0, at_risk_count=0,
                safe_count=0, total_portfolio_debt=0.0, avg_credit_utilization=0.0,
                npl_ratio=0.0, avg_pd=0.0, expected_loss=0.0,
            )
 
        critical = sum(1 for i in items if i.get("risk_tier") in ["red"])
        at_risk  = sum(1 for i in items if i.get("risk_tier") in ["orange", "yellow"])
        safe     = sum(1 for i in items if i.get("risk_tier") == "green")
        total_debt = sum(float(i.get("outstanding_balance", 0)) for i in items)
        avg_util   = sum(float(i.get("credit_utilization", 0)) for i in items) / max(len(items), 1)
        avg_pd     = sum(float(i.get("pd_probability", 0)) for i in items) / max(len(items), 1)
        npl_ratio  = (critical / max(len(items), 1)) * 100
        ecl        = sum(
            float(i.get("pd_probability", 0)) * float(i.get("outstanding_balance", 0)) * 0.45
            for i in items
        )
 
        return PortfolioMetrics(
            total_customers=len(items),
            critical_risk_count=critical,
            at_risk_count=at_risk,
            safe_count=safe,
            total_portfolio_debt=round(total_debt, 2),
            avg_credit_utilization=round(avg_util, 4),
            npl_ratio=round(npl_ratio, 2),
            avg_pd=round(avg_pd, 4),
            expected_loss=round(ecl, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
