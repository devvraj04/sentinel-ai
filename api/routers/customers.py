"""
api/routers/customers.py
Customer data endpoints for the Credit Officer Dashboard.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import boto3
from config.settings import get_settings
 
settings = get_settings()
router = APIRouter()
 
 
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        endpoint_url=settings.dynamodb_endpoint,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
 
 
class CustomerSummary(BaseModel):
    customer_id: str
    full_name: str
    pulse_score: int
    risk_tier: str
    credit_utilization: float
    monthly_income: float
    days_past_due: int
    last_intervention: Optional[str] = None
 
 
@router.get("", response_model=List[CustomerSummary])
async def list_customers(
    risk_tier: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
):
    """List customers with their current risk scores."""
    try:
        db = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        response = table.scan(Limit=limit)
        items = response.get("Items", [])
        customers = []
        for item in items:
            if risk_tier and item.get("risk_tier") != risk_tier:
                continue
            customers.append(CustomerSummary(
                customer_id=item["customer_id"],
                full_name=item.get("full_name", "Unknown"),
                pulse_score=int(item.get("pulse_score", 0)),
                risk_tier=item.get("risk_tier", "green"),
                credit_utilization=float(item.get("credit_utilization", 0)),
                monthly_income=float(item.get("monthly_income", 0)),
                days_past_due=int(item.get("days_past_due", 0)),
                last_intervention=item.get("last_intervention_type"),
            ))
        return sorted(customers, key=lambda x: x.pulse_score, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.get("/{customer_id}")
async def get_customer(customer_id: str):
    """Get full customer profile including risk details."""
    try:
        db = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        response = table.get_item(Key={"customer_id": customer_id})
        item = response.get("Item")
        if not item:
            raise HTTPException(status_code=404, detail="Customer not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
