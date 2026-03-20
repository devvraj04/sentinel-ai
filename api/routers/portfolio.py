"""
api/routers/portfolio.py
Portfolio-level risk metrics for Risk Manager Dashboard.
Full DynamoDB scan with pagination — no artificial limits.
"""
from __future__ import annotations

import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import get_settings

settings = get_settings()
router   = APIRouter()


def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(connect_timeout=10, read_timeout=10,
                      retries={"max_attempts": 1}),
    )


def scan_all(table) -> list:
    """Scan entire table with full pagination — returns all items."""
    items       = []
    scan_kwargs = {}
    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return items


class PortfolioMetrics(BaseModel):
    total_customers:       int
    critical_risk_count:   int
    at_risk_count:         int
    watch_count:           int
    safe_count:            int
    total_portfolio_debt:  float
    avg_credit_utilization:float
    npl_ratio:             float
    avg_pd:                float
    expected_loss:         float


class RiskTierBreakdown(BaseModel):
    tier:         str
    count:        int
    percentage:   float
    avg_score:    float
    total_debt:   float


@router.get("/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics():
    """
    Full portfolio risk metrics.
    Scans all customers in DynamoDB — no limit.
    ECL = PD × LGD × EAD  (LGD = 0.45 per RBI norms)
    """
    try:
        db    = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        items = scan_all(table)

        if not items:
            return PortfolioMetrics(
                total_customers=0, critical_risk_count=0,
                at_risk_count=0, watch_count=0, safe_count=0,
                total_portfolio_debt=0.0, avg_credit_utilization=0.0,
                npl_ratio=0.0, avg_pd=0.0, expected_loss=0.0,
            )

        LGD = 0.45

        critical = sum(1 for i in items if i.get("risk_tier") == "red")
        at_risk  = sum(1 for i in items if i.get("risk_tier") == "orange")
        watch    = sum(1 for i in items if i.get("risk_tier") == "yellow")
        safe     = sum(1 for i in items if i.get("risk_tier") == "green")

        total_debt = sum(float(i.get("outstanding_balance", 0)) for i in items)
        avg_util   = sum(float(i.get("credit_utilization", 0)) for i in items) / len(items)
        avg_pd     = sum(float(i.get("pd_probability", 0)) for i in items) / len(items)
        npl_ratio  = (critical / len(items)) * 100

        # ECL = PD × LGD × EAD
        ecl = sum(
            float(i.get("pd_probability", 0)) *
            LGD *
            float(i.get("outstanding_balance", 0))
            for i in items
        )

        return PortfolioMetrics(
            total_customers=       len(items),
            critical_risk_count=   critical,
            at_risk_count=         at_risk,
            watch_count=           watch,
            safe_count=            safe,
            total_portfolio_debt=  round(total_debt, 2),
            avg_credit_utilization=round(avg_util, 4),
            npl_ratio=             round(npl_ratio, 2),
            avg_pd=                round(avg_pd, 4),
            expected_loss=         round(ecl, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-breakdown")
async def get_risk_breakdown():
    """
    Detailed breakdown of customers per risk tier.
    Used for portfolio distribution charts.
    """
    try:
        db    = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        items = scan_all(table)

        if not items:
            return {"tiers": []}

        tiers = ["green", "yellow", "orange", "red"]
        total = len(items)
        result = []

        for tier in tiers:
            tier_items = [i for i in items if i.get("risk_tier") == tier]
            if not tier_items:
                continue
            avg_score  = sum(int(i.get("pulse_score", 0)) for i in tier_items) / len(tier_items)
            total_debt = sum(float(i.get("outstanding_balance", 0)) for i in tier_items)
            result.append({
                "tier":       tier,
                "count":      len(tier_items),
                "percentage": round(len(tier_items) / total * 100, 1),
                "avg_score":  round(avg_score, 1),
                "total_debt": round(total_debt, 2),
            })

        return {"tiers": result, "total_customers": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geography")
async def get_geography_breakdown():
    """Risk distribution by geography — for regional heatmaps."""
    try:
        db    = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        items = scan_all(table)

        geo_data = {}
        for item in items:
            geo = item.get("geography", "Unknown")
            if geo not in geo_data:
                geo_data[geo] = {"total": 0, "high_risk": 0, "total_debt": 0.0}
            geo_data[geo]["total"] += 1
            if item.get("risk_tier") in ["red", "orange"]:
                geo_data[geo]["high_risk"] += 1
            geo_data[geo]["total_debt"] += float(item.get("outstanding_balance", 0))

        result = [
            {
                "geography":      geo,
                "total":          v["total"],
                "high_risk":      v["high_risk"],
                "high_risk_pct":  round(v["high_risk"] / v["total"] * 100, 1),
                "total_debt":     round(v["total_debt"], 2),
            }
            for geo, v in sorted(geo_data.items(),
                                  key=lambda x: x[1]["high_risk"], reverse=True)
        ]

        return {"geographies": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))