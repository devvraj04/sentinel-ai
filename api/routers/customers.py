"""
api/routers/customers.py
Customer endpoints — no artificial limits, full DynamoDB pagination,
transaction history with pulse score impact per transaction.
"""
from __future__ import annotations

from typing import List, Optional

import boto3
import psycopg2
from botocore.config import Config
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config.settings import get_settings

settings = get_settings()
router   = APIRouter()


# ── DynamoDB helper ───────────────────────────────────────────────────────────
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(connect_timeout=10, read_timeout=10,
                      retries={"max_attempts": 1}),
    )


def get_db_conn():
    return psycopg2.connect(settings.database_url)


def scan_all(table) -> list:
    """
    Scan entire DynamoDB table handling pagination automatically.
    No artificial limit — returns all items.
    """
    items       = []
    scan_kwargs = {}
    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return items


# ── Response models ───────────────────────────────────────────────────────────
class CustomerSummary(BaseModel):
    customer_id:        str
    full_name:          str
    pulse_score:        int
    risk_tier:          str
    credit_utilization: float
    monthly_income:     float
    days_past_due:      int
    outstanding_balance:float
    pd_probability:     float
    top_factor:         Optional[str] = None
    intervention_flag:  Optional[bool] = None
    intervention_type:  Optional[str]  = None
    last_intervention:  Optional[str]  = None
    segment:            Optional[str]  = None
    geography:          Optional[str]  = None
    employment_status:  Optional[str]  = None
    preferred_channel:  Optional[str]  = None
    updated_at:         Optional[str]  = None


class TransactionRecord(BaseModel):
    txn_timestamp:    str
    txn_type:         str
    amount:           float
    category:         str
    channel:          str
    status:           str
    is_stress_signal: bool
    pulse_score_after: Optional[int]  = None
    score_change:      Optional[int]  = None


class CustomerDetail(BaseModel):
    customer_id:        str
    full_name:          str
    pulse_score:        int
    risk_tier:          str
    pd_probability:     float
    confidence:         float
    credit_utilization: float
    outstanding_balance:float
    monthly_income:     float
    credit_limit:       float
    days_past_due:      int
    employment_status:  str
    segment:            str
    geography:          str
    preferred_channel:  str
    top_factor:         Optional[str] = None
    intervention_flag:  Optional[bool] = None
    intervention_type:  Optional[str]  = None
    model_version:      Optional[str]  = None
    updated_at:         Optional[str]  = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[CustomerSummary])
async def list_customers(
    risk_tier: Optional[str] = Query(None, description="Filter: green/yellow/orange/red"),
    search:    Optional[str] = Query(None, description="Search by customer_id or name"),
    sort_by:   str           = Query("pulse_score", description="Sort field"),
    limit:     int           = Query(100, le=1500, description="Max results (up to 1500)"),
    offset:    int           = Query(0, description="Pagination offset"),
):
    """
    List all customers with risk data.
    Supports filtering by risk tier, search by ID/name, sorting.
    Returns ALL customers — no artificial cap.
    """
    try:
        db    = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        items = scan_all(table)

        customers = []
        for item in items:
            try:
                # Apply risk tier filter
                if risk_tier and item.get("risk_tier") != risk_tier:
                    continue
                # Apply search filter
                if search:
                    search_lower = search.lower()
                    if (search_lower not in item.get("customer_id", "").lower() and
                            search_lower not in item.get("full_name", "").lower()):
                        continue

                customers.append(CustomerSummary(
                    customer_id=        item["customer_id"],
                    full_name=          item.get("full_name", "Unknown"),
                    pulse_score=        int(item.get("pulse_score", 0)),
                    risk_tier=          item.get("risk_tier", "green"),
                    credit_utilization= float(item.get("credit_utilization", 0)),
                    monthly_income=     float(item.get("monthly_income", 0)),
                    days_past_due=      int(item.get("days_past_due", 0)),
                    outstanding_balance=float(item.get("outstanding_balance", 0)),
                    pd_probability=     float(item.get("pd_probability", 0)),
                    top_factor=         item.get("top_factor"),
                    intervention_flag=  item.get("intervention_flag"),
                    intervention_type=  item.get("intervention_type"),
                    last_intervention=  item.get("last_intervention_type"),
                    segment=            item.get("segment"),
                    geography=          item.get("geography"),
                    employment_status=  item.get("employment_status"),
                    preferred_channel=  item.get("preferred_channel"),
                    updated_at=         item.get("updated_at"),
                ))
            except Exception:
                continue

        # Sort
        reverse = True
        if sort_by == "pulse_score":
            customers.sort(key=lambda x: x.pulse_score, reverse=reverse)
        elif sort_by == "monthly_income":
            customers.sort(key=lambda x: x.monthly_income, reverse=reverse)
        elif sort_by == "days_past_due":
            customers.sort(key=lambda x: x.days_past_due, reverse=reverse)
        elif sort_by == "outstanding_balance":
            customers.sort(key=lambda x: x.outstanding_balance, reverse=reverse)

        return customers[offset: offset + limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}", response_model=CustomerDetail)
async def get_customer(customer_id: str):
    """Get complete customer profile from DynamoDB."""
    try:
        db    = get_dynamodb()
        table = db.Table(settings.dynamodb_table_scores)
        resp  = table.get_item(Key={"customer_id": customer_id})
        item  = resp.get("Item")
        if not item:
            raise HTTPException(status_code=404, detail="Customer not found")

        return CustomerDetail(
            customer_id=        item["customer_id"],
            full_name=          item.get("full_name", "Unknown"),
            pulse_score=        int(item.get("pulse_score", 0)),
            risk_tier=          item.get("risk_tier", "green"),
            pd_probability=     float(item.get("pd_probability", 0)),
            confidence=         float(item.get("confidence", 0)),
            credit_utilization= float(item.get("credit_utilization", 0)),
            outstanding_balance=float(item.get("outstanding_balance", 0)),
            monthly_income=     float(item.get("monthly_income", 0)),
            credit_limit=       float(item.get("credit_limit", 0)),
            days_past_due=      int(item.get("days_past_due", 0)),
            employment_status=  item.get("employment_status", "unknown"),
            segment=            item.get("segment", "unknown"),
            geography=          item.get("geography", "unknown"),
            preferred_channel=  item.get("preferred_channel", "unknown"),
            top_factor=         item.get("top_factor"),
            intervention_flag=  item.get("intervention_flag"),
            intervention_type=  item.get("intervention_type"),
            model_version=      item.get("model_version"),
            updated_at=         item.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}/transactions")
async def get_customer_transactions(
    customer_id: str,
    limit:  int = Query(100, le=500, description="Number of transactions"),
    offset: int = Query(0),
):
    """
    Get transaction history for a customer from PostgreSQL.
    Returns transactions sorted newest first with stress signal flags.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    txn_id,
                    txn_type,
                    amount,
                    merchant_category,
                    payment_channel,
                    payment_status,
                    txn_timestamp
                FROM transactions
                WHERE customer_id = %s
                ORDER BY txn_timestamp DESC
                LIMIT %s OFFSET %s
            """, (customer_id, limit, offset))
            rows = cur.fetchall()

            cur.execute(
                "SELECT COUNT(*) FROM transactions WHERE customer_id = %s",
                (customer_id,)
            )
            total = cur.fetchone()[0]

        conn.close()

        stress_types = {"auto_debit", "savings_withdrawal"}
        stress_cats  = {"lending_app"}

        transactions = []
        for row in rows:
            txn_id, txn_type, amount, category, channel, status, ts = row
            is_stress = (
                txn_type in stress_types or
                category in stress_cats or
                status == "failed"
            )
            transactions.append({
                "txn_id":          str(txn_id),
                "txn_timestamp":   ts.isoformat() if ts else "",
                "txn_type":        txn_type or "",
                "amount":          float(amount or 0),
                "category":        category or "other",
                "channel":         channel or "unknown",
                "status":          status or "success",
                "is_stress_signal":is_stress,
            })

        return {
            "customer_id":   customer_id,
            "total":         total,
            "limit":         limit,
            "offset":        offset,
            "transactions":  transactions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}/pulse-history")
async def get_pulse_history(
    customer_id: str,
    limit: int = Query(30, le=180),
):
    """
    Get Pulse Score history for a customer from PostgreSQL.
    Shows how the score changed over time — used for trend charts.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    pulse_score,
                    risk_tier,
                    pd_probability,
                    top_factor_1,
                    top_factor_2,
                    top_factor_3,
                    scored_at
                FROM pulse_score_history
                WHERE customer_id = %s
                ORDER BY scored_at DESC
                LIMIT %s
            """, (customer_id, limit))
            rows = cur.fetchall()
        conn.close()

        history = []
        for row in rows:
            score, tier, pd, f1, f2, f3, scored_at = row
            history.append({
                "pulse_score":    int(score),
                "risk_tier":      tier,
                "pd_probability": float(pd or 0),
                "top_factors":    [f for f in [f1, f2, f3] if f],
                "scored_at":      scored_at.isoformat() if scored_at else "",
            })

        return {
            "customer_id": customer_id,
            "history":     history,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}/score-impact")
async def get_transaction_score_impact(
    customer_id: str,
    limit: int = Query(50, le=200),
):
    """
    Returns recent transactions with their estimated impact on the Pulse Score.
    Shows WHICH transactions are driving the risk score up or down.

    Impact is calculated from the stress signal flags —
    each stress transaction contributes a score delta.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    txn_type,
                    amount,
                    merchant_category,
                    payment_status,
                    payment_channel,
                    txn_timestamp
                FROM transactions
                WHERE customer_id = %s
                ORDER BY txn_timestamp DESC
                LIMIT %s
            """, (customer_id, limit))
            rows = cur.fetchall()
        conn.close()

        # Score impact weights per transaction type
        # Positive = increases risk, Negative = decreases risk
        IMPACT_WEIGHTS = {
            # Stress signals — increase score
            ("auto_debit",    "failed"):        +8,
            ("utility_payment","failed"):        +5,
            ("upi_debit",     "lending_app"):    +7,
            ("savings_withdrawal", None):        +4,
            ("atm_withdrawal", None):            +3,
            # Healthy signals — decrease score
            ("salary_credit", "success"):        -6,
            ("utility_payment", "success"):      -2,
            ("credit_card_payment", "success"):  -1,
        }

        result = []
        running_score = 50  # approximate neutral baseline

        for row in reversed(rows):  # process oldest first for running score
            txn_type, amount, category, status, channel, ts = row

            # Determine impact
            impact = 0
            reason = "neutral"

            if status == "failed" and txn_type == "auto_debit":
                impact = +8
                reason = "Failed EMI payment — major stress signal"
            elif status == "failed" and txn_type == "utility_payment":
                impact = +5
                reason = "Failed utility payment — cash flow stress"
            elif category == "lending_app":
                impact = +7
                reason = "Borrowed from lending app — debt stress"
            elif txn_type == "savings_withdrawal" and amount > 5000:
                impact = +4
                reason = "Large savings withdrawal — buffer erosion"
            elif txn_type == "atm_withdrawal" and amount >= 5000:
                impact = +3
                reason = "Large ATM withdrawal — cash hoarding"
            elif txn_type == "salary_credit" and status == "success":
                impact = -6
                reason = "Salary received — positive cashflow"
            elif txn_type == "utility_payment" and status == "success":
                impact = -2
                reason = "Utility paid on time — financial discipline"

            running_score = max(1, min(100, running_score + impact))

            result.append({
                "txn_timestamp":   ts.isoformat() if ts else "",
                "txn_type":        txn_type or "",
                "amount":          float(amount or 0),
                "category":        category or "other",
                "channel":         channel or "unknown",
                "status":          status or "success",
                "score_impact":    impact,
                "score_after":     running_score,
                "impact_reason":   reason,
                "is_stress_signal":impact > 0,
            })

        # Return newest first
        result.reverse()

        return {
            "customer_id":        customer_id,
            "transactions":       result,
            "total_transactions": len(result),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))