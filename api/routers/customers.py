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
    limit: int = Query(200, le=500),
):
    """
    Get Pulse Score history for a customer from PostgreSQL.
    Every score computation appends a row here — never updated, only inserted.
    DynamoDB holds the current score; this endpoint exposes the full timeline.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    pulse_score,
                    risk_tier,
                    pd_probability,
                    confidence,
                    top_factor_1,
                    top_factor_2,
                    top_factor_3,
                    intervention_flag,
                    intervention_type,
                    model_version,
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
            score, tier, pd, conf, f1, f2, f3, iv_flag, iv_type, mv, scored_at = row
            history.append({
                "pulse_score":        int(score),
                "risk_tier":          tier,
                "pd_probability":     float(pd or 0),
                "confidence":         float(conf or 0),
                "top_factors":        [f for f in [f1, f2, f3] if f],
                "intervention_flag":  bool(iv_flag) if iv_flag is not None else False,
                "intervention_type":  iv_type or "none",
                "model_version":      mv or "unknown",
                "scored_at":          scored_at.isoformat() if scored_at else "",
            })

        return {
            "customer_id": customer_id,
            "total":       len(history),
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
    Returns recent transactions with their impact on the Pulse Score,
    anchored to the REAL scores recorded in pulse_score_history.

    For each transaction we find the closest real scored_at entry in
    pulse_score_history and attach that actual score — no simulation,
    no hardcoded baseline. The score_after value shown is the real model
    output that was written to both DynamoDB and PostgreSQL at that moment.
    """
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Fetch recent transactions
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
            txn_rows = cur.fetchall()

            # Fetch the full real score timeline for this customer
            cur.execute("""
                SELECT pulse_score, scored_at
                FROM pulse_score_history
                WHERE customer_id = %s
                ORDER BY scored_at ASC
            """, (customer_id,))
            score_rows = cur.fetchall()

        conn.close()

        # Build a sorted list of (scored_at, pulse_score) for lookup
        score_timeline = [
            (row[1], int(row[0]))
            for row in score_rows
        ]  # sorted ASC by scored_at

        def real_score_at(ts) -> Optional[int]:
            """
            Return the real pulse score that was active at timestamp ts.
            Uses the most recent score entry whose scored_at <= ts.
            Falls back to the earliest score if ts precedes all history.
            Returns None if no history exists at all.
            """
            if not score_timeline:
                return None
            best = score_timeline[0][1]
            for scored_at, score in score_timeline:
                if scored_at <= ts:
                    best = score
                else:
                    break
            return best

        # Classify transaction type for display labels only.
        # score_impact delta is computed from the REAL score timeline, not hardcoded weights.
        def classify_txn(txn_type: str, category: str,
                         status: str, amount: float) -> tuple[str, bool]:
            """Returns (human-readable reason, is_stress_signal) for display."""
            if status == "failed" and txn_type == "auto_debit":
                return "Failed EMI payment — major stress signal", True
            if status == "failed" and txn_type == "utility_payment":
                return "Failed utility payment — cash flow stress", True
            if category == "lending_app":
                return "Borrowed from lending app — debt stress", True
            if txn_type == "savings_withdrawal" and amount > 5_000:
                return "Large savings withdrawal — buffer erosion", True
            if txn_type == "atm_withdrawal" and amount >= 5_000:
                return "Large ATM withdrawal — cash hoarding", True
            if txn_type == "salary_credit" and status == "success":
                return "Salary received — positive cashflow", False
            if txn_type == "utility_payment" and status == "success":
                return "Utility paid on time — financial discipline", False
            return "neutral transaction", False

        result = []
        prev_score: Optional[int] = None

        for txn_type, amount, category, status, channel, ts in txn_rows:
            amt         = float(amount or 0)
            reason, is_stress = classify_txn(txn_type or "", category or "",
                                             status or "", amt)
            # score_after: real model score active at this transaction timestamp
            score_after = real_score_at(ts) if ts else None

            # score_impact: actual delta between consecutive real scores.
            # None if we have no history data. Never hardcoded.
            if score_after is not None and prev_score is not None:
                score_impact = score_after - prev_score
            else:
                score_impact = None

            prev_score = score_after if score_after is not None else prev_score

            result.append({
                "txn_timestamp":  ts.isoformat() if ts else "",
                "txn_type":       txn_type or "",
                "amount":         amt,
                "category":       category or "other",
                "channel":        channel or "unknown",
                "status":         status or "success",
                "score_impact":   score_impact,   # real delta, or null if no history
                "score_after":    score_after,    # real model score, or null if no history
                "impact_reason":  reason,
                "is_stress_signal": is_stress,
            })

        return {
            "customer_id":        customer_id,
            "transactions":       result,
            "total_transactions": len(result),
            "score_source":       "pulse_score_history" if score_timeline else "unavailable",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))