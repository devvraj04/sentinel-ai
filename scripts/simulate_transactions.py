"""
scripts/simulate_transactions.py
──────────────────────────────────────────────────────────────────────────────
PHASE 1 — Fast historical load  (~60-90 seconds)
  - Skips Kafka entirely, direct bulk SQL to PostgreSQL
  - DynamoDB customer state seeded once via batch_writer
  - All 7 stress signals embedded with realistic patterns

PHASE 2 — Real-time random stream (truly random, non-sequential)
  - ANY customer can transact at ANY time
  - Transaction type is random (salary, UPI, ATM, EMI, utility, etc.)
  - Amount and stress signals vary by customer risk profile
  - Publishes to Kafka + PostgreSQL + DynamoDB transactions table
  - Multiple customers transact simultaneously in each batch
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
import psycopg2
import psycopg2.extras
from botocore.config import Config
from faker import Faker

from config.settings import get_settings
from ingestion.schemas.transaction_event import (
    AccountType, MerchantCategory, PaymentStatus,
    TransactionEvent, TransactionType,
)

fake = Faker("en_IN")
settings = get_settings()
random.seed(42)

GEOGRAPHIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
               "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
SEGMENTS    = ["mass_retail", "mass_retail", "mass_retail", "affluent", "hni"]
CHANNELS    = ["mobile_app", "net_banking", "branch", "atm"]


# ── Connections ───────────────────────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(settings.database_url)

def get_dynamodb():
    """Always connects to real AWS — never localhost."""
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(
            connect_timeout=10,
            read_timeout=10,
            retries={"max_attempts": 2},
        ),
    )


# ── Customer builder ──────────────────────────────────────────────────────────
def make_customer(idx: int, risk_level: str) -> dict[str, Any]:
    monthly_income = random.randint(18_000, 250_000)
    return {
        "customer_id":         f"CUST{idx:05d}",
        "full_name":           fake.name(),
        "email":               fake.email(),
        "phone":               fake.phone_number()[:20],
        "monthly_income":      monthly_income,
        "salary_day":          random.randint(1, 7),
        "emi_amount":          monthly_income * random.uniform(0.15, 0.40),
        "emi_due_day":         random.randint(5, 12),
        "avg_savings_balance": monthly_income * random.uniform(1.0, 8.0),
        "credit_limit":        monthly_income * random.uniform(2.0, 6.0),
        "risk_level":          risk_level,
        "segment":             random.choice(SEGMENTS),
        "geography":           random.choice(GEOGRAPHIES),
        "employment_status":   random.choice(["salaried", "salaried", "self_employed"]),
        "preferred_channel":   random.choice(CHANNELS),
        "product_mix":         random.choice(["loan_only", "card_only", "both", "both"]),
    }

def build_customers() -> list[dict[str, Any]]:
    # Re-seed so customer profiles are always the same
    random.seed(42)
    out = []
    for i in range(1, 901):     out.append(make_customer(i, "healthy"))
    for i in range(901, 1301):  out.append(make_customer(i, "at_risk"))
    for i in range(1301, 1501): out.append(make_customer(i, "high_risk"))
    # Restore random state for simulation variety
    random.seed()
    return out


# ── PostgreSQL helpers ────────────────────────────────────────────────────────
def seed_customers_postgres(customers: list[dict], conn) -> None:
    print("  Seeding PostgreSQL customers...", end="", flush=True)
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO customers
                (customer_id, full_name, email, phone, segment,
                 geography, employment_status, monthly_income)
            VALUES %s
            ON CONFLICT (customer_id) DO NOTHING
        """, [(
            c["customer_id"], c["full_name"], c["email"], c["phone"],
            c["segment"], c["geography"], c["employment_status"], c["monthly_income"],
        ) for c in customers], page_size=500)
    conn.commit()
    print(f" ✓ {len(customers):,} inserted")

def bulk_insert_transactions(rows: list[tuple], conn) -> int:
    """5000-row batches. Caller controls commit."""
    if not rows:
        return 0
    BATCH = 5000
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            psycopg2.extras.execute_values(cur, """
                INSERT INTO transactions
                    (customer_id, account_id, txn_type, amount,
                     merchant_category, payment_channel, payment_status, txn_timestamp)
                VALUES %s
                ON CONFLICT DO NOTHING
            """, rows[i:i + BATCH], page_size=BATCH)
    conn.commit()
    return len(rows)

def event_to_pg_row(e: TransactionEvent) -> tuple:
    return (
        e.customer_id,
        getattr(e, "account_id", None),
        e.txn_type.value if hasattr(e.txn_type, "value") else str(e.txn_type),
        float(e.amount),
        e.merchant_category.value if hasattr(e.merchant_category, "value") else str(e.merchant_category),
        e.payment_channel.value if hasattr(e.payment_channel, "value") else str(e.payment_channel),
        e.payment_status.value if hasattr(e.payment_status, "value") else str(e.payment_status),
        e.txn_timestamp,
    )


# ── DynamoDB helpers ──────────────────────────────────────────────────────────
def seed_customers_dynamodb(customers: list[dict], db) -> None:
    print("  Seeding DynamoDB customer states...", end="", flush=True)
    table = db.Table(settings.dynamodb_table_scores)
    count = 0
    with table.batch_writer() as batch:
        for c in customers:
            risk = c["risk_level"]
            if risk == "healthy":
                score, tier = random.randint(5, 22), "green"
                pd = round(random.uniform(0.02, 0.12), 4)
            elif risk == "at_risk":
                score = random.randint(28, 58)
                tier  = random.choice(["yellow", "orange"])
                pd    = round(random.uniform(0.22, 0.55), 4)
            else:
                score = random.randint(65, 95)
                tier  = random.choice(["orange", "red"])
                pd    = round(random.uniform(0.58, 0.93), 4)

            batch.put_item(Item={
                "customer_id":         c["customer_id"],
                "full_name":           c["full_name"],
                "segment":             c["segment"],
                "geography":           c["geography"],
                "monthly_income":      Decimal(str(c["monthly_income"])),
                "credit_limit":        Decimal(str(round(c["credit_limit"], 2))),
                "outstanding_balance": Decimal(str(round(
                    c["monthly_income"] * random.uniform(0.5, 3.5), 2))),
                "credit_utilization":  Decimal(str(round(
                    random.uniform(0.4, 0.95) if risk != "healthy"
                    else random.uniform(0.05, 0.35), 4))),
                "pulse_score":         score,
                "risk_tier":           tier,
                "pd_probability":      Decimal(str(pd)),
                "confidence":          Decimal("1.0"),
                "days_past_due":       random.randint(0, 7) if risk == "high_risk" else 0,
                "employment_status":   c["employment_status"],
                "preferred_channel":   c["preferred_channel"],
                "top_factor":          "salary_delay_days" if risk == "high_risk"
                                       else "credit_utilization_delta",
                "updated_at":          datetime.now(timezone.utc).isoformat(),
                "model_version":       "2.0.0",
            })
            count += 1
    print(f" ✓ {count:,} inserted")

def write_txn_dynamodb(event: TransactionEvent, db, table_name: str) -> None:
    """Write single transaction to DynamoDB transactions table."""
    db.Table(table_name).put_item(Item={
        "customer_id":      event.customer_id,
        "txn_timestamp":    event.txn_timestamp.isoformat(),
        "txn_type":         event.txn_type.value if hasattr(event.txn_type, "value") else str(event.txn_type),
        "amount":           Decimal(str(round(float(event.amount), 2))),
        "channel":          event.payment_channel.value if hasattr(event.payment_channel, "value") else str(event.payment_channel),
        "category":         event.merchant_category.value if hasattr(event.merchant_category, "value") else str(event.merchant_category),
        "status":           event.payment_status.value if hasattr(event.payment_status, "value") else str(event.payment_status),
        "is_stress_signal": str(
            getattr(event, "is_auto_debit_failed", False) or
            getattr(event, "is_lending_app_upi", False)
        ),
        "ttl": int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp()),
    })


# ── Historical transaction generator (all 7 signals) ─────────────────────────
def generate_month_transactions(
    customer: dict[str, Any],
    year: int,
    month: int,
    month_offset: int = 0,
) -> list[TransactionEvent]:
    events  = []
    cid     = customer["customer_id"]
    risk    = customer["risk_level"]
    income  = customer["monthly_income"]
    sal_day = customer["salary_day"]
    emi_amt = customer["emi_amount"]
    emi_due = customer["emi_due_day"]
    avg_bal = customer["avg_savings_balance"]

    stress = 0.0
    if risk == "high_risk":
        stress = min(1.0, (month_offset + 6) / 6.0)
    elif risk == "at_risk":
        stress = min(0.6, (month_offset + 6) / 10.0)

    # Signal 1: Salary credit (delayed under stress)
    delay = 0
    if risk == "high_risk" and month_offset >= -3:
        delay = int(3 + stress * 6) + random.randint(0, 2)
    elif risk == "at_risk" and month_offset >= -2:
        delay = random.randint(1, 4)
    events.append(TransactionEvent(
        customer_id=cid, account_type=AccountType.SAVINGS,
        txn_type=TransactionType.SALARY_CREDIT,
        amount=round(income * random.uniform(0.96, 1.03) * (1.0 - 0.12 * stress)),
        merchant_category=MerchantCategory.OTHER,
        payment_channel="NEFT",
        txn_timestamp=datetime(year, month, min(sal_day + delay, 28),
                               random.randint(9, 12), 0, tzinfo=timezone.utc),
    ))

    # Signal 2: Savings drain (balance declining)
    num_drain = (int(2 + stress * 5) if risk == "high_risk" and month_offset >= -3
                 else (random.randint(1, 3) if risk == "at_risk" and month_offset >= -2 else 0))
    for _ in range(num_drain):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(500, avg_bal * random.uniform(0.06, 0.22) * (0.5 + stress)), 2),
            payment_channel="NEFT",
            txn_timestamp=datetime(year, month, random.randint(14, 28),
                                   random.randint(10, 18), 0, tzinfo=timezone.utc),
        ))

    # Signal 3: UPI to lending apps
    num_lending = (int(3 + stress * 10) if risk == "high_risk" and month_offset >= -3
                   else (random.randint(2, 5) if risk == "at_risk" and month_offset >= -2 else 0))
    for _ in range(num_lending):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(1500, 20000) * (0.5 + stress), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI",
            txn_timestamp=datetime(year, month, random.randint(10, 28),
                                   random.randint(9, 22), 0, tzinfo=timezone.utc),
            is_lending_app_upi=True,
        ))

    # Signal 4: Utility payment latency
    util_delay = (int(5 + stress * 12) if risk == "high_risk" and month_offset >= -3
                  else (random.randint(3, 7) if risk == "at_risk" else 0))
    for _ in range(random.randint(2, 5)):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(400, 5000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            txn_timestamp=datetime(year, month,
                                   min(8 + util_delay + random.randint(0, 3), 28),
                                   random.randint(9, 20), 0, tzinfo=timezone.utc),
        ))

    # Signal 5: Discretionary spending contraction
    if risk == "healthy":
        num_disc, amt_range = random.randint(10, 25), (300, 3500)
    elif risk == "at_risk":
        num_disc, amt_range = random.randint(4, 12), (150, 1500)
    else:
        num_disc  = max(1, int(random.randint(2, 8) * (1.0 - stress * 0.75)))
        amt_range = (80, 500)
    for _ in range(num_disc):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(*amt_range), 2),
            merchant_category=random.choice([
                MerchantCategory.DINING, MerchantCategory.ENTERTAINMENT,
                MerchantCategory.SHOPPING, MerchantCategory.DINING,
            ]),
            payment_channel="UPI",
            txn_timestamp=datetime(year, month, random.randint(1, 28),
                                   random.randint(11, 23), 0, tzinfo=timezone.utc),
        ))

    # Signal 6: ATM withdrawals (cash hoarding)
    if risk == "healthy":
        num_atm, atm_amts = random.randint(1, 3), [500, 1000, 2000]
    elif risk == "at_risk":
        num_atm, atm_amts = random.randint(3, 7), [1000, 2000, 5000]
    else:
        num_atm, atm_amts = int(5 + stress * 8), [2000, 5000, 10000]
    for _ in range(num_atm):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=float(random.choice(atm_amts)),
            payment_channel="ATM",
            txn_timestamp=datetime(year, month, random.randint(1, 28),
                                   random.randint(8, 22), 0, tzinfo=timezone.utc),
        ))

    # Signal 7: Failed auto-debit
    fail_prob = (min(0.93, 0.35 + stress * 0.55)
                 if risk == "high_risk" and month_offset >= -2
                 else (0.22 if risk == "at_risk" and month_offset >= -1 else 0.0))
    emi_failed = random.random() < fail_prob
    events.append(TransactionEvent(
        customer_id=cid, account_type=AccountType.LOAN,
        txn_type=TransactionType.AUTO_DEBIT,
        amount=round(emi_amt),
        payment_channel="ECS",
        payment_status=PaymentStatus.FAILED if emi_failed else PaymentStatus.SUCCESS,
        txn_timestamp=datetime(year, month, emi_due, 8, 0, tzinfo=timezone.utc),
        is_auto_debit_failed=emi_failed,
    ))
    if emi_failed and random.random() < 0.65:
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi_amt),
            payment_channel="ECS",
            payment_status=PaymentStatus.FAILED,
            txn_timestamp=datetime(year, month,
                                   min(emi_due + random.randint(2, 5), 28),
                                   8, 30, tzinfo=timezone.utc),
            is_auto_debit_failed=True,
        ))

    # Cross-env: credit card
    if customer["product_mix"] in ("card_only", "both"):
        for _ in range(random.randint(5, 15) if risk == "healthy" else random.randint(10, 28)):
            events.append(TransactionEvent(
                customer_id=cid, account_type=AccountType.CREDIT_CARD,
                txn_type=TransactionType.CREDIT_CARD_PAYMENT,
                amount=round(random.uniform(500, 8000) * (1.0 + stress * 0.5), 2),
                merchant_category=random.choice([
                    MerchantCategory.GROCERIES, MerchantCategory.DINING,
                    MerchantCategory.SHOPPING, MerchantCategory.OTHER,
                ]),
                payment_channel="POS",
                txn_timestamp=datetime(year, month, random.randint(1, 28),
                                       random.randint(10, 21), 0, tzinfo=timezone.utc),
            ))

    return sorted(events, key=lambda e: e.txn_timestamp)


# ── Real-time single transaction generator ────────────────────────────────────
# Weighted transaction types — reflects realistic bank transaction distribution
TXN_WEIGHTS = {
    "upi_payment":       30,   # most common — UPI payments
    "atm_withdrawal":    15,   # frequent
    "utility_payment":   12,   # regular monthly
    "credit_card":       12,   # card spend
    "salary_credit":      8,   # monthly salary
    "auto_debit":         8,   # EMI
    "savings_drain":      8,   # transfers out
    "lending_upi":        7,   # stress signal
}
TXN_TYPES   = list(TXN_WEIGHTS.keys())
TXN_PROBS   = [w / sum(TXN_WEIGHTS.values()) for w in TXN_WEIGHTS.values()]

def generate_random_txn(customer: dict[str, Any],
                        now: datetime) -> TransactionEvent:
    """
    Generate ONE completely random realistic transaction for a customer.
    Transaction type, amount, and stress signals are all randomized.
    High-risk customers have higher probability of stress transactions.
    """
    cid    = customer["customer_id"]
    risk   = customer["risk_level"]
    income = customer["monthly_income"]
    emi    = customer["emi_amount"]

    # High-risk customers: boost stress transaction probabilities
    if risk == "high_risk":
        weights = TXN_WEIGHTS.copy()
        weights["lending_upi"]   = 20   # 3x more likely
        weights["atm_withdrawal"]= 20   # 2x more likely
        weights["savings_drain"] = 15
        weights["auto_debit"]    = 10
        types  = list(weights.keys())
        probs  = [w / sum(weights.values()) for w in weights.values()]
    elif risk == "at_risk":
        weights = TXN_WEIGHTS.copy()
        weights["lending_upi"]   = 12
        weights["atm_withdrawal"]= 18
        types  = list(weights.keys())
        probs  = [w / sum(weights.values()) for w in weights.values()]
    else:
        types, probs = TXN_TYPES, TXN_PROBS

    txn_kind = random.choices(types, weights=probs, k=1)[0]

    # Build the event based on type
    if txn_kind == "upi_payment":
        cat = random.choice([
            MerchantCategory.DINING, MerchantCategory.ENTERTAINMENT,
            MerchantCategory.SHOPPING, MerchantCategory.GROCERIES,
            MerchantCategory.OTHER,
        ])
        amt = random.uniform(100, 3000) if risk == "healthy" else random.uniform(50, 800)
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(amt, 2),
            merchant_category=cat, payment_channel="UPI",
            txn_timestamp=now,
        )

    elif txn_kind == "atm_withdrawal":
        amts = ([500, 1000, 2000] if risk == "healthy"
                else [1000, 2000, 5000] if risk == "at_risk"
                else [2000, 5000, 10000])
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=float(random.choice(amts)),
            payment_channel="ATM", txn_timestamp=now,
        )

    elif txn_kind == "utility_payment":
        failed = risk == "high_risk" and random.random() < 0.15
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(400, 5000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.FAILED if failed else PaymentStatus.SUCCESS,
            txn_timestamp=now,
        )

    elif txn_kind == "credit_card":
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.CREDIT_CARD,
            txn_type=TransactionType.CREDIT_CARD_PAYMENT,
            amount=round(random.uniform(500, 8000) * (1.3 if risk == "high_risk" else 1.0), 2),
            merchant_category=random.choice([
                MerchantCategory.GROCERIES, MerchantCategory.DINING,
                MerchantCategory.SHOPPING, MerchantCategory.OTHER,
            ]),
            payment_channel="POS", txn_timestamp=now,
        )

    elif txn_kind == "salary_credit":
        # Delayed for high risk
        delay_mins = random.randint(0, 5 * 24 * 60) if risk == "high_risk" else 0
        ts = now - timedelta(minutes=delay_mins)
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SALARY_CREDIT,
            amount=round(income * random.uniform(0.92, 1.02)),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT", txn_timestamp=ts,
        )

    elif txn_kind == "auto_debit":
        fail_prob = (0.75 if risk == "high_risk"
                     else 0.25 if risk == "at_risk" else 0.03)
        failed = random.random() < fail_prob
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi),
            payment_channel="ECS",
            payment_status=PaymentStatus.FAILED if failed else PaymentStatus.SUCCESS,
            txn_timestamp=now,
            is_auto_debit_failed=failed,
        )

    elif txn_kind == "savings_drain":
        amt = customer["avg_savings_balance"] * random.uniform(0.05, 0.25)
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(500, amt), 2),
            payment_channel="NEFT", txn_timestamp=now,
        )

    else:  # lending_upi — stress signal
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(2000, 25000), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI",
            txn_timestamp=now,
            is_lending_app_upi=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Ultra-fast historical load (NO Kafka, direct SQL)
# ══════════════════════════════════════════════════════════════════════════════
def run_historical(months_back: int = 6,
                   db_conn=None, dynamo_db=None) -> None:
    """
    Bulk loads 6 months of history as fast as possible.
    Skips Kafka — direct PostgreSQL bulk inserts.
    After this, run the feature pipeline once to populate Redis.
    """
    customers = build_customers()

    if db_conn:
        seed_customers_postgres(customers, db_conn)
    if dynamo_db:
        seed_customers_dynamodb(customers, dynamo_db)

    now   = datetime.now(timezone.utc)
    total = 0

    for month_offset in range(-months_back, 1):
        t0     = time.time()
        target = now + timedelta(days=30 * month_offset)
        year, month = target.year, target.month
        print(f"  {year}-{month:02d} ...", end="", flush=True)

        all_rows = []
        for customer in customers:
            events   = generate_month_transactions(customer, year, month, month_offset)
            all_rows.extend([event_to_pg_row(e) for e in events])

        if db_conn:
            bulk_insert_transactions(all_rows, db_conn)

        elapsed = time.time() - t0
        total  += len(all_rows)
        print(f" {len(all_rows):,} rows  ({elapsed:.1f}s)")

    print(f"\n  ✓ Historical complete: {total:,} transactions in PostgreSQL")
    print("  ✓ Run next: python -m ingestion.consumers.feature_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Real-time random stream
# ══════════════════════════════════════════════════════════════════════════════
def run_realtime(db_conn=None, dynamo_db=None,
                 interval_seconds: float = 0.5) -> None:
    """
    Truly random real-time transaction injection.

    Every tick:
      - Picks 5-25 RANDOM customers (any risk level, any segment, any geography)
      - Generates 1 RANDOM transaction per customer (type, amount, channel all random)
      - High-risk customers have higher probability of stress transactions
      - No sequential order — CUST01450 can transact before CUST00001
      - Publishes to Kafka (feature pipeline auto-picks up)
      - Writes to PostgreSQL (immediate commit)
      - Writes to DynamoDB sentinel-transactions table
    """
    from ingestion.producers.transaction_producer import TransactionProducer

    customers     = build_customers()
    dynamo_table  = getattr(settings, "dynamodb_table_transactions",
                             "sentinel-transactions")
    total_count   = 0
    stress_count  = 0

    print("\n" + "=" * 60)
    print("REAL-TIME RANDOM STREAM")
    print(f"  Interval  : {interval_seconds}s per batch")
    print(f"  Batch size: 5-25 random customers per tick")
    print(f"  All 1500 customers eligible every tick")
    print(f"  DynamoDB  : {dynamo_table}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    with TransactionProducer() as producer:
        while True:
            try:
                now = datetime.now(timezone.utc)

                # ── Pick completely random customers ──────────────────────────
                # random.sample ensures no repeats within a batch
                # but any customer can appear in any batch
                batch_size = random.randint(5, 25)
                batch_customers = random.sample(customers, k=batch_size)

                batch_events = []
                batch_rows   = []

                for customer in batch_customers:
                    # Add small random jitter to timestamp (0-30 seconds apart)
                    jitter = timedelta(seconds=random.uniform(0, 30))
                    evt = generate_random_txn(customer, now + jitter)

                    batch_events.append(evt)
                    batch_rows.append(event_to_pg_row(evt))

                    is_stress = (
                        getattr(evt, "is_auto_debit_failed", False) or
                        getattr(evt, "is_lending_app_upi", False) or
                        evt.payment_status == PaymentStatus.FAILED
                    )
                    if is_stress:
                        stress_count += 1

                # 1. Kafka — feature pipeline auto-processes
                producer.publish_batch(batch_events)

                # 2. PostgreSQL — immediate commit
                if db_conn:
                    bulk_insert_transactions(batch_rows, db_conn)

                # 3. DynamoDB sentinel-transactions
                if dynamo_db:
                    for evt in batch_events:
                        try:
                            write_txn_dynamodb(evt, dynamo_db, dynamo_table)
                        except Exception:
                            pass

                total_count += len(batch_events)

                # Progress log every 200 events
                if total_count % 200 == 0:
                    ts = now.strftime("%H:%M:%S")
                    print(f"  [{ts}] Total: {total_count:,}  "
                          f"Stress signals: {stress_count:,}  "
                          f"Last batch: {len(batch_events)} customers  "
                          f"(IDs: {batch_customers[0]['customer_id']} "
                          f"→ {batch_customers[-1]['customer_id']})")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print(f"\n  Stream stopped.")
                print(f"  Total events injected : {total_count:,}")
                print(f"  Stress signals fired  : {stress_count:,}")
                break
            except Exception as e:
                print(f"  Error (retrying in 2s): {e}")
                time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentinel Transaction Simulator")
    parser.add_argument("--mode", choices=["hist", "realtime", "both"],
                        default="both",
                        help="hist=historical only | realtime=stream only | both=hist then stream")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of historical data (default: 6)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between real-time batches (default: 0.5)")
    args = parser.parse_args()

    print("=" * 60)
    print("SENTINEL — Transaction Simulator")
    print(f"  Mode      : {args.mode}")
    print(f"  Customers : 1500  (900 healthy / 400 at-risk / 200 high-risk)")
    print(f"  Months    : {args.months}")
    print("=" * 60)

    # PostgreSQL
    print("\nConnecting to PostgreSQL...", end="", flush=True)
    conn = get_db_conn()
    print(" ✓")

    # DynamoDB
    print("Connecting to AWS DynamoDB...", end="", flush=True)
    try:
        dynamo = get_dynamodb()
        dynamo.meta.client.list_tables(Limit=1)
        print(f" ✓ ({settings.aws_region})")
    except Exception as e:
        print(f" ✗  ({e})")
        print("  Continuing without DynamoDB")
        dynamo = None

    try:
        if args.mode in ("hist", "both"):
            print(f"\n{'='*60}")
            print("PHASE 1 — Fast Historical Load (direct SQL, no Kafka)")
            print(f"{'='*60}\n")
            t0 = time.time()
            run_historical(months_back=args.months,
                           db_conn=conn, dynamo_db=dynamo)
            elapsed = time.time() - t0
            print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM customers;")
                print(f"  Customers    : {cur.fetchone()[0]:,}")
                cur.execute("SELECT COUNT(*) FROM transactions;")
                print(f"  Transactions : {cur.fetchone()[0]:,}")

        if args.mode in ("realtime", "both"):
            if args.mode == "both":
                print("\nHistorical done. Starting real-time in 3s...")
                time.sleep(3)
            run_realtime(db_conn=conn, dynamo_db=dynamo,
                         interval_seconds=args.interval)

    finally:
        conn.close()
        print("\nDone.")