"""
scripts/simulate_transactions.py
──────────────────────────────────────────────────────────────────────────────
CHANGES MADE TO THIS FILE:
─────────────────────────────────────────────────────────────────────────────
  CHANGE 1 — No future-dated transactions (lines marked ▶ FIX 1)
    BEFORE: datetime(year, month, day, ...) was used directly everywhere.
            If year/month was the current month, days > today created
            future timestamps. e.g. today is March 20 but day=25 → March 25.
    AFTER:  A safe_datetime() helper is added. Every single timestamp in the
            entire file goes through this function which hard-caps the result
            to TODAY. Also added a final filter in generate_month_transactions
            and run_historical to drop any event that still slips through.

  CHANGE 2 — Risk level from financial profile, not customer ID (lines marked ▶ FIX 2)
    BEFORE: build_customers() assigned risk by ID range:
              CUST00001–CUST00900  → "healthy"
              CUST00901–CUST01300  → "at_risk"
              CUST01301–CUST01500  → "high_risk"
            This meant risk was 100% predictable from ID — unrealistic.
    AFTER:  compute_risk_from_profile() calculates risk from:
              - Debt-to-income ratio  (35% weight)
              - Savings buffer        (25% weight)
              - Employment stability  (20% weight)
              - Life shock event      (20% weight)
            make_customer() now builds a full financial profile first,
            then calls compute_risk_from_profile() to derive risk.
            Any customer ID can be any risk level.
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

fake     = Faker("en_IN")
settings = get_settings()

GEOGRAPHIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
               "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
SEGMENTS    = ["mass_retail", "mass_retail", "mass_retail", "affluent", "hni"]
CHANNELS    = ["mobile_app", "net_banking", "branch", "atm"]

# ▶ FIX 1 — Define TODAY as a module-level constant.
# Every timestamp in the file is capped against this value.
TODAY = datetime.now(timezone.utc).replace(
    hour=23, minute=59, second=59, microsecond=0
)


# ── Connections ───────────────────────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(settings.database_url)

def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(connect_timeout=10, read_timeout=10,
                      retries={"max_attempts": 2}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ▶ FIX 1 — TIMESTAMP SAFETY HELPERS (NEW — did not exist before)
# ══════════════════════════════════════════════════════════════════════════════
def cap_to_today(ts: datetime) -> datetime:
    """
    ▶ FIX 1: Hard-caps any datetime to TODAY.
    If the timestamp is in the future, TODAY is returned instead.
    This is the single source of truth for all timestamp capping.
    """
    return min(ts, TODAY)


def safe_datetime(year: int, month: int, day: int,
                  hour: int = 12, minute: int = 0) -> datetime:
    """
    ▶ FIX 1: Replaces bare datetime(...) calls throughout the file.
    Constructs a UTC datetime and immediately caps it to TODAY.
    Called for every single transaction timestamp in historical mode.
    """
    ts = datetime(year, month, min(day, 28), hour, minute, tzinfo=timezone.utc)
    return cap_to_today(ts)


# ══════════════════════════════════════════════════════════════════════════════
# ▶ FIX 2 — RISK FROM FINANCIAL PROFILE (NEW — did not exist before)
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk_from_profile(profile: dict) -> tuple[str, float]:
    """
    ▶ FIX 2: Derives risk_level and stress_base from the customer's
    actual financial health — not from their customer ID number.

    BEFORE: risk was assigned by ID range in build_customers().
    AFTER:  risk is computed here from four financial factors.

    Factors:
      debt_to_income   — EMI burden relative to income        (35% weight)
      savings_buffer   — how many months of income saved      (25% weight)
      employment_risk  — salaried < self_employed < gig       (20% weight)
      life_shock       — recent job loss / medical emergency  (20% weight)

    Returns:
      risk_level  — "healthy" | "at_risk" | "high_risk"
      stress_base — float 0-1 used to drive signal intensity
    """
    income   = profile["monthly_income"]
    emi      = profile["emi_amount"]
    savings  = profile["avg_savings_balance"]
    emp_type = profile["employment_status"]
    shock    = profile["has_life_shock"]

    dti           = min(emi / max(income, 1), 1.0)
    savings_ratio = min(savings / max(income * 3, 1), 1.0)
    emp_risk      = {"salaried": 0.15, "self_employed": 0.45, "gig": 0.70}.get(emp_type, 0.35)
    shock_risk    = 0.55 if shock else 0.0

    raw_risk = (
        dti               * 0.35 +
        (1 - savings_ratio) * 0.25 +
        emp_risk            * 0.20 +
        shock_risk          * 0.20
    )
    # Gaussian noise so identical profiles look different
    raw_risk = max(0.0, min(1.0, raw_risk + random.gauss(0, 0.07)))

    if raw_risk >= 0.60:
        return "high_risk", raw_risk
    elif raw_risk >= 0.35:
        return "at_risk", raw_risk
    else:
        return "healthy", raw_risk


# ── Customer builder ──────────────────────────────────────────────────────────
def make_customer(idx: int) -> dict[str, Any]:
    """
    ▶ FIX 2: Signature changed from make_customer(idx, risk_level)
    to make_customer(idx).
    Risk is no longer passed in — it is computed from the financial profile.

    BEFORE: make_customer(i, "healthy") / make_customer(i, "at_risk") etc.
    AFTER:  make_customer(i) — builds financial profile first, then derives risk.
    """
    # Realistic Indian income distribution
    income_bracket = random.choices(
        ["low", "mid", "high", "very_high"],
        weights=[35, 40, 18, 7],
    )[0]
    income_ranges = {
        "low":       (12_000,  35_000),
        "mid":       (35_000,  90_000),
        "high":      (90_000,  200_000),
        "very_high": (200_000, 500_000),
    }
    monthly_income = random.randint(*income_ranges[income_bracket])

    emp_type = random.choices(
        ["salaried", "self_employed", "gig"],
        weights=[60, 30, 10],
    )[0]

    # DTI varies across all income levels — high earners can be over-leveraged
    emi_ratio = random.choices(
        [0.10, 0.25, 0.40, 0.55, 0.65],
        weights=[20, 30, 25, 15, 10],
    )[0]
    emi_amount = monthly_income * emi_ratio * random.uniform(0.9, 1.1)

    savings_mult = random.choices(
        [0.3, 1.0, 2.5, 5.0, 10.0],
        weights=[15, 30, 30, 18, 7],
    )[0]
    avg_savings = monthly_income * savings_mult * random.uniform(0.8, 1.2)

    # ▶ FIX 2: 15% of ALL customers get a life shock — not just high-risk ones
    has_shock = random.random() < 0.15

    profile = {
        "customer_id":         f"CUST{idx:05d}",
        "full_name":           fake.name(),
        "email":               fake.email(),
        "phone":               fake.phone_number()[:20],
        "monthly_income":      monthly_income,
        "income_bracket":      income_bracket,
        "salary_day":          random.randint(1, 7),
        "emi_amount":          emi_amount,
        "emi_due_day":         random.randint(5, 12),
        "avg_savings_balance": avg_savings,
        "credit_limit":        monthly_income * random.uniform(1.5, 5.0),
        "segment":             random.choice(SEGMENTS),
        "geography":           random.choice(GEOGRAPHIES),
        "employment_status":   emp_type,
        "preferred_channel":   random.choice(CHANNELS),
        "product_mix":         random.choices(
            ["loan_only", "card_only", "both", "both"],
            weights=[25, 20, 45, 10],
        )[0],
        "has_life_shock":      has_shock,
        "salary_irregularity": random.uniform(0, 1) if emp_type != "salaried"
                               else random.uniform(0, 0.2),
    }

    # ▶ FIX 2: derive risk from profile — not from idx
    risk_level, stress_base = compute_risk_from_profile(profile)
    profile["risk_level"]  = risk_level
    profile["stress_base"] = stress_base
    return profile


def build_customers(n: int = 1500) -> list[dict[str, Any]]:
    """
    ▶ FIX 2: BEFORE — risk assigned by ID range:
        range(1, 901)    → "healthy"
        range(901, 1301) → "at_risk"
        range(1301, 1501)→ "high_risk"

    AFTER — all customers built identically via make_customer(i).
    Risk comes out of compute_risk_from_profile() which uses financial data.
    CUST00001 could be high_risk. CUST01499 could be healthy.
    """
    random.seed(42)
    # ▶ FIX 2: single loop — no more ID-range-based risk buckets
    customers = [make_customer(i) for i in range(1, n + 1)]
    random.seed()  # restore randomness

    counts = {"healthy": 0, "at_risk": 0, "high_risk": 0}
    for c in customers:
        counts[c["risk_level"]] += 1
    print(f"  Risk distribution (financial profile, NOT customer ID):")
    print(f"    Healthy   : {counts['healthy']:,}  ({counts['healthy']/n:.0%})")
    print(f"    At Risk   : {counts['at_risk']:,}  ({counts['at_risk']/n:.0%})")
    print(f"    High Risk : {counts['high_risk']:,}  ({counts['high_risk']/n:.0%})")
    return customers


# ── PostgreSQL helpers (UNCHANGED) ────────────────────────────────────────────
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
    import math
    with table.batch_writer() as batch:
        for c in customers:
            # ▶ FIX 2: pulse score derived from stress_base (financial profile)
            # BEFORE: score was picked from hardcoded ranges per risk bucket
            # AFTER:  score calculated from actual stress_base value
            stress  = c["stress_base"]
            pd_prob = min(0.95, max(0.02, stress * 0.9 + random.gauss(0, 0.04)))
            scaled  = 1.0 / (1.0 + math.exp(-10.0 * (pd_prob - 0.30)))
            score   = max(1, min(100, int(round(scaled * 100))))

            if score >= 70:   tier = "red"
            elif score >= 45: tier = "orange"
            elif score >= 25: tier = "yellow"
            else:             tier = "green"

            batch.put_item(Item={
                "customer_id":         c["customer_id"],
                "full_name":           c["full_name"],
                "segment":             c["segment"],
                "geography":           c["geography"],
                "monthly_income":      Decimal(str(c["monthly_income"])),
                "credit_limit":        Decimal(str(round(c["credit_limit"], 2))),
                "outstanding_balance": Decimal(str(round(
                    c["emi_amount"] * random.uniform(12, 36), 2))),
                "credit_utilization":  Decimal(str(round(
                    min(0.99, stress * 0.85 + random.uniform(0, 0.1)), 4))),
                "pulse_score":         score,
                "risk_tier":           tier,
                "pd_probability":      Decimal(str(round(pd_prob, 4))),
                "confidence":          Decimal("1.0"),
                "days_past_due":       (
                    random.randint(0, 15) if c["risk_level"] == "high_risk"
                    else random.randint(0, 3) if c["risk_level"] == "at_risk"
                    else 0),
                "employment_status":   c["employment_status"],
                "preferred_channel":   c["preferred_channel"],
                "top_factor":          (
                    "failed_auto_debit_count" if c.get("has_life_shock")
                    else "salary_delay_days"  if c["risk_level"] == "high_risk"
                    else "credit_utilization_delta"),
                # ▶ FIX 1: use TODAY instead of datetime.now() to stay consistent
                "updated_at":          TODAY.isoformat(),
                "model_version":       "2.0.0",
                "stress_base":         Decimal(str(round(stress, 4))),
            })
            count += 1
    print(f" ✓ {count:,} inserted")

def write_txn_dynamodb(event: TransactionEvent, db, table_name: str) -> None:
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
            getattr(event, "is_lending_app_upi", False) or
            event.payment_status == PaymentStatus.FAILED
        ),
        # ▶ FIX 1: TTL anchored to TODAY not datetime.now()
        "ttl": int((TODAY + timedelta(days=90)).timestamp()),
    })


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION GENERATOR
# ▶ FIX 1: every datetime(...) replaced with safe_datetime(...)
# ▶ FIX 2: stress driven by customer["stress_base"] not risk_level label
# ══════════════════════════════════════════════════════════════════════════════
def generate_month_transactions(
    customer: dict[str, Any],
    year: int,
    month: int,
    month_offset: int = 0,
) -> list[TransactionEvent]:
    events  = []
    cid     = customer["customer_id"]
    income  = customer["monthly_income"]
    sal_day = customer["salary_day"]
    emi_amt = customer["emi_amount"]
    emi_due = customer["emi_due_day"]
    avg_bal = customer["avg_savings_balance"]

    # ▶ FIX 2: BEFORE used risk_level string to gate signals
    # AFTER: uses stress_base (float 0-1) from financial profile
    stress = customer["stress_base"]
    time_factor      = max(0.0, (month_offset + 6) / 6.0)
    effective_stress = min(1.0, stress * (0.4 + 0.6 * time_factor))
    if customer.get("has_life_shock", False) and month_offset >= -2:
        effective_stress = min(1.0, effective_stress * 1.4)

    # Signal 1: Salary credit
    max_delay = int(8 * effective_stress * customer.get("salary_irregularity", 0.1) + 1)
    delay     = random.randint(0, max(0, max_delay)) if effective_stress > 0.2 else 0
    sal_amt   = income * random.uniform(0.94, 1.04) * (1.0 - 0.15 * effective_stress)
    events.append(TransactionEvent(
        customer_id=cid, account_type=AccountType.SAVINGS,
        txn_type=TransactionType.SALARY_CREDIT,
        amount=round(sal_amt),
        merchant_category=MerchantCategory.OTHER,
        payment_channel="NEFT",
        # ▶ FIX 1: safe_datetime() instead of bare datetime()
        txn_timestamp=safe_datetime(year, month,
                                    min(sal_day + delay, 28),
                                    random.randint(9, 12)),
    ))

    # Signal 2: Savings drain
    drain_events = int(effective_stress * 6)
    for _ in range(drain_events):
        drain_pct = random.uniform(0.04, 0.20) * effective_stress
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(500, avg_bal * drain_pct), 2),
            payment_channel="NEFT",
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month,
                                        random.randint(12, 28),
                                        random.randint(10, 18)),
        ))

    # Signal 3: UPI lending apps
    lending_propensity = effective_stress * (emi_amt / max(income, 1))
    num_lending        = int(lending_propensity * 12)
    for _ in range(num_lending):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(1000, min(25000, income * 0.5))
                         * (0.5 + effective_stress), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI",
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month,
                                        random.randint(8, 28),
                                        random.randint(9, 22)),
            is_lending_app_upi=True,
        ))

    # Signal 4: Utility payment latency
    base_util_day = random.randint(5, 10)
    util_delay    = int(effective_stress * 15)
    for _ in range(random.randint(2, 5)):
        util_day = min(base_util_day + util_delay + random.randint(0, 3), 28)
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 5000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month, util_day,
                                        random.randint(9, 20)),
        ))

    # Signal 5: Discretionary contraction
    base_disc = int(12 * (1 - effective_stress * 0.85))
    num_disc  = max(1, base_disc) + random.randint(0, 4)
    disc_amt  = (income * 0.05) * (1 - effective_stress * 0.7)
    for _ in range(num_disc):
        cat = random.choice([MerchantCategory.DINING, MerchantCategory.ENTERTAINMENT,
                             MerchantCategory.SHOPPING, MerchantCategory.DINING])
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(max(50, random.uniform(disc_amt * 0.5, disc_amt * 1.5)), 2),
            merchant_category=cat, payment_channel="UPI",
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month,
                                        random.randint(1, 28),
                                        random.randint(11, 23)),
        ))

    # Signal 6: ATM withdrawals
    base_atm    = random.randint(1, 3)
    stress_atm  = int(effective_stress * 8)
    num_atm     = base_atm + stress_atm
    atm_pool    = [500, 1000, 2000, 5000, 10000]
    atm_weights = [30, 30, 20, 15, 5]
    if effective_stress > 0.5:
        atm_weights = [5, 15, 25, 30, 25]
    for _ in range(num_atm):
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=float(random.choices(atm_pool, weights=atm_weights)[0]),
            payment_channel="ATM",
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month,
                                        random.randint(1, 28),
                                        random.randint(8, 22)),
        ))

    # Signal 7: Failed auto-debit
    dti       = emi_amt / max(income, 1)
    fail_prob = min(0.95, dti * effective_stress * 1.8)
    if customer.get("has_life_shock", False) and month_offset >= -2:
        fail_prob = min(0.95, fail_prob + 0.3)
    emi_failed = random.random() < fail_prob

    events.append(TransactionEvent(
        customer_id=cid, account_type=AccountType.LOAN,
        txn_type=TransactionType.AUTO_DEBIT,
        amount=round(emi_amt),
        payment_channel="ECS",
        payment_status=PaymentStatus.FAILED if emi_failed else PaymentStatus.SUCCESS,
        # ▶ FIX 1
        txn_timestamp=safe_datetime(year, month, emi_due, 8),
        is_auto_debit_failed=emi_failed,
    ))
    if emi_failed and random.random() < 0.65:
        events.append(TransactionEvent(
            customer_id=cid, account_type=AccountType.LOAN,
            txn_type=TransactionType.AUTO_DEBIT,
            amount=round(emi_amt),
            payment_channel="ECS",
            payment_status=PaymentStatus.FAILED,
            # ▶ FIX 1
            txn_timestamp=safe_datetime(year, month,
                                        min(emi_due + random.randint(2, 5), 28),
                                        8),
            is_auto_debit_failed=True,
        ))

    # Cross-env: credit card
    if customer["product_mix"] in ("card_only", "both"):
        credit_util = min(0.95, 0.2 + effective_stress * 0.7)
        card_spend  = customer["credit_limit"] * credit_util
        num_card    = random.randint(5, 20)
        for _ in range(num_card):
            events.append(TransactionEvent(
                customer_id=cid, account_type=AccountType.CREDIT_CARD,
                txn_type=TransactionType.CREDIT_CARD_PAYMENT,
                amount=round(card_spend / num_card * random.uniform(0.5, 1.5), 2),
                merchant_category=random.choice([
                    MerchantCategory.GROCERIES, MerchantCategory.DINING,
                    MerchantCategory.SHOPPING, MerchantCategory.OTHER,
                ]),
                payment_channel="POS",
                # ▶ FIX 1
                txn_timestamp=safe_datetime(year, month,
                                            random.randint(1, 28),
                                            random.randint(10, 21)),
            ))

    # ▶ FIX 1: final safety filter — drop any event still beyond TODAY
    events = [e for e in events if e.txn_timestamp <= TODAY]

    return sorted(events, key=lambda e: e.txn_timestamp)


# ── Real-time transaction generator ──────────────────────────────────────────
TXN_WEIGHTS = {
    "upi_payment":    30,
    "atm_withdrawal": 15,
    "utility_payment":12,
    "credit_card":    12,
    "salary_credit":   8,
    "auto_debit":      8,
    "savings_drain":   8,
    "lending_upi":     7,
}
TXN_TYPES = list(TXN_WEIGHTS.keys())
TXN_PROBS = [w / sum(TXN_WEIGHTS.values()) for w in TXN_WEIGHTS.values()]

def generate_random_txn(customer: dict[str, Any], now: datetime) -> TransactionEvent:
    """
    ▶ FIX 2: BEFORE used risk_level string ("high_risk", "at_risk") to
    set transaction type weights.
    AFTER: uses stress_base (float) which is continuous and financial-profile-based.

    ▶ FIX 1: Real-time transactions use `now` (current time) by default.
    The salary_credit case subtracts time (goes backwards, never forwards).
    """
    cid     = customer["customer_id"]
    income  = customer["monthly_income"]
    emi     = customer["emi_amount"]
    # ▶ FIX 2: use stress_base, not risk_level label
    stress  = customer["stress_base"]
    avg_bal = customer["avg_savings_balance"]

    # ▶ FIX 2: weights shift continuously based on stress_base float
    weights = {
        "upi_payment":    max(5,  int(30 * (1 - stress * 0.6))),
        "atm_withdrawal": max(5,  int(15 + stress * 15)),
        "utility_payment":10,
        "credit_card":    max(5,  int(12 * (1 + stress * 0.5))),
        "salary_credit":  8,
        "auto_debit":     8,
        "savings_drain":  max(2,  int(8  * stress * 2)),
        "lending_upi":    max(1,  int(7  * stress * 3)),
    }
    txn_kind = random.choices(list(weights.keys()),
                               weights=list(weights.values()), k=1)[0]

    if txn_kind == "upi_payment":
        spend = income * random.uniform(0.01, 0.06) * (1 - stress * 0.5)
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(max(50, spend), 2),
            merchant_category=random.choice([
                MerchantCategory.DINING, MerchantCategory.SHOPPING,
                MerchantCategory.GROCERIES, MerchantCategory.ENTERTAINMENT]),
            payment_channel="UPI", txn_timestamp=now,
        )
    elif txn_kind == "atm_withdrawal":
        pool = [500, 1000, 2000, 5000, 10000]
        wts  = [30, 30, 20, 15, 5]
        if stress > 0.5:
            wts = [5, 15, 25, 30, 25]
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.ATM_WITHDRAWAL,
            amount=float(random.choices(pool, weights=wts)[0]),
            payment_channel="ATM", txn_timestamp=now,
        )
    elif txn_kind == "utility_payment":
        fail = stress > 0.55 and random.random() < 0.2
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UTILITY_PAYMENT,
            amount=round(random.uniform(300, 5000), 2),
            merchant_category=MerchantCategory.UTILITIES,
            payment_channel="UPI",
            payment_status=PaymentStatus.FAILED if fail else PaymentStatus.SUCCESS,
            txn_timestamp=now,
        )
    elif txn_kind == "credit_card":
        util_amt = customer["credit_limit"] * min(0.95, 0.2 + stress * 0.6)
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.CREDIT_CARD,
            txn_type=TransactionType.CREDIT_CARD_PAYMENT,
            amount=round(random.uniform(util_amt * 0.02, util_amt * 0.15), 2),
            merchant_category=random.choice([
                MerchantCategory.GROCERIES, MerchantCategory.DINING,
                MerchantCategory.SHOPPING, MerchantCategory.OTHER]),
            payment_channel="POS", txn_timestamp=now,
        )
    elif txn_kind == "salary_credit":
        # ▶ FIX 1: delay goes BACKWARDS in time (subtracts), never forward
        delay_mins = int(stress * customer.get("salary_irregularity", 0.1) * 3 * 24 * 60)
        # ▶ FIX 1: cap result to TODAY after subtracting
        ts = cap_to_today(now - timedelta(minutes=random.randint(0, max(1, delay_mins))))
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SALARY_CREDIT,
            amount=round(income * random.uniform(0.93, 1.03) * (1 - stress * 0.1)),
            merchant_category=MerchantCategory.OTHER,
            payment_channel="NEFT", txn_timestamp=ts,
        )
    elif txn_kind == "auto_debit":
        dti       = emi / max(income, 1)
        # ▶ FIX 2: fail probability from stress_base (continuous) not risk label
        fail_prob = min(0.95, dti * stress * 1.8 +
                        (0.3 if customer.get("has_life_shock") else 0))
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
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.SAVINGS_WITHDRAWAL,
            amount=round(max(500, avg_bal * random.uniform(0.05, 0.22) * stress), 2),
            payment_channel="NEFT", txn_timestamp=now,
        )
    else:  # lending_upi
        return TransactionEvent(
            customer_id=cid, account_type=AccountType.SAVINGS,
            txn_type=TransactionType.UPI_DEBIT,
            amount=round(random.uniform(2000, min(25000, income * 0.6)), 2),
            merchant_category=MerchantCategory.LENDING_APP,
            payment_channel="UPI", txn_timestamp=now,
            is_lending_app_upi=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Historical load
# ▶ FIX 1: range ends at TODAY — skips any month that is in the future
# ══════════════════════════════════════════════════════════════════════════════
def run_historical(months_back: int = 6,
                   db_conn=None, dynamo_db=None) -> None:
    customers = build_customers()

    if db_conn:
        seed_customers_postgres(customers, db_conn)
    if dynamo_db:
        seed_customers_dynamodb(customers, dynamo_db)

    # ▶ FIX 1: anchor end of range to TODAY, not datetime.now()
    total = 0

    for month_offset in range(-months_back, 1):
        t0     = time.time()
        target = TODAY + timedelta(days=30 * month_offset)
        year, month = target.year, target.month

        # ▶ FIX 1: skip month entirely if its start is after today
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month_start > TODAY:
            print(f"  {year}-{month:02d} ... skipped (future)")
            continue

        print(f"  {year}-{month:02d} ...", end="", flush=True)

        all_rows = []
        for customer in customers:
            events = generate_month_transactions(customer, year, month, month_offset)
            # ▶ FIX 1: belt-and-suspenders — drop any remaining future events
            events = [e for e in events if e.txn_timestamp <= TODAY]
            all_rows.extend([event_to_pg_row(e) for e in events])

        if db_conn:
            bulk_insert_transactions(all_rows, db_conn)

        elapsed = time.time() - t0
        total  += len(all_rows)
        print(f" {len(all_rows):,} rows  ({elapsed:.1f}s)")

    print(f"\n  ✓ Done: {total:,} transactions — all dated ≤ {TODAY.date()}")
    print("  Next: python -m ingestion.consumers.feature_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Real-time random stream with live stress signal tracking
# ══════════════════════════════════════════════════════════════════════════════

# Stress signal definitions — what counts as a stress event and its severity
_STRESS_SIGNALS = {
    "failed_auto_debit": {
        "check":    lambda evt: getattr(evt, "is_auto_debit_failed", False),
        "label":    "Failed Auto-Debit (EMI missed)",
        "severity": "HIGH",
        "impact":   "+8 pulse score",
    },
    "lending_app_upi": {
        "check":    lambda evt: getattr(evt, "is_lending_app_upi", False),
        "label":    "Borrowed from Lending App",
        "severity": "HIGH",
        "impact":   "+7 pulse score",
    },
    "failed_utility": {
        "check":    lambda evt: (
            evt.txn_type == TransactionType.UTILITY_PAYMENT
            and evt.payment_status == PaymentStatus.FAILED
        ),
        "label":    "Failed Utility Payment",
        "severity": "MEDIUM",
        "impact":   "+5 pulse score",
    },
    "large_atm": {
        "check":    lambda evt: (
            evt.txn_type == TransactionType.ATM_WITHDRAWAL
            and float(evt.amount) >= 5000
        ),
        "label":    "Large ATM Withdrawal (cash hoarding)",
        "severity": "MEDIUM",
        "impact":   "+4 pulse score",
    },
    "large_savings_drain": {
        "check":    lambda evt: (
            evt.txn_type == TransactionType.SAVINGS_WITHDRAWAL
            and float(evt.amount) >= 10000
        ),
        "label":    "Large Savings Withdrawal",
        "severity": "MEDIUM",
        "impact":   "+4 pulse score",
    },
}


def _classify_stress(evt: TransactionEvent) -> tuple[str, dict] | None:
    """Return (signal_key, signal_def) if this event is a stress signal, else None."""
    for key, sig in _STRESS_SIGNALS.items():
        try:
            if sig["check"](evt):
                return key, sig
        except Exception:
            pass
    return None


def _print_stress_summary(stress_log: dict, total: int, stress_total: int) -> None:
    """
    Prints a full ranked summary of which customers fired stress signals.
    Called on Ctrl+C or every 500 total transactions.
    """
    from collections import defaultdict

    print("\n" + "=" * 70)
    print("  STRESS SIGNAL REPORT — customers to verify in dashboard + DB")
    print(f"  Total transactions : {total:,}")
    print(f"  Stress signals     : {stress_total:,}")
    print(f"  Affected customers : {len(stress_log):,}")
    print("=" * 70)

    if not stress_log:
        print("  No stress signals fired yet.")
        return

    # Sort by number of stress events descending
    sorted_log = sorted(stress_log.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\n  {'Rank':<5} {'Customer ID':<14} {'Signals':>8}  "
          f"{'HIGH':>5}  {'MED':>5}  Top Signal")
    print("  " + "-" * 68)

    for rank, (cid, events) in enumerate(sorted_log, 1):
        high = sum(1 for e in events if e["severity"] == "HIGH")
        med  = sum(1 for e in events if e["severity"] == "MEDIUM")
        top  = events[0]["label"]
        marker = ">>>" if high > 0 else "  >"
        print(f"  {marker} {rank:<4} {cid:<14} {len(events):>8}  "
              f"{high:>5}  {med:>5}  {top}")

    # Detailed breakdown of top 10
    print(f"\n  TOP 10 DETAILED (verify these in dashboard + PostgreSQL):")
    print("  " + "-" * 68)
    for rank, (cid, events) in enumerate(sorted_log[:10], 1):
        signal_counts: dict[str, int] = {}
        for e in events:
            signal_counts[e["label"]] = signal_counts.get(e["label"], 0) + 1
        last_ts  = max(e["ts"] for e in events)
        total_amt= sum(e["amount"] for e in events)
        print(f"\n  #{rank} {cid}  ({len(events)} signals, "
              f"last: {last_ts.strftime('%H:%M:%S')}, "
              f"total: Rs.{total_amt:,.0f})")
        for label, cnt in sorted(signal_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"      {label:<46} x{cnt}")

    # Verification SQL
    top5 = [cid for cid, _ in sorted_log[:5]]
    ids_str = "', '".join(top5)
    print(f"""
  VERIFICATION SQL — paste into PostgreSQL to confirm:

    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE txn_type='auto_debit'
                         AND payment_status='failed')          AS failed_emis,
        COUNT(*) FILTER (WHERE merchant_category='lending_app') AS lending_txns,
        COUNT(*) FILTER (WHERE txn_type='atm_withdrawal'
                         AND amount>=5000)                     AS large_atm,
        COUNT(*) FILTER (WHERE payment_status='failed')        AS total_failed,
        MAX(txn_timestamp)                                     AS last_txn
    FROM transactions
    WHERE customer_id IN ('{ids_str}')
    GROUP BY customer_id
    ORDER BY failed_emis DESC;
""")
    print("  Search these IDs in the dashboard → Score Impact tab")
    print("  to see how each transaction moved the Pulse Score.")
    print("=" * 70 + "\n")


def run_realtime(db_conn=None, dynamo_db=None,
                 interval_seconds: float = 0.5) -> None:
    from collections import defaultdict
    from ingestion.producers.transaction_producer import TransactionProducer

    customers    = build_customers()
    dynamo_table = getattr(settings, "dynamodb_table_transactions",
                           "sentinel-transactions")
    total         = 0
    stress_total  = 0
    # stress_log: customer_id → list of stress event dicts
    stress_log: dict[str, list[dict]] = defaultdict(list)

    print("\n" + "=" * 60)
    print("REAL-TIME RANDOM STREAM")
    print("  Risk from financial profile — any ID can be any tier")
    print("  Stress signals printed live as they fire")
    print("  Press Ctrl+C for full customer stress report")
    print("=" * 60)
    print(f"\n  {'Time':>8}  {'Customer ID':<14} {'Signal':<42} {'Amount':>12}  Sev")
    print("  " + "-" * 82)

    with TransactionProducer() as producer:
        while True:
            try:
                now   = datetime.now(timezone.utc)
                batch = random.sample(customers, k=random.randint(5, 25))

                batch_events = []
                batch_rows   = []

                for customer in batch:
                    jitter = timedelta(seconds=random.uniform(0, 30))
                    # ▶ FIX 1: cap jitter so it never exceeds TODAY
                    ts  = cap_to_today(now + jitter)
                    evt = generate_random_txn(customer, ts)
                    batch_events.append(evt)
                    batch_rows.append(event_to_pg_row(evt))

                    # ── Stress signal detection ───────────────────────────────
                    result = _classify_stress(evt)
                    if result:
                        sig_key, sig_def = result
                        stress_total += 1
                        amt = float(evt.amount)

                        # Record for summary report
                        stress_log[customer["customer_id"]].append({
                            "signal_key": sig_key,
                            "label":      sig_def["label"],
                            "severity":   sig_def["severity"],
                            "impact":     sig_def["impact"],
                            "amount":     amt,
                            "ts":         evt.txn_timestamp,
                        })

                        # Print live stress event
                        sev_marker = ">>>" if sig_def["severity"] == "HIGH" else " > "
                        print(
                            f"  {sev_marker} "
                            f"{evt.txn_timestamp.strftime('%H:%M:%S'):>8}  "
                            f"{customer['customer_id']:<14} "
                            f"{sig_def['label']:<42} "
                            f"Rs.{amt:>9,.0f}  "
                            f"{sig_def['severity']}"
                        )

                producer.publish_batch(batch_events)

                if db_conn:
                    bulk_insert_transactions(batch_rows, db_conn)

                if dynamo_db:
                    for evt in batch_events:
                        try:
                            write_txn_dynamodb(evt, dynamo_db, dynamo_table)
                        except Exception:
                            pass

                total += len(batch_events)

                # Print a brief progress line every 200 transactions
                if total % 200 == 0:
                    ts_str = now.strftime("%H:%M:%S")
                    print(f"\n  --- [{ts_str}] Total: {total:,} | "
                          f"Stress signals: {stress_total:,} | "
                          f"Affected customers: {len(stress_log):,} ---\n")

                # Print interim summary every 500 transactions
                if total % 500 == 0 and stress_log:
                    _print_stress_summary(stress_log, total, stress_total)

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                # Final summary on exit
                _print_stress_summary(stress_log, total, stress_total)
                print(f"  Stream stopped. Total txns: {total:,} | "
                      f"Stress: {stress_total:,}")
                break
            except Exception as e:
                print(f"  Error (retrying): {e}")
                time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["hist", "realtime", "both"], default="both")
    parser.add_argument("--months",   type=int,   default=6)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("SENTINEL — Transaction Simulator")
    print(f"  Mode    : {args.mode}")
    # ▶ FIX 1 confirmation in startup message
    print(f"  Max date: {TODAY.date()} (no future transactions)")
    # ▶ FIX 2 confirmation in startup message
    print(f"  Risk    : derived from financial profile (not ID range)")
    print("=" * 60 + "\n")

    print("Connecting to PostgreSQL...", end="", flush=True)
    conn = get_db_conn()
    print(" ✓")

    print("Connecting to AWS DynamoDB...", end="", flush=True)
    try:
        dynamo = get_dynamodb()
        dynamo.meta.client.list_tables(Limit=1)
        print(f" ✓ ({settings.aws_region})")
    except Exception as e:
        print(f" ✗ ({e}) — skipping DynamoDB")
        dynamo = None

    try:
        if args.mode in ("hist", "both"):
            print(f"\n{'='*60}")
            print("PHASE 1 — Historical Load (no Kafka, direct SQL)")
            print(f"{'='*60}\n")
            t0 = time.time()
            run_historical(args.months, db_conn=conn, dynamo_db=dynamo)
            print(f"\n  Total time: {time.time()-t0:.1f}s")
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM customers;")
                print(f"  Customers    : {cur.fetchone()[0]:,}")
                cur.execute("SELECT COUNT(*) FROM transactions;")
                print(f"  Transactions : {cur.fetchone()[0]:,}")
                # ▶ FIX 1 — verify no future dates made it through
                cur.execute("SELECT MAX(txn_timestamp) FROM transactions;")
                max_ts = cur.fetchone()[0]
                print(f"  Latest txn   : {max_ts}  (must be ≤ {TODAY.date()})")

        if args.mode in ("realtime", "both"):
            if args.mode == "both":
                print("\nStarting real-time stream in 3s...")
                time.sleep(3)
            run_realtime(db_conn=conn, dynamo_db=dynamo,
                         interval_seconds=args.interval)
    finally:
        conn.close()
        print("\nDone.")