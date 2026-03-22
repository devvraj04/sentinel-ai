# Sentinel AI — v7 Implementation Plan
## Raw Transactions, Simulator Modes, Rich Customer Profiles & Blind Model Detection

**Version:** v7  
**Based on:** v6 codebase audit  
**Scope:** 8 files changed, 2 new files, 1 new migration

---

## What This Plan Fixes

| # | Problem | File(s) |
|---|---------|---------|
| 1 | Simulator has no `--mode` argument — historical and real-time are the same loop | `scripts/simulate_transactions.py` |
| 2 | Historical data is being scored by the model — it should only build the transaction log | `scripts/simulate_transactions.py` |
| 3 | `insert_transaction()` stores only 8 fields — missing sender/receiver, balance tracking, balance delta | `scripts/simulate_transactions.py` |
| 4 | `TransactionEvent` still passes `txn_type` pre-classified — model should infer this from raw counterparty facts | `ingestion/schemas/transaction_event.py` |
| 5 | `customers` table is minimal — missing occupation, employer, current balance, salary date, loan details | `data_warehouse/schemas/init.sql` |
| 6 | `simulate_transactions.py` does not write to the `accounts` table | `scripts/simulate_transactions.py` |
| 7 | `build_training_data.py` missing `salary_delay_zscore`, `atm_spend_zscore` Z-score features | `models/training_pipelines/build_training_data.py` |
| 8 | `FEATURE_COLS` in `train_lgbm.py` missing `atm_spend_zscore` | `models/lightgbm/train_lgbm.py` |

---

## The Mental Model — What Changes

### Before (v6)
```
Simulator generates "lending_app" kind
  → sets txn_type=UPI_DEBIT, merchant_category=LENDING_APP
  → also calls score_customer() on every transaction
  → model scores against 0 history on first run
```

### After (v7)
```
Simulator --mode seed (historical):
  → Generates realistic Indian bank transactions
  → Each transaction has: sender_id, receiver_id, amount, balance_before,
    balance_after, balance_change_pct, payment_status, platform
  → txn_type is NOT passed — transaction_classifier.py infers it from
    receiver counterparty name/VPA pattern
  → Scores NOTHING — only writes transaction history
  → Run until each customer has 90 days of transactions

scripts/build_baselines.py (run once after seed):
  → Reads transaction history
  → Computes per-customer mean balance, salary day, EMI success rate, etc.
  → Writes to customer_baseline table

Simulator --mode realtime:
  → Same transaction format as seed
  → feature_pipeline reads each new transaction
  → Loads customer baseline from PostgreSQL
  → Computes Z-scores vs that customer's personal baseline
  → LightGBM scores — model detects stress itself from:
      * Large balance drop relative to that customer's normal
      * Receiver VPA matching lending app keyword
      * Payment failure on known EMI day
      * ATM spike above that customer's own average
```

---

## File 1: `data_warehouse/schemas/migrate_004_rich_customers.sql` *(NEW)*

**Purpose:** Extend the `customers` table with full Indian banking customer profile fields and add a running balance tracker. This replaces `migrate_002` additions which were partial.

```sql
-- ────────────────────────────────────────────────────────────────────────────
-- Migration 004 — Rich Customer Profile
--
-- Extends customers table with Indian banking context:
--   - Occupation and employer (for income stability signals)
--   - Current live savings balance (updated after every transaction)
--   - Salary date and variability
--   - Loan details (outstanding principal, EMI, due date)
--   - Credit card limit and current utilisation
--   - UPI VPA (used as sender/receiver ID in transactions)
--
-- Apply:
--   docker cp data_warehouse/schemas/migrate_004_rich_customers.sql sentinel-postgres:/tmp/m4.sql
--   docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m4.sql
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE customers
    ADD COLUMN IF NOT EXISTS occupation              VARCHAR(100),
    ADD COLUMN IF NOT EXISTS employer_name           VARCHAR(200),
    ADD COLUMN IF NOT EXISTS employer_type           VARCHAR(50)  DEFAULT 'private',
    -- salary_type: 'salaried', 'self_employed', 'gig', 'retired', 'student'
    ADD COLUMN IF NOT EXISTS salary_type             VARCHAR(30)  DEFAULT 'salaried',
    -- expected_salary_day: 1-31, typical day of month salary arrives
    ADD COLUMN IF NOT EXISTS expected_salary_day     INTEGER      DEFAULT 3,
    -- salary_variability: 0.0=fixed salary, 1.0=highly irregular (gig workers)
    ADD COLUMN IF NOT EXISTS salary_variability      NUMERIC(4,2) DEFAULT 0.1,

    -- Live running balance — updated after every transaction
    -- This is the ACTUAL current savings account balance, not a static estimate
    ADD COLUMN IF NOT EXISTS current_savings_balance NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS balance_updated_at      TIMESTAMPTZ,

    -- Loan details
    ADD COLUMN IF NOT EXISTS loan_outstanding        NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS loan_original_amount    NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS loan_type               VARCHAR(50),
    ADD COLUMN IF NOT EXISTS emi_amount              NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS emi_due_day             INTEGER       DEFAULT 5,
    ADD COLUMN IF NOT EXISTS loan_start_date         DATE,
    ADD COLUMN IF NOT EXISTS loan_tenure_months      INTEGER       DEFAULT 36,

    -- Credit card
    ADD COLUMN IF NOT EXISTS credit_limit            NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS credit_outstanding      NUMERIC(15,2) DEFAULT 0,

    -- UPI Virtual Payment Address — used as counterparty_id in transactions
    -- Format: firstname.lastname@bankname (e.g., rahul.sharma@sbi)
    ADD COLUMN IF NOT EXISTS upi_vpa                 VARCHAR(100),

    -- Geography and banking details
    ADD COLUMN IF NOT EXISTS city                    VARCHAR(100),
    ADD COLUMN IF NOT EXISTS state                   VARCHAR(100),
    ADD COLUMN IF NOT EXISTS bank_name               VARCHAR(100) DEFAULT 'SBI',
    ADD COLUMN IF NOT EXISTS account_number_masked   VARCHAR(20),  -- last 4 digits only

    -- Risk profile (set at customer creation, NOT updated by the model)
    ADD COLUMN IF NOT EXISTS risk_segment            VARCHAR(30)  DEFAULT 'standard',
    ADD COLUMN IF NOT EXISTS has_active_loan         BOOLEAN      DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS has_credit_card         BOOLEAN      DEFAULT FALSE;

-- ── Index for balance queries ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_customers_balance
    ON customers(current_savings_balance DESC);

CREATE INDEX IF NOT EXISTS idx_customers_upi_vpa
    ON customers(upi_vpa);

COMMENT ON COLUMN customers.current_savings_balance IS
    'Live running balance updated by simulate_transactions after every transaction.
     Used as balance_before for the next transaction on this account.
     The feature pipeline reads this to compute balance_zscore.';

COMMENT ON COLUMN customers.upi_vpa IS
    'Customer UPI Virtual Payment Address. Used as sender_id when the customer
     initiates a payment, and as receiver_id when they receive a payment.
     The transaction classifier uses this to identify self-transfers vs P2P vs merchant.';
```

---

## File 2: `data_warehouse/schemas/migrate_005_rich_transactions.sql` *(NEW)*

**Purpose:** Extend the `transactions` table with sender/receiver fields, balance tracking, and balance change metrics. The model uses these raw fields — it never receives a pre-classified label.

```sql
-- ────────────────────────────────────────────────────────────────────────────
-- Migration 005 — Rich Transaction Schema
--
-- Adds sender/receiver identity, balance before/after, balance change percent.
-- The model receives ONLY these raw facts — it infers whether the transaction
-- is stress-inducing from patterns in the data, not from pre-set labels.
--
-- KEY DESIGN PRINCIPLE:
--   The transactions table stores FACTS, not interpretations.
--   - sender_id and receiver_id are UPI VPAs or bank account references
--   - balance_before and balance_after are actual account balances
--   - balance_change_pct = (balance_after - balance_before) / balance_before
--   - payment_status (success/failed) is a fact, not a stress label
--   - The model infers: "large drop + receiver matches lending VPA = stress"
--
-- Apply:
--   docker cp data_warehouse/schemas/migrate_005_rich_transactions.sql sentinel-postgres:/tmp/m5.sql
--   docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m5.sql
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE transactions
    -- Sender and receiver (raw UPI VPA or account reference)
    -- sender_id = customer's UPI VPA for debits, payer VPA for credits
    -- receiver_id = payee UPI VPA for debits, customer's VPA for credits
    ADD COLUMN IF NOT EXISTS sender_id              VARCHAR(150),
    ADD COLUMN IF NOT EXISTS sender_name            VARCHAR(200),
    ADD COLUMN IF NOT EXISTS receiver_id            VARCHAR(150),
    ADD COLUMN IF NOT EXISTS receiver_name          VARCHAR(200),

    -- Balance tracking on the customer's PRIMARY savings account
    -- Never negative — enforced at application layer
    ADD COLUMN IF NOT EXISTS balance_before         NUMERIC(15,2),
    ADD COLUMN IF NOT EXISTS balance_after          NUMERIC(15,2),

    -- Derived balance metrics — computed at insert time, never by the model
    -- balance_change_pct: negative = balance dropped, positive = balance grew
    -- e.g. -0.40 means balance dropped 40% from this transaction
    ADD COLUMN IF NOT EXISTS balance_change_pct     NUMERIC(8,4),

    -- Payment platform (raw fact, not label)
    -- UPI / NEFT / IMPS / RTGS / NACH / BBPS / POS / ATM / NetBanking
    ADD COLUMN IF NOT EXISTS platform               VARCHAR(30) DEFAULT 'unknown',

    -- Reference number for traceability (UTR for NEFT/RTGS, RRN for UPI)
    ADD COLUMN IF NOT EXISTS reference_number       VARCHAR(60);

-- Index for balance analysis
CREATE INDEX IF NOT EXISTS idx_txn_balance_after
    ON transactions(customer_id, balance_after, txn_timestamp DESC)
    WHERE balance_after IS NOT NULL;

-- Index for sender/receiver lookups (counterparty pattern analysis)
CREATE INDEX IF NOT EXISTS idx_txn_receiver
    ON transactions(receiver_id, txn_timestamp DESC)
    WHERE receiver_id IS NOT NULL;

COMMENT ON COLUMN transactions.sender_id IS
    'Raw UPI VPA or bank account of the sender. For customer-initiated debits,
     this is the customer own VPA. The transaction_classifier.py uses
     receiver_id patterns to infer transaction purpose — no pre-set labels.';

COMMENT ON COLUMN transactions.balance_change_pct IS
    'Computed at insert time: (balance_after - balance_before) / ABS(balance_before).
     Negative = balance dropped. The feature pipeline uses this directly.
     A drop of -0.60 for a low-balance customer is more alarming than -0.05
     for a high-balance customer — the model learns this from the Z-scores.';
```

---

## File 3: `ingestion/schemas/transaction_event.py` *(MODIFY)*

**Purpose:** Replace the pre-classified `txn_type` field with raw `sender_id`, `receiver_id`, `balance_before`, `balance_after`, `balance_change_pct`. The `transaction_classifier.py` infers purpose from counterparty patterns — the model never sees a pre-assigned label like `SALARY_CREDIT` or `AUTO_DEBIT`.

**Replace the entire file with:**

```python
"""
ingestion/schemas/transaction_event.py
──────────────────────────────────────────────────────────────────────────────
Defines the canonical shape of every transaction event flowing through Sentinel.

DESIGN PRINCIPLE — RAW FACTS ONLY:
  The TransactionEvent carries only factual, observable data about the
  transaction. It does NOT carry any interpretive labels. The pipeline
  infers meaning from the raw facts.

  REMOVED (were label leakage):
    - txn_type (e.g. SALARY_CREDIT, AUTO_DEBIT) ← pre-classified label
    - merchant_category (e.g. LENDING_APP) ← pre-classified label
    - is_lending_app_upi ← explicit stress flag
    - is_auto_debit_failed ← explicit stress flag

  ADDED (raw facts):
    - sender_id: UPI VPA of the sender (e.g. rahul.sharma@sbi)
    - receiver_id: UPI VPA or account ref of the receiver (e.g. slice@upi)
    - sender_name: human-readable sender name
    - receiver_name: human-readable receiver name (e.g. "Slice Fintech Pvt Ltd")
    - balance_before: savings balance before this transaction
    - balance_after: savings balance after this transaction
    - balance_change_pct: (after - before) / abs(before)
    - platform: UPI / NEFT / IMPS / ATM / ECS / POS etc.
    - payment_status: success / failed / pending / reversed

  The transaction_classifier.py uses receiver_id patterns (e.g. slice@upi,
  lazypay@upi, bescom@bbps) to infer transaction purpose for FEATURE
  COMPUTATION ONLY — never to set a flag or label on the event itself.

Indian Banking Context:
  - UPI VPA format: name@bankcode (e.g. rahul.sharma@sbi, swiggy@yesbank)
  - NACH/ECS: used for recurring EMI auto-debits
  - BBPS: used for utility bill payments (electricity, gas, water)
  - IMPS/NEFT/RTGS: used for salary credits and large transfers
  - ATM: cash withdrawals — counterparty is the ATM location code
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PaymentStatus(str):
    SUCCESS  = "success"
    FAILED   = "failed"
    PENDING  = "pending"
    REVERSED = "reversed"


class TransactionEvent(BaseModel):
    """
    Single raw transaction event — atomic unit of the entire system.

    This is a FACT RECORD, not a labelled event. It contains exactly what
    the bank ledger contains: who sent, who received, how much, what platform,
    did it succeed, and what were the account balances before and after.

    The model infers EVERYTHING ELSE from these raw facts.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    event_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id:    str = Field(..., min_length=1, max_length=50)
    account_id:     Optional[str] = None

    # ── Counterparty (raw VPA or account reference) ───────────────────────────
    # For a DEBIT: sender = customer, receiver = payee
    # For a CREDIT: sender = payer, receiver = customer
    sender_id:      Optional[str] = None   # UPI VPA or bank account ref
    sender_name:    Optional[str] = None   # e.g. "Rahul Sharma" or "TCS Payroll"
    receiver_id:    Optional[str] = None   # e.g. "slice@upi", "bescom@bbps"
    receiver_name:  Optional[str] = None   # e.g. "Slice Fintech", "BESCOM"

    # ── Transaction facts ─────────────────────────────────────────────────────
    amount:         float = Field(..., gt=0)
    platform:       str = "unknown"        # UPI / NEFT / IMPS / ATM / ECS / BBPS / POS
    payment_status: str = "success"        # success / failed / pending / reversed
    reference_number: Optional[str] = None # UTR (NEFT/RTGS) or RRN (UPI)

    # ── Balance tracking ─────────────────────────────────────────────────────
    # balance_before and balance_after are the customer's PRIMARY savings balance
    # BEFORE and AFTER this transaction is applied.
    # balance_change_pct = (balance_after - balance_before) / abs(balance_before)
    # Negative = balance dropped. Never stored as positive when balance falls.
    balance_before:     Optional[float] = None
    balance_after:      Optional[float] = None
    balance_change_pct: Optional[float] = None  # computed by simulator, stored as fact

    # ── Timing ────────────────────────────────────────────────────────────────
    txn_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Kafka metadata (filled by consumer) ──────────────────────────────────
    kafka_partition: Optional[int] = None
    kafka_offset:    Optional[int] = None
    ingested_at:     Optional[datetime] = None

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)

    @model_validator(mode="after")
    def compute_balance_change(self) -> "TransactionEvent":
        """Auto-compute balance_change_pct if balances are provided but pct is not."""
        if (self.balance_change_pct is None and
                self.balance_before is not None and
                self.balance_after is not None and
                abs(self.balance_before) > 0):
            self.balance_change_pct = round(
                (self.balance_after - self.balance_before) / abs(self.balance_before), 4
            )
        return self

    def to_dict(self) -> dict:
        data = self.model_dump()
        data["txn_timestamp"] = self.txn_timestamp.isoformat()
        if self.ingested_at:
            data["ingested_at"] = self.ingested_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionEvent":
        return cls(**data)
```

---

## File 4: `ingestion/enrichment/transaction_classifier.py` *(MODIFY)*

**Purpose:** This classifier reads the raw `receiver_id`, `sender_id`, `receiver_name`, `platform`, and `payment_status` from the TransactionEvent and returns an **inferred classification dict** used ONLY by the feature pipeline for computing features. It does NOT modify the TransactionEvent. The model never sees these inferences directly — it sees the Z-scores and ratios computed from them.

**Add these functions / update the classify method:**

```python
# The classifier must now work from receiver_id patterns, not txn_type labels.
# It infers:
#   inferred_purpose: "salary" | "emi" | "utility" | "lending_borrow" |
#                     "grocery" | "dining" | "atm_cash" | "p2p_transfer" |
#                     "investment_redeem" | "cc_bill" | "other"
#   inferred_direction: "credit" | "debit"
#   is_likely_emi: bool  (receiver matches known NACH/ECS EMI patterns)
#   is_likely_salary: bool (sender matches payroll patterns + platform=NEFT)
#   is_likely_lending: bool (receiver matches lending app VPA list)
#   is_likely_utility: bool (receiver matches utility biller list)

def classify(
    sender_id: str | None,
    receiver_id: str | None,
    sender_name: str | None,
    receiver_name: str | None,
    platform: str,
    payment_status: str,
    amount: float,
) -> dict:
    """
    Infer transaction purpose from raw counterparty and platform facts.

    Returns a classification dict used by build_feature_vector() ONLY.
    This classification is NOT stored on the TransactionEvent.
    The model never directly reads these fields.

    The inference chain:
    1. receiver_id/name against known VPA databases (lending apps, utilities, etc.)
    2. platform type (ECS → likely EMI, NEFT_credit → likely salary)
    3. Amount patterns (e.g. round ₹X,000 from NACH → EMI)
    4. Keyword matching on receiver_name
    """
    r_id   = (receiver_id   or "").lower()
    s_id   = (sender_id     or "").lower()
    r_name = (receiver_name or "").lower()
    s_name = (sender_name   or "").lower()
    plat   = platform.lower()

    # ── Lending app detection ─────────────────────────────────────────────────
    is_likely_lending = (
        any(k in r_id for k in LENDING_APP_VPAS) or
        any(k in r_name for k in LENDING_APP_KEYWORDS) or
        any(k in r_id for k in LENDING_APP_KEYWORDS)
    )

    # ── Salary detection ──────────────────────────────────────────────────────
    # Salary arrives via NEFT/IMPS from employer, not UPI from individual
    is_likely_salary = (
        plat in ("neft", "imps", "rtgs") and
        (any(k in s_name for k in SALARY_KEYWORDS) or
         any(k in s_id  for k in SALARY_KEYWORDS)) and
        amount > 5000  # salaries are not small amounts
    )

    # ── EMI / auto-debit detection ────────────────────────────────────────────
    # ECS/NACH debits are almost always EMIs
    is_likely_emi = (
        plat in ("ecs", "nach", "mandate") or
        any(k in r_name for k in EMI_KEYWORDS) or
        any(k in r_id   for k in EMI_KEYWORDS)
    )

    # ── Utility bill detection ────────────────────────────────────────────────
    is_likely_utility = (
        plat == "bbps" or
        any(k in r_id   for k in UTILITY_BILLER_KEYWORDS) or
        any(k in r_name for k in UTILITY_BILLER_KEYWORDS)
    )

    # ── ATM cash withdrawal ───────────────────────────────────────────────────
    is_atm = plat == "atm"

    # ── P2P transfer ──────────────────────────────────────────────────────────
    is_p2p = (
        plat == "upi" and
        not is_likely_lending and not is_likely_utility and
        not any(k in r_id for k in list(DINING_KEYWORDS) + list(GROCERY_KEYWORDS))
    )

    # ── Investment redemption ─────────────────────────────────────────────────
    is_investment_redeem = any(k in r_id + r_name for k in INVESTMENT_KEYWORDS)

    # ── Direction ────────────────────────────────────────────────────────────
    # Credit: salary, loan disbursement, P2P received, reversal
    # Debit: all outflows
    is_credit = is_likely_salary or (
        plat in ("neft", "imps", "rtgs") and
        any(k in s_name for k in SALARY_KEYWORDS)
    )

    # ── Inferred purpose ─────────────────────────────────────────────────────
    if is_likely_salary:       purpose = "salary"
    elif is_likely_emi:        purpose = "emi"
    elif is_likely_lending:    purpose = "lending_borrow"
    elif is_likely_utility:    purpose = "utility"
    elif is_atm:               purpose = "atm_cash"
    elif is_investment_redeem: purpose = "investment_redeem"
    else:                      purpose = "other"

    return {
        "inferred_purpose":       purpose,
        "inferred_direction":     "credit" if is_credit else "debit",
        "is_likely_emi":          is_likely_emi,
        "is_likely_salary":       is_likely_salary,
        "is_likely_lending":      is_likely_lending,
        "is_likely_utility":      is_likely_utility,
        "is_likely_atm":          is_atm,
        "is_likely_p2p":          is_p2p,
        "is_likely_investment":   is_investment_redeem,
        "payment_failed":         payment_status == "failed",
    }
```

---

## File 5: `scripts/simulate_transactions.py` *(MODIFY)*

This is the largest change. Three things change:

### 5.1 — Add `--mode` argument

```python
# In the __main__ argparse block, add:
parser.add_argument(
    "--mode",
    choices=["seed", "realtime"],
    default="realtime",
    help=(
        "seed     = historical mode. Generates 90 days of transactions per customer "
        "with NO model scoring. Use this once to build the transaction baseline. "
        "Run build_baselines.py after this completes.\n"
        "realtime = live mode. Scores every transaction against the customer's "
        "personal baseline. Run AFTER seed + build_baselines."
    )
)
```

### 5.2 — Two separate run modes

```python
# Replace run_pipeline() call in __main__ with:
if args.mode == "seed":
    run_seed_pipeline(
        n_customers=args.customers,
        days=args.days,          # new arg: --days 90
        db_conn=conn,
    )
    print("\n✓ Seed complete. Now run: python -m scripts.build_baselines --days 90")
else:
    run_realtime_pipeline(
        n_customers=args.customers,
        delay_ms=args.delay,
        db_conn=conn,
        dynamo_db=dynamo,
    )
```

**`run_seed_pipeline(n_customers, days, db_conn)`:**
- Creates customers, writes them to PostgreSQL (with ALL new fields from migration 004)
- Creates accounts table rows (savings + loan + credit card)
- Generates transactions with backdated timestamps spanning `days` days
- For each transaction: writes to PostgreSQL with all new fields (sender_id, receiver_id, balance_before, balance_after, balance_change_pct, platform)
- Does NOT publish to Kafka
- Does NOT call score_customer()
- Updates `customers.current_savings_balance` after every transaction

**`run_realtime_pipeline(n_customers, delay_ms, db_conn, dynamo_db)`:**
- Loads existing customers from PostgreSQL (created by seed run)
- Generates real-time transactions
- Publishes to Kafka
- Writes to PostgreSQL
- Calls score_customer() after every transaction

### 5.3 — Raw TransactionEvent construction

Remove all calls that set `txn_type` explicitly. Instead, the event carries only raw facts:

```python
# OLD (pre-labelled):
TransactionEvent(
    customer_id=cid,
    txn_type=TransactionType.SALARY_CREDIT,       # ← REMOVE
    merchant_category=MerchantCategory.OTHER,      # ← REMOVE
    amount=52000,
    payment_channel="NEFT",
    payment_status=PaymentStatus.SUCCESS,
)

# NEW (raw facts only):
TransactionEvent(
    customer_id=cid,
    account_id=f"{cid}_SAV",
    sender_id="tcs_payroll@hdfcbank",              # ← ADD: employer VPA/reference
    sender_name="TCS Technologies Pvt Ltd",         # ← ADD: employer name
    receiver_id=customer["upi_vpa"],               # ← ADD: customer's own VPA
    receiver_name=customer["full_name"],
    amount=52000,
    platform="NEFT",                               # ← renamed from payment_channel
    payment_status="success",
    balance_before=state.current_balance,          # ← ADD: balance before credit
    balance_after=state.current_balance + 52000,   # ← ADD: balance after credit
    # balance_change_pct auto-computed by model validator
)
```

### 5.4 — Known counterparty VPA table (within simulate_transactions.py)

```python
# Add at module level — realistic Indian UPI VPAs for simulation
EMPLOYER_VPAS = {
    "tcs":       ("tcs_payroll@hdfcbank",   "TCS Technologies Pvt Ltd"),
    "infosys":   ("infosys_hr@axisbank",    "Infosys Limited"),
    "wipro":     ("wipro_salary@icici",     "Wipro Limited"),
    "generic":   ("employer_salary@neft",   "Employer Payroll"),
}

LENDING_APP_VPAS_SIM = [
    ("slice@upi",        "Slice Fintech Pvt Ltd"),
    ("lazypay@icici",    "LazyPay - PayU Finance"),
    ("kreditbee@upi",    "KreditBee Inc"),
    ("moneyview@upi",    "Money View - Whizdm Innovations"),
    ("navi@upi",         "Navi Technologies"),
    ("mpokket@upi",      "mPokket - Bengaluru Finance"),
    ("cashe@upi",        "CASHe - Bhanix Finance"),
]

UTILITY_VPAS_SIM = [
    ("bescom@bbps",      "BESCOM - Bangalore Electricity"),
    ("jio@bbps",         "Reliance Jio Infocomm"),
    ("airtel@bbps",      "Airtel Payments"),
    ("mahanagar_gas@bbps","Mahanagar Gas Ltd"),
    ("tata_power@bbps",  "Tata Power Company"),
]

EMI_VPAS_SIM = [
    ("hdfc_loan@ecs",    "HDFC Bank Home Loan"),
    ("sbi_car@nach",     "SBI Car Loan - ECS"),
    ("bajaj_personal@nach", "Bajaj Finserv EMI"),
    ("axis_loan@ecs",    "Axis Bank Personal Loan"),
]

ATM_LOCATIONS = [
    ("ATM001@sbi",       "SBI ATM - Koramangala"),
    ("ATM002@hdfc",      "HDFC ATM - MG Road"),
    ("ATM003@icici",     "ICICI ATM - Indiranagar"),
]

GROCERY_VPAS_SIM = [
    ("bigbasket@upi",    "BigBasket"),
    ("blinkit@upi",      "Blinkit"),
    ("jiomart@upi",      "JioMart"),
    ("dmart@upi",        "D-Mart"),
]

DINING_VPAS_SIM = [
    ("swiggy@icici",     "Swiggy"),
    ("zomato@hdfc",      "Zomato"),
    ("dominos@upi",      "Domino's Pizza"),
]
```

### 5.5 — Update `insert_transaction()` to write new fields

```python
def insert_transaction(evt: TransactionEvent, conn) -> None:
    """Write transaction to PostgreSQL with all raw fact fields."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transactions
                (customer_id, account_id,
                 sender_id, sender_name, receiver_id, receiver_name,
                 amount, platform, payment_status,
                 balance_before, balance_after, balance_change_pct,
                 reference_number, txn_timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            evt.customer_id,
            evt.account_id,
            evt.sender_id,
            evt.sender_name,
            evt.receiver_id,
            evt.receiver_name,
            float(evt.amount),
            evt.platform,
            evt.payment_status,
            evt.balance_before,
            evt.balance_after,
            evt.balance_change_pct,
            evt.reference_number,
            evt.txn_timestamp,
        ))
    conn.commit()
```

### 5.6 — Update `ensure_customer_postgres()` to write all new fields

```python
def ensure_customer_postgres(customer: dict, conn) -> None:
    """Write full customer profile including all baseline-required fields."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO customers
                (customer_id, full_name, email, phone,
                 segment, geography, employment_status,
                 salary_type, occupation, employer_name, employer_type,
                 monthly_income, expected_salary_day, salary_variability,
                 current_savings_balance,
                 loan_outstanding, loan_original_amount, loan_type,
                 emi_amount, emi_due_day, loan_tenure_months,
                 credit_limit, credit_outstanding,
                 upi_vpa, city, state, bank_name,
                 has_active_loan, has_credit_card, risk_segment)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (customer_id) DO UPDATE SET
                current_savings_balance = EXCLUDED.current_savings_balance,
                loan_outstanding        = EXCLUDED.loan_outstanding,
                credit_outstanding      = EXCLUDED.credit_outstanding,
                updated_at              = NOW()
        """, (
            customer["customer_id"],
            customer["full_name"],
            customer["email"],
            customer["phone"],
            customer["segment"],
            customer["geography"],
            customer["employment_status"],
            customer["employment_status"],        # salary_type mirrors employment_status
            customer.get("occupation", "Unknown"),
            customer.get("employer_name", ""),
            "private" if customer["employment_status"] == "salaried" else "self",
            float(customer["monthly_income"]),
            int(customer["salary_day"]),
            float(customer.get("salary_irregularity", 0.1)),
            float(customer["avg_savings_balance"]),  # initial balance
            float(customer["emi_amount"] * customer.get("loan_tenure_months", 24)),  # outstanding ≈ EMI × remaining
            float(customer["emi_amount"] * 36),   # original amount
            customer.get("loan_type", "personal_loan"),
            float(customer["emi_amount"]),
            int(customer["emi_due_day"]),
            36,
            float(customer["credit_limit"]),
            0.0,  # fresh customer, no CC outstanding
            customer["upi_vpa"],
            customer["geography"],
            "Maharashtra",  # default state — extend with proper mapping
            customer.get("bank_name", "SBI"),
            customer["emi_amount"] > 0,
            customer["credit_limit"] > 0,
            customer.get("risk_level", "standard"),
        ))
    conn.commit()
```

### 5.7 — Update `make_customer()` to add new fields

```python
# In make_customer(), add these to the profile dict:

OCCUPATIONS = {
    "salaried":      ["Software Engineer", "Bank Manager", "Teacher", "Doctor",
                       "Accountant", "HR Manager", "Sales Executive", "Engineer"],
    "self_employed": ["Shop Owner", "Consultant", "Freelancer", "Contractor",
                       "Trader", "Restaurant Owner", "Tutor"],
    "gig":           ["Delivery Executive", "Cab Driver", "Gig Worker"],
}

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra",
         "Punjab National Bank", "Bank of Baroda", "Canara Bank"]

LOAN_TYPES = ["personal_loan", "home_loan", "car_loan", "education_loan", "gold_loan"]

# Add to profile dict:
profile["occupation"]    = random.choice(OCCUPATIONS.get(emp, ["Employee"]))
profile["employer_name"] = fake.company() if emp == "salaried" else profile["full_name"]
profile["bank_name"]     = random.choice(BANKS)
profile["loan_type"]     = random.choice(LOAN_TYPES)
profile["loan_tenure_months"] = random.choice([12, 24, 36, 48, 60])
profile["upi_vpa"]       = (
    f"{profile['full_name'].lower().replace(' ', '.')[:15]}"
    f"@{profile['bank_name'].lower().replace(' ', '')[:8]}"
)
```

---

## File 6: `ingestion/consumers/feature_pipeline.py` *(MODIFY)*

**Purpose:** Update `_compute_and_store()` to call `transaction_classifier.classify()` on each transaction to get inferred purpose, and use these inferences to compute features — never store them as labels.

```python
# In _compute_and_store(), after building the DataFrame:

from ingestion.enrichment.transaction_classifier import classify

# Classify each transaction based on raw facts (not stored as labels)
def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transaction_classifier to each row and add inferred feature columns.
    These columns are for FEATURE COMPUTATION ONLY — never returned to the model.
    """
    df = df.copy()
    classifications = df.apply(
        lambda r: classify(
            sender_id=r.get("sender_id"),
            receiver_id=r.get("receiver_id"),
            sender_name=r.get("sender_name"),
            receiver_name=r.get("receiver_name"),
            platform=r.get("platform", "unknown"),
            payment_status=r.get("payment_status", "success"),
            amount=float(r.get("amount", 0)),
        ), axis=1
    )
    df["_is_lending"]   = classifications.apply(lambda c: c["is_likely_lending"])
    df["_is_salary"]    = classifications.apply(lambda c: c["is_likely_salary"])
    df["_is_emi"]       = classifications.apply(lambda c: c["is_likely_emi"])
    df["_is_utility"]   = classifications.apply(lambda c: c["is_likely_utility"])
    df["_is_atm"]       = classifications.apply(lambda c: c["is_likely_atm"])
    df["_payment_failed"] = classifications.apply(lambda c: c["payment_failed"])
    # Add balance_change_pct if missing (old transactions without it)
    if "balance_change_pct" not in df.columns:
        df["balance_change_pct"] = None
    return df
```

---

## File 7: `sagemaker/inference.py` — `build_feature_vector()` *(MODIFY)*

**Purpose:** Update all feature computations to use `_` prefixed enrichment columns (from classifier) instead of the old `is_lending_app_upi` / `txn_type` labels. The function signature must also accept `db_conn` for baseline loading (already done in v6).

```python
# Replace ALL references to old label columns:

# OLD:
lend_s = csum(df_short, "is_lending_app_upi", True)
# NEW:
lend_s = df_short[df_short["_is_lending"] == True]["amount"].sum() if "_is_lending" in df_short.columns else \
         df_short[df_short.get("receiver_id", "").str.contains("|".join(LENDING_APP_KEYWORDS), na=False)]["amount"].sum()

# OLD:
fd14 = df_short[(df_short["txn_type"]=="auto_debit") & (df_short["payment_status"]=="failed")]
# NEW:
fd14 = df_short[df_short.get("_payment_failed", False) == True] if "_payment_failed" in df_short.columns else \
       df_short[df_short["payment_status"] == "failed"]

# OLD (hardcoded threshold):
fv["flag_salary"] = 1.0 if delay > 7 else 0.0
# NEW (personalised):
fv["flag_salary"] = 1.0 if delay > baseline["salary_delay_threshold"] else 0.0
```

---

## File 8: `models/training_pipelines/build_training_data.py` *(MODIFY)*

**Purpose:** Add the missing `salary_delay_zscore` and `atm_spend_zscore` Z-score feature simulations which were missing in v6.

```python
# In simulate_zscores() or equivalent function, ADD:

# salary_delay_zscore: how many std deviations late the salary is vs customer's norm
salary_delay_zscore = np.clip(
    (core["salary_delay_days"] - 1.2) / 2.5 +   # (value - pop_mean) / pop_std
    RNG.normal(0, 0.2, n),
    -5, 5
)
core["salary_delay_zscore"] = salary_delay_zscore

# atm_spend_zscore: deviation in ATM spending from customer's personal average
atm_spend_zscore = np.clip(
    np.log1p(core["atm_withdrawal_spike"].values) +
    RNG.normal(0, 0.25, n),
    -5, 5
)
core["atm_spend_zscore"] = atm_spend_zscore

# lending_spend_zscore was already added in v6 — verify it's present
```

---

## File 9: `models/lightgbm/train_lgbm.py` *(MODIFY)*

**Purpose:** Add missing `atm_spend_zscore` and `salary_delay_zscore` to `FEATURE_COLS`.

```python
# In FEATURE_COLS list, add after the existing Z-score features:
"balance_zscore",           # already present in v6
"salary_delay_zscore",      # ADD — was missing
"atm_spend_zscore",         # ADD — was missing
"lending_spend_zscore",     # already present in v6
"emi_reliability_score",    # already present in v6
```

---

## Complete Run Order After Implementation

```powershell
# ── 1. Start infrastructure ───────────────────────────────────────────────────
docker compose up -d
# Wait until: docker compose ps  → all healthy

# ── 2. Apply all migrations ───────────────────────────────────────────────────
docker cp data_warehouse\schemas\migrate_001_pulse_history.sql sentinel-postgres:/tmp/m1.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m1.sql

docker cp data_warehouse\schemas\migrate_002_txn_redesign.sql sentinel-postgres:/tmp/m2.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m2.sql

docker cp data_warehouse\schemas\migrate_003_customer_baseline.sql sentinel-postgres:/tmp/m3.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m3.sql

docker cp data_warehouse\schemas\migrate_004_rich_customers.sql sentinel-postgres:/tmp/m4.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m4.sql

docker cp data_warehouse\schemas\migrate_005_rich_transactions.sql sentinel-postgres:/tmp/m5.sql
docker exec sentinel-postgres psql -U sentinel -d sentinel_db -f /tmp/m5.sql

# ── 3. Initialise Kafka and DynamoDB ─────────────────────────────────────────
python -m scripts.init_kafka_topics
python -m scripts.init_dynamodb

# ── 4. Generate training data and train model ─────────────────────────────────
python -m models.training_pipelines.build_training_data
python -m models.lightgbm.train_lgbm
# Verify feature names: head -2 models\lightgbm\lgbm_model.txt
# Should show: feature_names=salary_delay_days ... balance_zscore ... atm_spend_zscore

# ── 5. Start feature pipeline ─────────────────────────────────────────────────
# (Terminal 2 — keep running)
python -m ingestion.consumers.feature_pipeline

# ── 6. Start API ──────────────────────────────────────────────────────────────
# (Terminal 3 — keep running)
uvicorn api.main:app --reload --port 8001

# ── 7. PHASE 1 — Seed historical transactions (NO scoring) ───────────────────
# (Terminal 4)
python -m scripts.simulate_transactions --mode seed --customers 100 --days 90
# Runs ~5,000–10,000 transactions, then exits automatically
# Console shows: "✓ Seed complete. Now run: python -m scripts.build_baselines"

# ── 8. Build per-customer baselines ──────────────────────────────────────────
python -m scripts.build_baselines --days 90
# Expected: "Computed: 100  Skipped: 0"

# Verify baselines are personalised (every row different):
# docker exec sentinel-postgres psql -U sentinel -d sentinel_db -c
# "SELECT customer_id, ROUND(balance_mean) as avg_bal,
#   salary_day_mean, salary_delay_threshold, atm_spike_threshold
#  FROM customer_baseline ORDER BY customer_id LIMIT 10;"

# ── 9. PHASE 2 — Real-time scoring (model detects stress blindly) ─────────────
# (Terminal 4)
python -m scripts.simulate_transactions --mode realtime --customers 100 --delay 300

# ── 10. Start dashboard ───────────────────────────────────────────────────────
# (Terminal 5)
cd dashboards && npm run dev
# Open: http://localhost:3000
# Login: admin@sentinel.bank / sentinel_admin
```

---

## How to Verify the Model Is Detecting Stress Blindly

After running for ~10 minutes in realtime mode, run these checks:

**1. Confirm transactions have sender/receiver (not pre-labelled)**
```sql
SELECT sender_id, receiver_id, amount, balance_before, balance_after,
       balance_change_pct, platform, payment_status
FROM transactions
WHERE customer_id = 'CUST00001'
ORDER BY txn_timestamp DESC LIMIT 5;
```
Expected: `receiver_id` like `slice@upi` or `bescom@bbps` — NOT `txn_type='auto_debit'`

**2. Confirm model detected a lending-app transaction without being told**
```sql
SELECT t.receiver_id, t.amount, t.balance_change_pct,
       p.pulse_score, p.top_factor_1, p.scored_at
FROM transactions t
JOIN pulse_score_history p ON p.customer_id = t.customer_id
WHERE t.receiver_id LIKE '%@upi'
  AND t.amount > 5000
ORDER BY p.scored_at DESC LIMIT 5;
```
Expected: `top_factor_1` = `lending_spend_zscore` or `upi_lending_spike_ratio` — model detected it from the Z-score, not from a flag

**3. Confirm two customers have different score responses to same transaction amount**
```sql
SELECT c.customer_id,
       ROUND(b.balance_mean) as their_normal_balance,
       ROUND(b.balance_std) as their_balance_std,
       ROUND(b.atm_spike_threshold, 2) as their_atm_threshold
FROM customer_baseline b
JOIN customers c USING(customer_id)
ORDER BY b.balance_mean ASC
LIMIT 10;
```
Expected: every row has different values — personalised baselines confirmed

**4. Confirm scoring API returns Z-score features in SHAP top_factors**
```powershell
curl -X POST http://localhost:8001/api/v1/score `
  -H "Content-Type: application/json" `
  -d '{"customer_id": "CUST00001"}'
```
Expected: `top_factors[0].feature_name` = `balance_zscore` or `lending_spend_zscore`  
NOT: `Column_0` and NOT: `flag_lending` with a hardcoded threshold

---

## Summary of All Files Changed

| File | Action | Key Change |
|------|--------|-----------|
| `data_warehouse/schemas/migrate_004_rich_customers.sql` | **NEW** | Full customer profile: occupation, employer, live balance, UPI VPA, loan details |
| `data_warehouse/schemas/migrate_005_rich_transactions.sql` | **NEW** | sender_id, receiver_id, balance_before, balance_after, balance_change_pct |
| `ingestion/schemas/transaction_event.py` | **REPLACE** | Remove txn_type/merchant_category; add sender_id, receiver_id, balance fields |
| `ingestion/enrichment/transaction_classifier.py` | **MODIFY** | Classify from receiver VPA patterns, not stored labels |
| `scripts/simulate_transactions.py` | **MODIFY** | Add --mode seed/realtime; raw TransactionEvent; write all DB fields; no scoring in seed mode |
| `ingestion/consumers/feature_pipeline.py` | **MODIFY** | Call classifier per transaction to get inferred features |
| `sagemaker/inference.py` | **MODIFY** | Use _is_lending/_is_emi columns; personalised thresholds from baseline |
| `models/training_pipelines/build_training_data.py` | **MODIFY** | Add salary_delay_zscore and atm_spend_zscore simulation |
| `models/lightgbm/train_lgbm.py` | **MODIFY** | Add atm_spend_zscore, salary_delay_zscore to FEATURE_COLS |

---

*The model never sees a label. It sees a receiver VPA, a balance drop, a Z-score, and a platform. It decides.*
