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
