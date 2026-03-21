"""
ingestion/schemas/transaction_event.py
──────────────────────────────────────────────────────────────────────────────
Defines the canonical shape of every transaction event that flows through
the system. Every producer MUST create TransactionEvent objects.
Every consumer receives TransactionEvent objects.
This is the contract between ingestion and all downstream systems.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
 
from pydantic import BaseModel, Field, field_validator
 
 
class TransactionType(str, Enum):
    SALARY_CREDIT    = "salary_credit"
    UPI_DEBIT        = "upi_debit"
    UPI_CREDIT       = "upi_credit"
    ATM_WITHDRAWAL   = "atm_withdrawal"
    AUTO_DEBIT       = "auto_debit"          # EMI / recurring payment
    UTILITY_PAYMENT  = "utility_payment"
    CREDIT_CARD_PAYMENT = "credit_card_payment"
    NEFT_RTGS        = "neft_rtgs"
    POS_DEBIT        = "pos_debit"
    SAVINGS_WITHDRAWAL = "savings_withdrawal"
    LOAN_DISBURSEMENT  = "loan_disbursement"
    INTEREST_DEBIT     = "interest_debit"
    REVERSAL           = "reversal"
    OTHER              = "other"
    TRANSFER_OUT = "transfer_out"
 
 
class PaymentStatus(str, Enum):
    SUCCESS  = "success"
    FAILED   = "failed"
    PENDING  = "pending"
    REVERSED = "reversed"
 
 
class MerchantCategory(str, Enum):
    DINING        = "dining"
    ENTERTAINMENT = "entertainment"
    GROCERIES     = "groceries"
    UTILITIES     = "utilities"
    FUEL          = "fuel"
    HEALTHCARE    = "healthcare"
    EDUCATION     = "education"
    LENDING_APP   = "lending_app"    # UPI to lending apps — key stress signal
    GAMBLING      = "gambling"
    LOTTERY       = "lottery"
    TRAVEL        = "travel"
    SHOPPING      = "shopping"
    GROCERY       = "grocery" 
    OTHER         = "other"
 
 
class AccountType(str, Enum):
    LOAN        = "loan"
    CREDIT_CARD = "credit_card"
    SAVINGS     = "savings"
    CURRENT     = "current"
 
 
class TransactionEvent(BaseModel):
    """Single transaction event — the atomic unit of the entire system."""
 
    # Identity
    event_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id:    str = Field(..., min_length=1, max_length=50)
    account_id:     Optional[str] = None
    account_type:   AccountType
 
    # Transaction details
    txn_type:           TransactionType = TransactionType.OTHER  # Auto-classified by enrichment
    amount:             float = Field(..., gt=0)
    merchant_category:  MerchantCategory = MerchantCategory.OTHER
    payment_channel:    str = "unknown"  # Legacy — prefer `platform`
    payment_status:     PaymentStatus = PaymentStatus.SUCCESS

    # Indian banking context — counterparty & platform
    counterparty_id:    Optional[str] = None   # UPI VPA (e.g., swiggy@upi) or bank account
    counterparty_name:  Optional[str] = None   # Human-readable name (e.g., Swiggy, BESCOM)
    reference_number:   Optional[str] = None   # UTR/RRN/IMPS reference
    platform:           str = "unknown"        # UPI/NEFT/IMPS/RTGS/NACH/BBPS/POS/ATM/NetBanking

    # Balance tracking (never negative for savings/current accounts)
    balance_before:     Optional[float] = None  # Account balance before this transaction
    balance_after:      Optional[float] = None  # Account balance after this transaction
 
    # Timing
    txn_timestamp:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
 
    # Derived flags (set by enrichment classifier, not the producer)
    is_lending_app_upi:  bool = False   # True when UPI goes to a lending/NBFC app
    is_auto_debit_failed: bool = False  # True when auto-debit bounced
    is_p2p_transfer:     bool = False   # True when peer-to-peer UPI transfer
    is_investment_txn:   bool = False   # True for MF/FD/investment transactions
 
    # Kafka metadata (filled by consumer, not producer)
    kafka_partition:    Optional[int] = None
    kafka_offset:       Optional[int] = None
    ingested_at:        Optional[datetime] = None
 
    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)
 
    def to_dict(self) -> dict:
        """Serialize to dict for Kafka message value."""
        data = self.model_dump()
        # Convert datetime to ISO string for JSON serialization
        data["txn_timestamp"] = self.txn_timestamp.isoformat()
        if self.ingested_at:
            data["ingested_at"] = self.ingested_at.isoformat()
        return data
 
    @classmethod
    def from_dict(cls, data: dict) -> "TransactionEvent":
        """Deserialize from Kafka message."""
        return cls(**data)

