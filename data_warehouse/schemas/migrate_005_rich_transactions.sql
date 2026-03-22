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
