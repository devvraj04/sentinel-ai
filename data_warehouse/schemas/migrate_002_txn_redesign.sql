-- ─────────────────────────────────────────────────────────────────────────────
-- Migration 002 — Transaction System Redesign (Indian Context)
--
-- Adds customer profile fields (occupation, loans, balances) and
-- transaction-level balance tracking + counterparty info.
--
-- Safe to run multiple times (all statements use IF NOT EXISTS).
--
-- How to apply:
--   docker exec -i sentinel-postgres psql -U sentinel -d sentinel_db < \
--       data_warehouse/schemas/migrate_002_txn_redesign.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Customer profile extensions ──────────────────────────────────────────────
ALTER TABLE customers
    ADD COLUMN IF NOT EXISTS occupation           VARCHAR(100),
    ADD COLUMN IF NOT EXISTS employer_name        VARCHAR(200),
    ADD COLUMN IF NOT EXISTS pan_masked           VARCHAR(14),
    ADD COLUMN IF NOT EXISTS expected_salary_day  INTEGER,
    ADD COLUMN IF NOT EXISTS emi_amount           NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS emi_due_day          INTEGER,
    ADD COLUMN IF NOT EXISTS credit_limit         NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_loan_amount    NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS loan_type            VARCHAR(50),
    ADD COLUMN IF NOT EXISTS savings_balance      NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS current_account_balance NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_savings_balance  NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tenure_months        INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS preferred_channel    VARCHAR(50) DEFAULT 'UPI',
    ADD COLUMN IF NOT EXISTS product_mix          VARCHAR(50) DEFAULT 'both';

-- ── Transaction table extensions ─────────────────────────────────────────────
ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS balance_before     NUMERIC(15,2),
    ADD COLUMN IF NOT EXISTS balance_after      NUMERIC(15,2),
    ADD COLUMN IF NOT EXISTS counterparty_id    VARCHAR(100),
    ADD COLUMN IF NOT EXISTS counterparty_name  VARCHAR(200),
    ADD COLUMN IF NOT EXISTS reference_number   VARCHAR(50),
    ADD COLUMN IF NOT EXISTS platform           VARCHAR(30) DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS account_type       VARCHAR(30);

-- ── Accounts table extensions ────────────────────────────────────────────────
ALTER TABLE accounts
    ADD COLUMN IF NOT EXISTS balance_current    NUMERIC(15,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS interest_rate      NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS tenure_months      INTEGER,
    ADD COLUMN IF NOT EXISTS disbursement_date  DATE,
    ADD COLUMN IF NOT EXISTS maturity_date      DATE;

-- ── Balance constraint: balance_after >= 0 for savings/current (enforced in app layer)
-- PostgreSQL CHECK constraint on existing table requires careful handling
-- We add a partial index for quick lookups of negative balances (monitoring)
CREATE INDEX IF NOT EXISTS idx_txn_negative_balance
    ON transactions(customer_id, balance_after)
    WHERE balance_after < 0;

-- ── Index for counterparty lookups ───────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_txn_counterparty
    ON transactions(counterparty_id)
    WHERE counterparty_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_txn_platform
    ON transactions(platform);

-- ── Verify ───────────────────────────────────────────────────────────────────
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'customers'
ORDER BY ordinal_position;

SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'transactions'
ORDER BY ordinal_position;
