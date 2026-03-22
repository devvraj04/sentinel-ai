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
