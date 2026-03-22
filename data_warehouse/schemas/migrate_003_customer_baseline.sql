-- ── Customer statistical baseline ──────────────────────────────────────────────
-- Computed from first 90 days of transaction history per customer.
-- Updated every 30 days via scripts/build_baselines.py.
-- All feature computations in build_feature_vector() use this table
-- instead of hardcoded thresholds.

CREATE TABLE IF NOT EXISTS customer_baseline (
    customer_id             VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id),
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    observation_days        INTEGER NOT NULL DEFAULT 0,  -- how many days of history used

    -- ── Balance statistics ────────────────────────────────────────────────────
    balance_mean            NUMERIC(15,2) DEFAULT 0,   -- rolling 90-day mean balance
    balance_std             NUMERIC(15,2) DEFAULT 1,   -- std deviation of balance
    balance_min_90d         NUMERIC(15,2) DEFAULT 0,   -- lowest balance in 90 days
    balance_max_90d         NUMERIC(15,2) DEFAULT 0,   -- peak balance in 90 days
    balance_trend_slope     NUMERIC(8,4)  DEFAULT 0,   -- positive=growing, negative=eroding

    -- ── Salary statistics ─────────────────────────────────────────────────────
    salary_day_mean         NUMERIC(5,2) DEFAULT 3,    -- typical day-of-month salary arrives
    salary_day_std          NUMERIC(5,2) DEFAULT 1,    -- std deviation of salary day
    salary_amount_mean      NUMERIC(15,2) DEFAULT 0,
    salary_amount_std       NUMERIC(15,2) DEFAULT 1,
    salary_months_observed  INTEGER DEFAULT 0,

    -- ── EMI / auto-debit statistics ───────────────────────────────────────────
    emi_amount_mean         NUMERIC(15,2) DEFAULT 0,
    emi_day_mean            NUMERIC(5,2) DEFAULT 5,
    emi_success_rate        NUMERIC(5,4) DEFAULT 1.0,  -- 1.0 = never bounced
    emi_attempts_90d        INTEGER DEFAULT 0,
    emi_failures_90d        INTEGER DEFAULT 0,

    -- ── Spending behaviour statistics ────────────────────────────────────────
    monthly_spend_mean      NUMERIC(15,2) DEFAULT 0,
    monthly_spend_std       NUMERIC(15,2) DEFAULT 1,
    discretionary_mean      NUMERIC(15,2) DEFAULT 0,   -- dining/entertainment/travel
    discretionary_std       NUMERIC(15,2) DEFAULT 1,
    atm_monthly_mean        NUMERIC(15,2) DEFAULT 0,
    atm_monthly_std         NUMERIC(15,2) DEFAULT 1,
    utility_day_mean        NUMERIC(5,2) DEFAULT 10,   -- typical day utility bills paid
    utility_day_std         NUMERIC(5,2) DEFAULT 3,

    -- ── Credit behaviour statistics ───────────────────────────────────────────
    cc_utilization_mean     NUMERIC(5,4) DEFAULT 0.3,
    cc_utilization_std      NUMERIC(5,4) DEFAULT 0.1,
    upi_to_lending_mean     NUMERIC(15,2) DEFAULT 0,   -- monthly UPI to lending apps (₹)
    upi_to_lending_std      NUMERIC(15,2) DEFAULT 1,

    -- ── P2P / informal lending ────────────────────────────────────────────────
    p2p_outflow_mean        NUMERIC(15,2) DEFAULT 0,
    p2p_outflow_std         NUMERIC(15,2) DEFAULT 1,

    -- ── Anomaly sensitivity thresholds (per customer, not population) ─────────
    -- Computed as: threshold = mean + (z_multiplier * std)
    -- z_multiplier = 2.0 for most signals (flags at 2 std deviations from normal)
    balance_drop_threshold  NUMERIC(8,4) DEFAULT 0.30, -- fraction drop that is alarming
    salary_delay_threshold  NUMERIC(5,2) DEFAULT 7,    -- days late that is alarming
    atm_spike_threshold     NUMERIC(5,2) DEFAULT 3.0,  -- ratio vs baseline that is alarming
    lending_spike_threshold NUMERIC(5,2) DEFAULT 2.5,  -- ratio vs baseline

    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_baseline_customer
    ON customer_baseline(customer_id);

CREATE INDEX IF NOT EXISTS idx_baseline_computed
    ON customer_baseline(computed_at DESC);

-- ── Baseline computation log (audit trail) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS baseline_computation_log (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    txn_count       INTEGER,
    observation_days INTEGER,
    notes           TEXT
);
