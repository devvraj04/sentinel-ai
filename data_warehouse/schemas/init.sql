-- ────────────────────────────────────────────────────────────────────────────
-- Sentinel PostgreSQL Schema
-- Runs automatically when PostgreSQL container first starts
-- ────────────────────────────────────────────────────────────────────────────
 
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
 
-- ── Users (dashboard login) ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    full_name   VARCHAR(255) NOT NULL,
    role        VARCHAR(50) NOT NULL CHECK (role IN ('credit_officer','risk_manager','collections_officer','admin')),
    password_hash VARCHAR(255) NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
 
-- ── Customer base ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(50) PRIMARY KEY,
    full_name       VARCHAR(255) NOT NULL,
    email           VARCHAR(255),
    phone           VARCHAR(20),
    segment         VARCHAR(50) DEFAULT 'mass_retail',
    geography       VARCHAR(100),
    employment_status VARCHAR(50),
    monthly_income  NUMERIC(15,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
 
-- ── Customer accounts (loans, cards, deposits) ────────────────────────────────
CREATE TABLE IF NOT EXISTS accounts (
    account_id      VARCHAR(50) PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL REFERENCES customers(customer_id),
    account_type    VARCHAR(30) NOT NULL CHECK (account_type IN ('loan','credit_card','savings','current')),
    product_name    VARCHAR(100),
    credit_limit    NUMERIC(15,2),
    outstanding_balance NUMERIC(15,2) DEFAULT 0,
    emi_amount      NUMERIC(15,2),
    emi_due_date    INTEGER,
    status          VARCHAR(30) DEFAULT 'active',
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
 
-- ── Raw transactions ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    txn_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id         VARCHAR(50) NOT NULL REFERENCES customers(customer_id),
    account_id          VARCHAR(50) REFERENCES accounts(account_id),
    txn_type            VARCHAR(50) NOT NULL,
    amount              NUMERIC(15,2) NOT NULL,
    merchant_category   VARCHAR(100),
    payment_channel     VARCHAR(50),
    payment_status      VARCHAR(30) DEFAULT 'success',
    txn_timestamp       TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
 
CREATE INDEX idx_txn_customer_time ON transactions(customer_id, txn_timestamp DESC);
CREATE INDEX idx_txn_type ON transactions(txn_type);
 
-- ── Behavioral features (computed by feature pipeline) ────────────────────────
CREATE TABLE IF NOT EXISTS customer_behavioral_features (
    id                      BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(50) NOT NULL REFERENCES customers(customer_id),
    computed_at             TIMESTAMPTZ NOT NULL,
    salary_delay_days       NUMERIC(5,2),
    balance_wow_drop_pct    NUMERIC(8,4),
    upi_lending_spike_ratio NUMERIC(8,4),
    utility_payment_latency NUMERIC(5,2),
    discretionary_contraction NUMERIC(8,4),
    atm_withdrawal_spike    NUMERIC(8,4),
    failed_auto_debit_count INTEGER,
    credit_utilization_delta NUMERIC(8,4),
    drift_score             NUMERIC(8,4),
    UNIQUE(customer_id, computed_at)
);
 
CREATE INDEX idx_features_customer_time ON customer_behavioral_features(customer_id, computed_at DESC);
 
-- ── Pulse Score history ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pulse_score_history (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL REFERENCES customers(customer_id),
    pulse_score     INTEGER NOT NULL CHECK (pulse_score BETWEEN 0 AND 100),
    risk_tier       VARCHAR(20) NOT NULL CHECK (risk_tier IN ('green','yellow','orange','red')),
    pd_probability  NUMERIC(8,6) NOT NULL,
    confidence      NUMERIC(5,4),
    top_factor_1    VARCHAR(100),
    top_factor_2    VARCHAR(100),
    top_factor_3    VARCHAR(100),
    shap_values     JSONB,
    model_version   VARCHAR(50),
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
 
CREATE INDEX idx_pulse_customer_time ON pulse_score_history(customer_id, scored_at DESC);
CREATE INDEX idx_pulse_risk_tier ON pulse_score_history(risk_tier, scored_at DESC);
 
-- ── Interventions log ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interventions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id         VARCHAR(50) NOT NULL REFERENCES customers(customer_id),
    intervention_type   VARCHAR(50) NOT NULL,
    channel             VARCHAR(30) NOT NULL,
    message_sent        TEXT,
    pulse_score_at_trigger INTEGER,
    risk_tier_at_trigger VARCHAR(20),
    top_factor          VARCHAR(100),
    customer_response   VARCHAR(20),
    outcome             VARCHAR(50),
    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    responded_at        TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ
);
 
CREATE INDEX idx_interventions_customer ON interventions(customer_id, triggered_at DESC);
 
-- ── Model registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(50) NOT NULL,
    model_type      VARCHAR(50) NOT NULL,
    artifact_path   VARCHAR(500),
    auc_roc         NUMERIC(6,4),
    precision_score NUMERIC(6,4),
    recall_score    NUMERIC(6,4),
    f1_score        NUMERIC(6,4),
    air_score       NUMERIC(6,4),
    psi_score       NUMERIC(6,4),
    is_production   BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    UNIQUE(model_name, version)
);
 
-- ── Insert default admin user ─────────────────────────────────────────────────
-- Password: sentinel_admin (bcrypt hash generated with passlib)
INSERT INTO users (email, full_name, role, password_hash) VALUES
(
    'admin@sentinel.bank',
    'System Admin',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LedYs/OWKTBnGgTmm'
) ON CONFLICT DO NOTHING;
