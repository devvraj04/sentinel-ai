-- ─────────────────────────────────────────────────────────────────────────────
-- Migration 001 — pulse_score_history: add intervention columns
--
-- Run this ONCE on any existing database that was created before this change.
-- Safe to run multiple times (all statements use IF NOT EXISTS / IF EXISTS).
--
-- How to apply:
--   docker exec -i sentinel-postgres psql -U sentinel -d sentinel_db < \
--       data_warehouse/schemas/migrate_001_pulse_history.sql
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE pulse_score_history
    ADD COLUMN IF NOT EXISTS intervention_flag  BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS intervention_type  VARCHAR(100) DEFAULT 'none';

-- Extra index for fast "latest score per customer" dashboard queries
CREATE INDEX IF NOT EXISTS idx_pulse_latest
    ON pulse_score_history(customer_id, scored_at DESC);

-- ── Verify ────────────────────────────────────────────────────────────────────
SELECT
    column_name,
    data_type,
    column_default
FROM information_schema.columns
WHERE table_name = 'pulse_score_history'
ORDER BY ordinal_position;