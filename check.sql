SELECT COUNT(*) AS history_rows FROM pulse_score_history;

SELECT customer_id, pulse_score, risk_tier, scored_at
FROM pulse_score_history
ORDER BY scored_at DESC
LIMIT 5;