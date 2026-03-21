"""
ingestion/consumers/feature_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Kafka consumer that:
1. Reads enriched transaction events from 'transactions-enriched' topic
2. Accumulates a 90-day window of transactions per customer in memory
3. Recomputes all behavioral signals on EVERY transaction (real-time)
4. Writes signals to Feast Online Store (Redis) for real-time scoring
5. Also writes to PostgreSQL for offline training data
 
Run this continuously — it IS the real-time feature computation engine.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import json
import signal
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
 
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import KafkaError
 
from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from features.transformations.behavioral_signals import BehavioralSignalEngine
from ingestion.schemas.transaction_event import TransactionEvent
 
setup_logging()
logger = get_logger(__name__)
settings = get_settings()

# Max transactions to keep in memory per customer (90 days approx)
MAX_TXN_HISTORY = 500
 
 
class FeaturePipeline:
    """Real-time feature computation pipeline — scores EVERY transaction."""
 
    def __init__(self) -> None:
        self.engine = BehavioralSignalEngine()
        # In-memory buffer: customer_id -> list of transaction dicts
        self._buffers: dict[str, list[dict]] = defaultdict(list)
        self._event_counts: dict[str, int] = defaultdict(int)
        self._running = True
 
        self._warm_start()

        # Graceful shutdown on Ctrl+C or SIGTERM
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _warm_start(self) -> None:
        """Pre-load recent transactions from PostgreSQL to avoid empty baseline on restart."""
        try:
            import psycopg2
            import psycopg2.extras
            logger.info("Warm-starting buffers from PostgreSQL...")
            conn = psycopg2.connect(settings.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT customer_id, amount, txn_type, merchant_category, 
                           payment_status, account_type, txn_timestamp
                    FROM transactions 
                    WHERE txn_timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY txn_timestamp ASC
                """)
                for row in cur.fetchall():
                    cid = row["customer_id"]
                    # Format correctly as strings/bools as expected by the signal engine
                    event = {
                        "customer_id": cid,
                        "amount": float(row["amount"]),
                        "txn_type": row["txn_type"],
                        "merchant_category": row["merchant_category"] or "unknown",
                        "payment_status": row["payment_status"] or "completed",
                        "account_type": row["account_type"] or "savings",
                        "txn_timestamp": row["txn_timestamp"].isoformat(),
                        "is_lending_app_upi": row["merchant_category"] == "lending_app",
                        "is_auto_debit_failed": row["txn_type"] == "auto_debit" and row["payment_status"] == "failed"
                    }
                    self._buffer_event(event)

            conn.close()
            logger.info("Warm-start complete", total_events=sum(len(b) for b in self._buffers.values()))
        except Exception as e:
            logger.warning("Warm-start failed, starting with empty buffers", error=str(e))

 
    def _shutdown(self, *args) -> None:
        logger.info("Shutting down feature pipeline...")
        self._running = False
 
    def _get_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            settings.kafka_topic_transactions_raw,
            bootstrap_servers=settings.kafka_servers_list,
            group_id=f"{settings.kafka_consumer_group_id}-features",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )
 
    def _buffer_event(self, event_dict: dict) -> None:
        cid = event_dict.get("customer_id")
        if not cid:
            return
        buf = self._buffers[cid]
        buf.append(event_dict)
        # Keep only last MAX_TXN_HISTORY events (slide the window)
        if len(buf) > MAX_TXN_HISTORY:
            self._buffers[cid] = buf[-MAX_TXN_HISTORY:]
        self._event_counts[cid] += 1
 
    def _compute_and_store(self, customer_id: str) -> None:
        """Compute behavioral signals and write to Feast online store."""
        buf = self._buffers.get(customer_id, [])
        if not buf:
            return
        try:
            df = pd.DataFrame(buf)
            df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
            signals = self.engine.compute(customer_id, df)
 
            # Write to Feast online store (Redis)
            self._write_to_feast(customer_id, signals)

            # Trigger full ML re-score → writes new Pulse Score to DynamoDB
            self._trigger_pulse_score(customer_id)

            logger.debug(
                "Signals computed",
                customer_id=customer_id,
                drift_score=round(signals.drift_score, 3),
                has_stress=signals.has_early_stress(),
            )
        except Exception as exc:
            logger.error("Signal computation failed", customer_id=customer_id, error=str(exc))
 
    def _write_to_feast(self, customer_id: str, signals) -> None:
        """Write computed signals to Feast online store via direct Redis write."""
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url)
        key = f"sentinel:features:{customer_id}"
        data = {k: str(v) for k, v in signals.to_feature_vector().items()}
        data["computed_at"] = signals.computed_at.isoformat()
        r.hset(key, mapping=data)
        r.expire(key, 86400)  # 24-hour TTL

    def _trigger_pulse_score(self, customer_id: str) -> None:
        """
        After features are refreshed in Redis, call PulseScorer to:
          1. Read the updated features from Redis
          2. Run the LightGBM model
          3. Write the new Pulse Score + risk tier to DynamoDB
        This is what makes real-time transactions affect DynamoDB scores
        via the full ML model path.
        """
        try:
            from serving.bentoml_service.pulse_scorer import get_scorer
            scorer = get_scorer()
            scorer.score(customer_id, force_refresh=True)
            logger.debug("Pulse score updated via ML model", customer_id=customer_id)
        except Exception as exc:
            logger.warning(
                "Pulse score update failed (non-fatal)",
                customer_id=customer_id,
                error=str(exc),
            )
 
    def run(self) -> None:
        logger.info("Feature pipeline starting (real-time mode: scoring every transaction)...")
        consumer = self._get_consumer()
        processed = 0
 
        try:
            while self._running:
                records = consumer.poll(timeout_ms=1000)
                for tp, messages in records.items():
                    for msg in messages:
                        try:
                            self._buffer_event(msg.value)
                            cid = msg.value.get("customer_id", "")
                            # Score EVERY transaction — no throttling
                            if cid:
                                self._compute_and_store(cid)
                            processed += 1
                            if processed % 1000 == 0:
                                logger.info("Feature pipeline progress", processed=processed)
                        except Exception as exc:
                            logger.error("Event processing error", error=str(exc))
        finally:
            consumer.close()
            logger.info("Feature pipeline stopped", total_processed=processed)
 
 
if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()