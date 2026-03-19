"""
ingestion/producers/transaction_producer.py
──────────────────────────────────────────────────────────────────────────────
Publishes TransactionEvent objects to Kafka.
Key design: partition by customer_id so all events for one customer
go to the same partition — preserving order for behavioral analysis.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import json
import time
from typing import Optional
 
from kafka import KafkaProducer
from kafka.errors import KafkaError
 
from config.logging_config import get_logger
from config.settings import get_settings
from ingestion.schemas.transaction_event import TransactionEvent
 
logger = get_logger(__name__)
settings = get_settings()
 
 
class TransactionProducer:
    """Thread-safe Kafka producer for transaction events."""
 
    def __init__(self) -> None:
        self._producer: Optional[KafkaProducer] = None
 
    def _get_producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=settings.kafka_servers_list,
                # Serialize value to JSON bytes
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Serialize key to bytes (for partition routing by customer_id)
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                # Reliability: wait for all in-sync replicas to ack
                acks=1,
                # Retry on transient errors
                retries=5,
                retry_backoff_ms=500,
                # Compression for throughput
                compression_type="gzip",
                # Batching
                linger_ms=10,
                batch_size=65536,

                request_timeout_ms=30000,
            )
            logger.info("Kafka producer initialized", servers=settings.kafka_bootstrap_servers)
        return self._producer
 
    def publish(self, event: TransactionEvent, topic: Optional[str] = None) -> bool:
        """
        Publish a single TransactionEvent.
        Returns True on success, False on failure.
        Key = customer_id ensures all events for a customer land on same partition.
        """
        target_topic = topic or settings.kafka_topic_transactions_raw
        try:
            future = self._get_producer().send(
                topic=target_topic,
                key=event.customer_id,
                value=event.to_dict(),
            )
            # Block until ack (with 10s timeout) for reliability
            record_metadata = future.get(timeout=10)
            logger.debug(
                "Event published",
                customer_id=event.customer_id,
                topic=record_metadata.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
            )
            return True
        except KafkaError as exc:
            logger.error(
                "Failed to publish event",
                customer_id=event.customer_id,
                error=str(exc),
            )
            # Publish to Dead Letter Queue so no event is lost
            self._send_to_dlq(event, str(exc))
            return False
 
    def publish_batch(self, events: list[TransactionEvent]) -> dict[str, int]:
        """Publish a batch of events. Returns counts of success/failure."""
        results = {"success": 0, "failed": 0}
        for event in events:
            if self.publish(event):
                results["success"] += 1
            else:
                results["failed"] += 1
        self._get_producer().flush()
        return results
 
    def _send_to_dlq(self, event: TransactionEvent, error_message: str) -> None:
        """Send failed events to Dead Letter Queue for later inspection."""
        try:
            dlq_payload = {**event.to_dict(), "_dlq_error": error_message, "_dlq_at": time.time()}
            self._get_producer().send(
                topic="dlq-transactions",
                key=event.customer_id,
                value=dlq_payload,
            )
        except Exception as exc:
            logger.critical("DLQ send also failed", error=str(exc))
 
    def close(self) -> None:
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Kafka producer closed")
 
    def __enter__(self) -> "TransactionProducer":
        return self
 
    def __exit__(self, *args) -> None:
        self.close()
