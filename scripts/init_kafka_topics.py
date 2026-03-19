"""
scripts/init_kafka_topics.py
Creates all required Kafka topics with correct partition and replication config.
Run this ONCE after Kafka starts for the first time.
"""
import sys
import time
 
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
 
from config.settings import get_settings
 
settings = get_settings()
 
TOPICS = [
    NewTopic(
        name=settings.kafka_topic_transactions_raw,
        num_partitions=12,         # partition by customer_id hash
        replication_factor=1,      # 1 broker in dev; set to 3 in prod
        topic_configs={"retention.ms": str(7 * 24 * 60 * 60 * 1000)},  # 7 days
    ),
    NewTopic(
        name=settings.kafka_topic_transactions_enriched,
        num_partitions=12,
        replication_factor=1,
        topic_configs={"retention.ms": str(7 * 24 * 60 * 60 * 1000)},
    ),
    NewTopic(
        name=settings.kafka_topic_pulse_scores,
        num_partitions=6,
        replication_factor=1,
        topic_configs={"retention.ms": str(30 * 24 * 60 * 60 * 1000)},  # 30 days
    ),
    NewTopic(
        name="dlq-transactions",   # Dead Letter Queue for failed events
        num_partitions=3,
        replication_factor=1,
        topic_configs={"retention.ms": str(30 * 24 * 60 * 60 * 1000)},
    ),
]
 
 
def create_topics() -> None:
    print(f"Connecting to Kafka at {settings.kafka_bootstrap_servers}...")
    # Retry logic — Kafka may still be starting
    for attempt in range(10):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=settings.kafka_servers_list,
                client_id="sentinel-admin",
                request_timeout_ms=10000,
            )
            break
        except Exception as e:
            if attempt == 9:
                print(f"Failed to connect after 10 attempts: {e}")
                sys.exit(1)
            print(f"  Attempt {attempt + 1}/10 failed, retrying in 5s...")
            time.sleep(5)
 
    created = []
    skipped = []
    for topic in TOPICS:
        try:
            admin.create_topics([topic])
            created.append(topic.name)
            print(f"  ✓ Created: {topic.name} ({topic.num_partitions} partitions)")
        except TopicAlreadyExistsError:
            skipped.append(topic.name)
            print(f"  - Skipped (already exists): {topic.name}")
 
    admin.close()
    print(f"\nDone. Created: {len(created)}, Skipped: {len(skipped)}")
 
 
if __name__ == "__main__":
    create_topics()
