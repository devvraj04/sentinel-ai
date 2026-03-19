"""
scripts/init_dynamodb.py
──────────────────────────────────────────────────────────────────────────────
Creates all DynamoDB tables on real AWS.
Run ONCE before starting the simulator.

Tables:
  sentinel-customer-scores    — live Pulse Score per customer (dashboard reads this)
  sentinel-interventions      — intervention history per customer
  sentinel-audit-log          — immutable system decision log (regulatory)
  sentinel-transactions       — real-time transaction feed (NEW)
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def load_env(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass

load_env()

REGION     = os.environ.get("AWS_REGION", "ap-south-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def get_client():
    return boto3.client(
        "dynamodb",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(connect_timeout=10, read_timeout=10,
                      retries={"max_attempts": 2}),
    )


TABLES = [
    # ── 1. Customer live scores (dashboard) ──────────────────────────────────
    {
        "TableName": "sentinel-customer-scores",
        "KeySchema": [
            {"AttributeName": "customer_id", "KeyType": "HASH"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "customer_id", "AttributeType": "S"},
            {"AttributeName": "risk_tier",   "AttributeType": "S"},
            {"AttributeName": "pulse_score", "AttributeType": "N"},
        ],
        "GlobalSecondaryIndexes": [
            {
                # Query: "give me all RED tier customers" — for dashboard list
                "IndexName": "risk-tier-index",
                "KeySchema": [
                    {"AttributeName": "risk_tier",   "KeyType": "HASH"},
                    {"AttributeName": "pulse_score", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },

    # ── 2. Intervention history ───────────────────────────────────────────────
    {
        "TableName": "sentinel-interventions",
        "KeySchema": [
            {"AttributeName": "customer_id",  "KeyType": "HASH"},
            {"AttributeName": "triggered_at", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "customer_id",  "AttributeType": "S"},
            {"AttributeName": "triggered_at", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },

    # ── 3. Audit log (regulatory, immutable) ─────────────────────────────────
    {
        "TableName": "sentinel-audit-log",
        "KeySchema": [
            {"AttributeName": "record_id",  "KeyType": "HASH"},
            {"AttributeName": "created_at", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "record_id",  "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },

    # ── 4. Real-time transactions (NEW) ───────────────────────────────────────
    {
        "TableName": "sentinel-transactions",
        "KeySchema": [
            {"AttributeName": "customer_id",    "KeyType": "HASH"},   # partition
            {"AttributeName": "txn_timestamp",  "KeyType": "RANGE"},  # sort
        ],
        "AttributeDefinitions": [
            {"AttributeName": "customer_id",   "AttributeType": "S"},
            {"AttributeName": "txn_timestamp", "AttributeType": "S"},
            {"AttributeName": "txn_type",      "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                # Query: "give me all failed auto-debits today"
                "IndexName": "txn-type-index",
                "KeySchema": [
                    {"AttributeName": "txn_type",     "KeyType": "HASH"},
                    {"AttributeName": "txn_timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },
]

TTL_CONFIG = {
    # Transactions auto-expire after 90 days (keeps table lean)
    "sentinel-transactions": "ttl",
    # Audit log never expires (regulatory requirement)
}


def create_tables():
    client = get_client()

    print("Connecting to AWS DynamoDB...")
    try:
        existing = client.list_tables()["TableNames"]
        print(f"  ✓ Connected — existing tables: {existing}\n")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)

    for table_def in TABLES:
        name = table_def["TableName"]

        if name in existing:
            print(f"  - Skipped (already exists): {name}")
            continue

        try:
            client.create_table(**table_def)
            print(f"  Creating {name}...", end="", flush=True)

            # Wait for table to be active
            waiter = client.get_waiter("table_exists")
            waiter.wait(
                TableName=name,
                WaiterConfig={"Delay": 2, "MaxAttempts": 30},
            )
            print(" ✓ Active")

            # Enable TTL if configured
            if name in TTL_CONFIG:
                try:
                    client.update_time_to_live(
                        TableName=name,
                        TimeToLiveSpecification={
                            "Enabled": True,
                            "AttributeName": TTL_CONFIG[name],
                        },
                    )
                    print(f"    ✓ TTL enabled ({TTL_CONFIG[name]})")
                except Exception as e:
                    print(f"    ⚠ TTL setup failed: {e}")

        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ResourceInUseException":
                print(f"  - Skipped (already exists): {name}")
            else:
                print(f"  ✗ Error creating {name}: {e}")
                raise

    print("\nAll tables ready:")
    final = client.list_tables()["TableNames"]
    for t in sorted(final):
        if t.startswith("sentinel"):
            desc = client.describe_table(TableName=t)["Table"]
            print(f"  ✓ {t:40s} {desc['TableStatus']}")


if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL — DynamoDB Table Setup (AWS)")
    print(f"Region: {REGION}")
    print("=" * 60 + "\n")
    create_tables()
    print("\nDone. Update your .env:")
    print("  DYNAMODB_TABLE_TRANSACTIONS=sentinel-transactions")