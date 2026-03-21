# Sentinel AI — Pre-Delinquency Intervention Engine

> **Real-time behavioural risk scoring for retail banking.**  
> Every transaction generates a Pulse Score. Every score has a SHAP explanation. Every intervention is triggered by the model — not a rule.

---

## Table of Contents

1. [What Sentinel Does](#1-what-sentinel-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [Repository Structure](#4-repository-structure)
5. [Step 0 — Clone & Configure](#5-step-0--clone--configure)
6. [Step 1 — Start All Infrastructure](#6-step-1--start-all-infrastructure)
7. [Step 2 — Initialise Kafka Topics](#7-step-2--initialise-kafka-topics)
8. [Step 3 — Initialise DynamoDB Tables](#8-step-3--initialise-dynamodb-tables)
9. [Step 4 — Start the Feature Pipeline](#9-step-4--start-the-feature-pipeline)
10. [Step 5 — Start the FastAPI Backend](#10-step-5--start-the-fastapi-backend)
11. [Step 6 — Run Historical (Seeded) Simulation](#11-step-6--run-historical-seeded-simulation)
12. [Step 7 — Run Real-Time Simulation](#12-step-7--run-real-time-simulation)
13. [Step 8 — Start the Dashboard](#13-step-8--start-the-dashboard)
14. [Verification Checklist](#14-verification-checklist)
    - [Verify Kafka Topics & Messages](#141-verify-kafka-topics--messages)
    - [Verify DynamoDB Scores](#142-verify-dynamodb-scores)
    - [Verify Feature Extraction](#143-verify-feature-extraction)
    - [Verify Per-Transaction Score Impact](#144-verify-per-transaction-score-impact)
    - [Verify Model-Detected Stress (Not Manual Flags)](#145-verify-model-detected-stress-not-manual-flags)
15. [Real-Time Stress Tracker](#15-real-time-stress-tracker)
16. [API Reference](#16-api-reference)
17. [Known Issues & Remaining Bottlenecks](#17-known-issues--remaining-bottlenecks)
18. [Environment Variables Reference](#18-environment-variables-reference)

---

## 1. What Sentinel Does

Sentinel monitors retail banking customers for early signs of financial stress and intervenes **before** the customer misses a payment.

| Signal | What it detects |
|--------|----------------|
| Salary delay | Employer paying later than usual |
| Lending app UPI | Customer borrowing from informal lenders |
| Failed auto-debits | EMI bounce — acute cash stress |
| ATM withdrawal spike | Cash hoarding behaviour |
| Balance erosion | Savings declining week-over-week |
| Utility latency | Bills being pushed to end of month |
| Discretionary contraction | Customer cutting non-essential spend |
| Credit utilisation delta | Maxing credit cards |

Each transaction is scored by a **LightGBM model** in real time. The model outputs a **Pulse Score (0–100)** — higher means higher probability of default. SHAP values tell you exactly which behavioural features drove the score.

**Scoring semantics:**
```
Score 70–100  → 🔴 Red    — Critical. Trigger payment holiday or restructure.
Score 45–69   → 🟠 Orange — At risk.  Offer flexible EMI.
Score 25–44   → 🟡 Yellow — Watch.    Send preventive digital nudge.
Score 0–24    → 🟢 Green  — Safe.     No action required.
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENTINEL PIPELINE                            │
│                                                                     │
│  simulate_transactions.py                                           │
│    │                                                                │
│    ├── Generates transactions (historical seed + real-time)         │
│    ├── Publishes to Kafka (transactions-raw topic)                  │
│    └── Writes to PostgreSQL (transactions table)                    │
│                                                                     │
│  feature_pipeline.py  [consumer]                                    │
│    │                                                                │
│    ├── Reads from Kafka (transactions-raw)                          │
│    ├── Accumulates 90-day transaction window per customer           │
│    ├── Computes 8 behavioural signals (BehavioralSignalEngine)      │
│    ├── Writes features to Redis (sentinel:features:{cid})           │
│    └── Triggers PulseScorer → LightGBM inference                   │
│                                                                     │
│  PulseScorer (serving/bentoml_service/pulse_scorer.py)              │
│    │                                                                │
│    ├── Reads features from Redis                                    │
│    ├── Runs LightGBM → PD probability                               │
│    ├── Maps PD → Pulse Score (sigmoid, centre = model threshold)    │
│    ├── Computes SHAP top-7 factor contributions                     │
│    ├── Writes current score to DynamoDB                             │
│    └── Appends to PostgreSQL pulse_score_history                    │
│                                                                     │
│  FastAPI  (api/)                                                    │
│    └── Exposes scores, customer data, portfolio metrics             │
│                                                                     │
│  Next.js Dashboard  (dashboards/)                                   │
│    └── Risk Manager + Credit Officer views                          │
└─────────────────────────────────────────────────────────────────────┘

Infrastructure: Docker Compose
  kafka · zookeeper · schema-registry · kafka-ui
  postgres · redis · dynamodb-local · mlflow
```

---

## 3. Prerequisites

Install everything below before proceeding. **All versions are tested.**

| Tool | Minimum Version | Check |
|------|----------------|-------|
| Docker Desktop | 24.x | `docker --version` |
| Docker Compose | 2.x (bundled) | `docker compose version` |
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | Any | `git --version` |

> **macOS / Linux only.** Windows users should run inside WSL 2.

**Memory:** Docker needs at least **4 GB RAM** allocated (Kafka + Postgres + DynamoDB + Redis).  
Go to Docker Desktop → Settings → Resources → Memory → set to 4 GB minimum.

---

## 4. Repository Structure

```
sentinel-ai/
├── api/                        FastAPI backend
│   └── routers/                auth, customers, portfolio, scoring
├── config/                     Settings (all from .env)
├── dashboards/                 Next.js 16 dashboard (React 19)
├── data/                       Parquet training features
├── data_warehouse/schemas/     PostgreSQL init.sql (auto-runs on first start)
├── docker-compose.yaml         All infrastructure services
├── features/transformations/   BehavioralSignalEngine
├── ingestion/
│   ├── consumers/              feature_pipeline.py — Kafka → Redis → Score
│   ├── enrichment/             Transaction classifier (lending, salary, utility)
│   ├── producers/              TransactionProducer (Kafka publisher)
│   └── schemas/                TransactionEvent Pydantic model
├── intervention/               Bedrock message generator + SNS trigger
├── models/
│   ├── lightgbm/               lgbm_model.joblib + shap_explainer.joblib
│   └── training_pipelines/     build_training_data.py
├── requirements.txt
├── sagemaker/                  SageMaker train.py + inference.py
└── scripts/
    ├── init_dynamodb.py        Create DynamoDB tables (run once)
    ├── init_kafka_topics.py    Create Kafka topics (run once)
    ├── realtime_stress_tracker.py  Live stress signal feed
    └── simulate_transactions.py    Transaction generator (seed + real-time)
```

---

## 5. Step 0 — Clone & Configure

### 5.1 Clone the repository

```bash
git clone https://github.com/devvraj04/sentinel-ai.git
cd sentinel-ai
```

### 5.2 Create your `.env` file

```bash
cp .env.example .env   # if provided, or create manually
```

Create `.env` in the project root with the following content. For local development, use these exact values — they match the Docker Compose service names:

```env
# ── Application ───────────────────────────────────────────────────────────────
ENVIRONMENT=development
SECRET_KEY=change-this-to-a-random-64-char-string-before-any-deploy
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
LOG_LEVEL=INFO

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://sentinel:sentinel2024@localhost:5432/sentinel_db

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_SCHEMA_REGISTRY_URL=http://localhost:8081
KAFKA_TOPIC_TRANSACTIONS_RAW=transactions-raw
KAFKA_TOPIC_TRANSACTIONS_ENRICHED=transactions-enriched
KAFKA_TOPIC_PULSE_SCORES=pulse-scores
KAFKA_CONSUMER_GROUP_ID=sentinel-group

# ── AWS (local DynamoDB — no real AWS account needed for local dev) ───────────
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
DYNAMODB_ENDPOINT=http://localhost:8000
DYNAMODB_TABLE_SCORES=sentinel-customer-scores
DYNAMODB_TABLE_INTERVENTIONS=sentinel-interventions
DYNAMODB_TABLE_AUDIT=sentinel-audit-log
DYNAMODB_TABLE_TRANSACTIONS=sentinel-transactions

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_EXPERIMENT_NAME=sentinel-risk-models
```

> **Important:** `SECRET_KEY` must be a long random string. Generate one with:
> ```bash
> python -c "import secrets; print(secrets.token_hex(32))"
> ```

### 5.3 Install Python dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows WSL

# Install all dependencies
pip install -r requirements.txt
```

> **Troubleshooting:** If `lightgbm` fails on Apple Silicon, try:
> ```bash
> pip install lightgbm --no-binary lightgbm
> ```

---

## 6. Step 1 — Start All Infrastructure

This single command starts every service: Kafka, Zookeeper, Schema Registry, Kafka UI, PostgreSQL, Redis, DynamoDB Local, and MLflow.

```bash
docker compose up -d
```

**Wait for all services to be healthy.** Check status with:

```bash
docker compose ps
```

Expected output — all services should show `healthy` or `running`:

```
NAME                      STATUS          PORTS
sentinel-zookeeper        running (healthy)   0.0.0.0:2181->2181/tcp
sentinel-kafka            running (healthy)   0.0.0.0:9092->9092/tcp, 0.0.0.0:29092->29092/tcp
sentinel-schema-registry  running             0.0.0.0:8081->8081/tcp
sentinel-kafka-ui         running             0.0.0.0:8080->8080/tcp
sentinel-postgres         running (healthy)   0.0.0.0:5432->5432/tcp
sentinel-redis            running (healthy)   0.0.0.0:6379->6379/tcp
sentinel-dynamodb         running             0.0.0.0:8000->8000/tcp
sentinel-mlflow           running             0.0.0.0:5001->5001/tcp
```

> **Kafka takes 20–30 seconds** to reach healthy state after Zookeeper. If you see `starting`, wait and re-run `docker compose ps`.

**Service URLs for reference:**

| Service | URL |
|---------|-----|
| Kafka UI | http://localhost:8080 |
| Kafka (external) | localhost:29092 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |
| DynamoDB Local | http://localhost:8000 |
| MLflow | http://localhost:5001 |

**Verify PostgreSQL schema was applied:**

The `data_warehouse/schemas/init.sql` file auto-runs when the PostgreSQL container first starts. Verify all tables exist:

```bash
docker exec -it sentinel-postgres psql -U sentinel -d sentinel_db -c "\dt"
```

Expected output:

```
              List of relations
 Schema |           Name                | Type  |  Owner
--------+-------------------------------+-------+---------
 public | accounts                      | table | sentinel
 public | customer_behavioral_features  | table | sentinel
 public | customers                     | table | sentinel
 public | interventions                 | table | sentinel
 public | model_registry                | table | sentinel
 public | pulse_score_history           | table | sentinel
 public | transactions                  | table | sentinel
 public | users                         | table | sentinel
```

If tables are missing, apply the schema manually:

```bash
docker exec -i sentinel-postgres psql -U sentinel -d sentinel_db \
  < data_warehouse/schemas/init.sql
```

---

## 7. Step 2 — Initialise Kafka Topics

Run this **once** after Kafka is healthy. It creates the four required topics with correct partition counts:

```bash
python -m scripts.init_kafka_topics
```

Expected output:

```
Connecting to Kafka at localhost:29092...
  ✓ Created: transactions-raw (12 partitions)
  ✓ Created: transactions-enriched (12 partitions)
  ✓ Created: pulse-scores (6 partitions)
  ✓ Created: dlq-transactions (3 partitions)

Done. Created: 4, Skipped: 0
```

If you see `Skipped (already exists)` for all topics, that is fine — topics already exist from a previous run.

**Verify topics in Kafka UI:**

Open http://localhost:8080 → select cluster `sentinel-local` → click **Topics**.  
You should see all four topics listed with their partition counts.

---

## 8. Step 3 — Initialise DynamoDB Tables

Sentinel uses **DynamoDB Local** (no AWS account required). Run this once:

```bash
python -m scripts.init_dynamodb
```

Expected output:

```
✓ Created: sentinel-customer-scores (GSI: risk-tier-index)
✓ Created: sentinel-interventions
✓ Created: sentinel-audit-log
✓ Created: sentinel-transactions
```

**Verify DynamoDB tables:**

```bash
aws dynamodb list-tables \
  --endpoint-url http://localhost:8000 \
  --region ap-south-1 \
  --no-sign-request
```

Expected:

```json
{
    "TableNames": [
        "sentinel-audit-log",
        "sentinel-customer-scores",
        "sentinel-interventions",
        "sentinel-transactions"
    ]
}
```

---

## 9. Step 4 — Start the Feature Pipeline

The feature pipeline is the **core real-time engine**. It:

1. Reads every transaction from Kafka
2. Maintains a 90-day rolling transaction window per customer
3. Computes 8 behavioural signals after every transaction
4. Writes features to Redis
5. Triggers LightGBM scoring → writes Pulse Score to DynamoDB

Open a **dedicated terminal** and keep it running throughout your session:

```bash
python -m ingestion.consumers.feature_pipeline
```

Expected startup output:

```
[INFO] Warm-starting buffers from PostgreSQL...
[INFO] Warm-start complete  total_events=0
[INFO] Feature pipeline starting (real-time mode: scoring every transaction)...
```

> **What warm-start does:** On restart, the pipeline reads the last 30 days of transactions from PostgreSQL so it has a baseline — customers are not scored on an empty window. On first run, the warm-start will return 0 events (no history yet), which is expected.

**Leave this terminal open.** Every transaction processed will emit a debug log line.

---

## 10. Step 5 — Start the FastAPI Backend

Open a **second terminal**:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> **Port conflict:** DynamoDB Local also uses port 8000. Change the API port if needed:
> ```bash
> uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
> ```
> Then update the dashboard's API base URL accordingly (see Step 8).

**Wait for the startup log:**

```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
[INFO] LightGBM model loaded (local)  version=v3  cv_auc=0.9012
[INFO] SHAP explainer loaded
[INFO] Redis connected
[INFO] DynamoDB connected
```

**Verify the API is healthy:**

```bash
curl http://localhost:8001/health
```

Expected:

```json
{"status": "healthy", "service": "sentinel-api", "version": "1.0.0"}
```

**View all API endpoints:**

Open http://localhost:8001/docs for the full Swagger UI.

---

## 11. Step 6 — Run Historical (Seeded) Simulation

Before running real-time transactions, seed each customer with **90 days of historical data**. This is critical — without a historical baseline, the feature pipeline has no reference point and stress signals cannot be computed accurately.

Open a **third terminal** and run the simulator. It generates transactions, publishes them to Kafka, and writes them to PostgreSQL. The feature pipeline (Step 4) picks them up and scores each customer:

```bash
python -m scripts.simulate_transactions --customers 100 --delay 0
```

**What this does:**

- Creates 100 synthetic customers (seeded with `random.seed(42)` — deterministic and reproducible)
- Generates a realistic mix of transactions: salary credits, EMI auto-debits, UPI payments, ATM withdrawals, utility payments, lending app transfers, discretionary spend
- Publishes every transaction to Kafka topic `transactions-raw`
- Writes every transaction to PostgreSQL `transactions` table
- Scores each customer via LightGBM after every single transaction
- Writes live Pulse Score to DynamoDB and PostgreSQL `pulse_score_history`

**Console output format:**

```
       #  Customer     Transaction      Amount   Signal           Score     Δ  Streak  Tier
  ──────────────────────────────────────────────────────────────────────────────────────────
       1  CUST00001    salary_credit   ₹52,000                       12    +12       0  ● green
       2  CUST00047    auto_debit_fail  ₹8,200  [EMI FAIL]           54    +42       0  🟠 orange
       3  CUST00012    upi_lending     ₹15,000  [LENDING APP]        68    +35       0  🟠 orange
       4  CUST00001    utility_pay      ₹1,200                        9     -3       1  ● green
```

**Column guide:**

| Column | Meaning |
|--------|---------|
| `#` | Global transaction counter |
| `Customer` | Customer ID |
| `Transaction` | Transaction type |
| `Amount` | Transaction amount in INR |
| `Signal` | Stress signal label (empty = healthy) |
| `Score` | Current Pulse Score (0–100) |
| `Δ` | Score change from previous transaction |
| `Streak` | Consecutive healthy transactions for this customer |
| `Tier` | Risk tier: ● green / ◆ yellow / ▲ orange / ■ red |

**Let the simulator run until each customer has at least 30–50 transactions**, then press `Ctrl+C`. You will see a ranked summary:

```
╔══════════════════════════════════════════════════════════╗
║           SENTINEL — SESSION SUMMARY                    ║
╠══════════════════════════════════════════════════════════╣
║  Total transactions : 4,821                             ║
║  Customers scored   : 100                               ║
║                                                          ║
║  FINAL RISK DISTRIBUTION                                ║
║  🔴 Red    (≥70) :  12 customers                        ║
║  🟠 Orange (≥45) :  28 customers                        ║
║  🟡 Yellow (≥25) :  35 customers                        ║
║  🟢 Green  (< 25) :  25 customers                       ║
╚══════════════════════════════════════════════════════════╝
```

---

## 12. Step 7 — Run Real-Time Simulation

After seeding historical data (Step 6), restart the simulator for continuous real-time mode. This simulates an infinite transaction stream — exactly what production looks like:

```bash
python -m scripts.simulate_transactions --customers 100 --delay 300
```

The `--delay 300` adds 300 ms between transactions so you can watch scores change in real time on the dashboard.

**Options:**

```bash
# Fast mode — no delay, maximum throughput
python -m scripts.simulate_transactions --customers 100

# Slow mode — one transaction every 500ms, easy to follow on screen
python -m scripts.simulate_transactions --customers 100 --delay 500

# Smaller cohort — 20 customers, easy to monitor individuals
python -m scripts.simulate_transactions --customers 20 --delay 400
```

---

## 13. Step 8 — Start the Dashboard

Open a **fourth terminal**:

```bash
cd dashboards
npm install           # first time only
npm run dev
```

Dashboard opens at: **http://localhost:3000**

**Login credentials** (seeded into PostgreSQL by `init.sql`):

```
Email:    admin@sentinel.bank
Password: sentinel_admin
```

**Dashboard sections:**

| Section | Path | What you see |
|---------|------|-------------|
| Landing | `/` | System overview |
| Risk Manager | `/dashboard` | Portfolio-level risk distribution, ECL, geography heatmap |
| Customer List | `/dashboard/customers` | All 100 customers, sortable by Pulse Score, filterable by tier |
| Customer Detail | `/dashboard/customers/{id}` | Individual score, SHAP factors, transaction history |

---

## 14. Verification Checklist

### 14.1 Verify Kafka Topics & Messages

**Open Kafka UI:** http://localhost:8080

1. Click **Topics** in the left sidebar
2. Click `transactions-raw`
3. Click **Messages** tab
4. You should see every transaction as a JSON message:

```json
{
  "transaction_id": "txn-a1b2c3d4",
  "customer_id": "CUST00047",
  "amount": "15000.00",
  "txn_type": "upi_transfer",
  "merchant_category": "lending_app",
  "payment_status": "success",
  "account_type": "savings",
  "txn_timestamp": "2026-03-22T10:34:21.445Z"
}
```

5. Click `pulse-scores` topic — you should see scored results flowing through after each transaction.

**From the command line:**

```bash
# Consume the last 10 messages from transactions-raw
docker exec -it sentinel-kafka \
  kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic transactions-raw \
  --from-beginning \
  --max-messages 10
```

---

### 14.2 Verify DynamoDB Scores

Check that Pulse Scores are being written to DynamoDB after every transaction:

```bash
# List all scored customers
aws dynamodb scan \
  --table-name sentinel-customer-scores \
  --endpoint-url http://localhost:8000 \
  --region ap-south-1 \
  --no-sign-request \
  --query "Items[*].{ID:customer_id.S, Score:pulse_score.N, Tier:risk_tier.S}" \
  --output table
```

Expected output:

```
-----------------------------------------
|                  Scan                 |
+----------+-------+--------------------+
|    ID    | Score |        Tier        |
+----------+-------+--------------------+
| CUST00001|  12   |  green             |
| CUST00047|  68   |  orange            |
| CUST00012|  82   |  red               |
+----------+-------+--------------------+
```

**Check a specific customer:**

```bash
aws dynamodb get-item \
  --table-name sentinel-customer-scores \
  --key '{"customer_id": {"S": "CUST00047"}}' \
  --endpoint-url http://localhost:8000 \
  --region ap-south-1 \
  --no-sign-request
```

You should see the full item including `pulse_score`, `risk_tier`, `pd_probability`, `top_factor`, `intervention_flag`, and `updated_at`.

---

### 14.3 Verify Feature Extraction

Check that the feature pipeline is correctly extracting and storing behavioural features in Redis after each transaction:

```bash
# Inspect Redis features for a customer
docker exec -it sentinel-redis \
  redis-cli HGETALL sentinel:features:CUST00047
```

Expected output — you should see 9 computed feature values:

```
 1) "salary_delay_days"
 2) "3.0"
 3) "balance_wow_drop_pct"
 4) "0.1823"
 5) "upi_lending_spike_ratio"
 6) "4.2"
 7) "utility_payment_latency"
 8) "18.5"
 9) "discretionary_contraction"
10) "0.62"
11) "atm_withdrawal_spike"
12) "2.8"
13) "failed_auto_debit_count"
14) "2.0"
15) "credit_utilization_delta"
16) "0.041"
17) "drift_score"
18) "2.14"
19) "computed_at"
20) "2026-03-22T10:34:21.445+00:00"
```

**Key things to verify:**

- `upi_lending_spike_ratio` > 1.0 means the customer is borrowing more than their historical average
- `failed_auto_debit_count` > 0 means EMI bounces were detected in the last 14 days
- `drift_score` > 1.5 means composite behavioural drift exceeds the stress threshold
- `computed_at` timestamp should be recent (updated after each transaction)

**Check all customers with features in Redis:**

```bash
docker exec -it sentinel-redis redis-cli KEYS "sentinel:features:*" | wc -l
```

This count should equal the number of customers who have received at least one transaction.

**Verify feature pipeline in PostgreSQL:**

```bash
docker exec -it sentinel-postgres psql -U sentinel -d sentinel_db -c "
SELECT customer_id, computed_at,
       salary_delay_days, failed_auto_debit_count,
       drift_score, upi_lending_spike_ratio
FROM customer_behavioral_features
ORDER BY computed_at DESC
LIMIT 10;
"
```

---

### 14.4 Verify Per-Transaction Score Impact

This is the core verification: **does every transaction change the Pulse Score?**

**Via the API:**

```bash
# Get transaction-level score impact for a customer
curl http://localhost:8001/api/v1/customers/CUST00047/score-impact
```

Expected response — each transaction shows the model-derived score change:

```json
{
  "customer_id": "CUST00047",
  "transactions": [
    {
      "txn_timestamp": "2026-03-22T10:34:21Z",
      "txn_type": "auto_debit",
      "amount": 8200.0,
      "status": "failed",
      "score_after": 54,
      "score_impact": 42,
      "impact_reason": "Failed EMI payment — major stress signal",
      "is_stress_signal": true
    },
    {
      "txn_timestamp": "2026-03-22T09:12:05Z",
      "txn_type": "salary_credit",
      "amount": 52000.0,
      "status": "success",
      "score_after": 12,
      "score_impact": -8,
      "impact_reason": "Salary received — positive cashflow",
      "is_stress_signal": false
    }
  ],
  "score_source": "pulse_score_history"
}
```

> **Important:** `score_source` must say `pulse_score_history`. If it says `unavailable`, no historical scoring has been written yet — let the simulator run longer.

**Via PostgreSQL — full score timeline:**

```bash
docker exec -it sentinel-postgres psql -U sentinel -d sentinel_db -c "
SELECT customer_id, pulse_score, risk_tier, pd_probability, confidence,
       top_factor_1, scored_at
FROM pulse_score_history
WHERE customer_id = 'CUST00047'
ORDER BY scored_at DESC
LIMIT 20;
"
```

You should see a new row for every transaction — the score trail proves the model fires on every event.

---

### 14.5 Verify Model-Detected Stress (Not Manual Flags)

**Sentinel flags transactions by running the LightGBM model — not by hardcoded rules.** Here is how to verify this:

#### Test 1 — Confirm SHAP explains each score

```bash
curl http://localhost:8001/api/v1/score \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST00047"}'
```

Expected response:

```json
{
  "customer_id": "CUST00047",
  "pulse_score": 68,
  "risk_tier": "orange",
  "pd_probability": 0.4821,
  "confidence": 0.7243,
  "top_factors": [
    {
      "feature_name": "failed_auto_debit_count",
      "contribution": 0.1823,
      "human_readable": "Failed auto-debit attempts detected",
      "direction": "increases_risk",
      "raw_value": 2.0
    },
    {
      "feature_name": "upi_lending_spike_ratio",
      "contribution": 0.1412,
      "human_readable": "Increased UPI transactions to lending apps",
      "direction": "increases_risk",
      "raw_value": 4.2
    },
    {
      "feature_name": "salary_delay_days",
      "contribution": 0.0924,
      "human_readable": "Salary credited later than usual",
      "direction": "increases_risk",
      "raw_value": 3.0
    }
  ],
  "intervention_recommended": true,
  "intervention_type": "flexible_emi_offer",
  "scored_at": "2026-03-22T10:34:31Z",
  "model_version": "v3",
  "cached": false
}
```

**What to verify:**

- `top_factors` is populated — SHAP values are computed, not fabricated
- `contribution` values differ per customer and per transaction — the model is reading actual features
- `direction: "increases_risk"` means that feature is pushing the score higher
- `cached: false` means this was a fresh model inference, not a cached result

#### Test 2 — Verify stress is detected, not pre-labelled

The `is_stress_signal` field in `/customers/{id}/transactions` is set by the API based on transaction type — it is a **display label only**. The actual risk elevation is always driven by the model through the `/score` endpoint.

To confirm the model is doing the detection:

1. Find a customer with a recent lending-app transaction:

```bash
docker exec -it sentinel-postgres psql -U sentinel -d sentinel_db -c "
SELECT t.customer_id, t.txn_type, t.merchant_category,
       t.amount, t.txn_timestamp,
       p.pulse_score, p.pd_probability, p.top_factor_1
FROM transactions t
JOIN pulse_score_history p ON p.customer_id = t.customer_id
WHERE t.merchant_category = 'lending_app'
  AND p.scored_at >= t.txn_timestamp
ORDER BY t.txn_timestamp DESC
LIMIT 5;
"
```

2. Check that `pd_probability` is elevated and `top_factor_1` is `upi_lending_spike_ratio` — this proves the model detected the lending signal, not a hardcoded rule.

#### Test 3 — Confirm feature drift triggers without manual input

```bash
# Check which customers have drift_score > 1.5 (early stress threshold)
docker exec -it sentinel-redis redis-cli --scan --pattern "sentinel:features:*" | \
  xargs -I{} sh -c 'echo {} && docker exec sentinel-redis redis-cli HGET {} drift_score'
```

Any customer with `drift_score > 1.5` has crossed the composite stress threshold. Cross-reference with their Pulse Score via DynamoDB — elevated drift should correlate with elevated score.

---

## 15. Real-Time Stress Tracker

Run this alongside the simulator to see a live console feed of every stress event as it happens:

```bash
# In a separate terminal
python -m scripts.realtime_stress_tracker
```

Output:

```
══════════════════════════════════════════════════════════════════════
SENTINEL — Real-Time Stress Signal Tracker
  Watching PostgreSQL for new stress transactions...
  Started at : 2026-03-22 10:30:00 UTC
  Poll every : 2s
  Press Ctrl+C for full judge summary report
══════════════════════════════════════════════════════════════════════

  Time       Customer ID    Signal                                      Amount  Sev
  ────────────────────────────────────────────────────────────────────────────────
  10:34:21   CUST00047      ⚠  Failed Auto-Debit (EMI missed)       Rs.8,200  HIGH
  10:34:45   CUST00012      ⚠  Borrowed from Lending App           Rs.15,000  HIGH
  10:35:02   CUST00088      ⚡ Large ATM Withdrawal (cash hoarding)  Rs.7,500  MEDIUM
```

Press `Ctrl+C` for a ranked summary of most-stressed customers in the session, saved to `stress_signal_report.json`.

---

## 16. API Reference

All endpoints are available at `http://localhost:8001`. View interactive docs at http://localhost:8001/docs.

### Authentication

```bash
# Get a JWT token
curl -X POST http://localhost:8001/api/v1/auth/login \
  -F "username=admin@sentinel.bank" \
  -F "password=sentinel_admin"
```

Include the token in subsequent requests: `-H "Authorization: Bearer <token>"`

### Score a customer

```bash
POST /api/v1/score
Content-Type: application/json

{"customer_id": "CUST00047"}
```

### List all customers

```bash
GET /api/v1/customers?risk_tier=red&sort_by=pulse_score&limit=50
```

Query params: `risk_tier` (green/yellow/orange/red), `search`, `sort_by`, `limit`, `offset`

### Customer detail

```bash
GET /api/v1/customers/CUST00047
```

### Pulse Score history (full timeline)

```bash
GET /api/v1/customers/CUST00047/pulse-history
```

### Per-transaction score impact

```bash
GET /api/v1/customers/CUST00047/score-impact
```

### Transaction history

```bash
GET /api/v1/customers/CUST00047/transactions?limit=100
```

### Portfolio metrics

```bash
GET /api/v1/portfolio/metrics
GET /api/v1/portfolio/risk-breakdown
GET /api/v1/portfolio/geography
```

---

## 17. Known Issues & Remaining Bottlenecks

These are confirmed bugs in the current codebase. They do not prevent the system from running but affect scoring accuracy.

---

### ⚠ Issue 1 — Feature Gap: 9 Features Written to Redis, 46 Expected by Model

**File:** `features/transformations/behavioral_signals.py` — `to_feature_vector()` method (line 55–68)  
**Severity:** High  
**Impact:** The LightGBM model was trained on 46 features. The feature pipeline only writes 9 of them to Redis. The remaining 37 default to `0.0` at inference time — silently, with no warning logged.

**Missing features include:** `salary_amount_drop_pct`, `upi_lending_total_amount`, `discretionary_txn_count`, `atm_amount_spike`, `failed_auto_debit_amount`, `failed_utility_count`, all 9 `drift_*` fields, all 9 `flag_*` fields, `ead_estimate`, `total_txn_count`, `total_txn_amount`, `income_coverage_ratio`, `monthly_income`, `tenure_months`.

**Workaround in place:** The simulator (`simulate_transactions.py`) uses its own `build_feature_vector()` that computes all 46 features and scores directly — bypassing this gap. Production inference via `feature_pipeline.py → pulse_scorer.py` uses the incomplete 9-feature path.

**Fix required:**
```python
# In behavioral_signals.py, to_feature_vector() must be expanded to output all 46 fields
# OR: BehavioralSignalEngine.compute() must accept customer profile context
# (income, emi_amount, credit_limit, segment) to compute the full vector
```

---

### ⚠ Issue 2 — Sigmoid Center Not Dynamically Set from Model Threshold

**File:** `serving/bentoml_service/pulse_scorer.py` — `_load()` method  
**Severity:** Medium  
**Impact:** `scoring_utils.py` has a `set_sigmoid_center()` function designed to be called with the model's trained ECL-optimal threshold. The function exists and is correct. But `_load()` never calls it. The sigmoid center defaults to `0.42` regardless of what threshold the trained model actually produced.

**The function exists — it just isn't wired:**
```python
# scoring_utils.py — the function is there:
def set_sigmoid_center(center: float) -> None:
    global _SIGMOID_CENTER
    _SIGMOID_CENTER = center

# pulse_scorer.py _load() — this call is missing:
if self._model_package:
    threshold = self._model_package.get("threshold", 0.42)
    set_sigmoid_center(threshold)          # ← ADD THIS LINE
```

---

### ⚠ Issue 3 — JWT Secret Key Has Insecure Default

**File:** `config/settings.py` — line 28  
**Severity:** Medium  
**Impact:** `SECRET_KEY` defaults to `"dev-secret-key-change-me"`. If `.env` is absent or incomplete, all JWT tokens are signed with a public string. Any attacker can forge admin tokens.

**Fix required:** Change the field definition to remove the default, making it mandatory:
```python
# Before (insecure):
secret_key: str = Field(default="dev-secret-key-change-me")

# After (safe — startup fails if not set):
secret_key: str = Field(...)
```

Always set `SECRET_KEY` in `.env` using a 64-character random string.

---

### ⚠ Issue 4 — `force_refresh` Parameter is Still a No-Op

**File:** `serving/bentoml_service/pulse_scorer.py` — `PulseScorer.score()` and `SageMakerScorer.score()`  
**Severity:** Low (mitigated by scoring-on-every-transaction)  
**Impact:** The feature pipeline calls `scorer.score(customer_id, force_refresh=True)` expecting a guaranteed fresh ML inference. The parameter is accepted but never used inside the method body. In practice this does not cause a real problem because Redis features are always fresh (written immediately before scoring), but the parameter misleads anyone reading the code.

---

## 18. Environment Variables Reference

| Variable | Required | Default (local) | Description |
|----------|----------|-----------------|-------------|
| `SECRET_KEY` | **Yes** | *(none — set it)* | JWT signing key — must be random, 64+ chars |
| `ENVIRONMENT` | No | `development` | `development` / `staging` / `production` |
| `DATABASE_URL` | **Yes** | `postgresql://sentinel:sentinel2024@localhost:5432/sentinel_db` | PostgreSQL connection string |
| `REDIS_URL` | **Yes** | `redis://localhost:6379` | Redis connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | **Yes** | `localhost:29092` | Kafka broker(s), comma-separated |
| `AWS_REGION` | **Yes** | `ap-south-1` | AWS region (or local DynamoDB region) |
| `AWS_ACCESS_KEY_ID` | **Yes** | `local` | AWS key (set to `local` for DynamoDB Local) |
| `AWS_SECRET_ACCESS_KEY` | **Yes** | `local` | AWS secret (set to `local` for DynamoDB Local) |
| `DYNAMODB_ENDPOINT` | No | `http://localhost:8000` | DynamoDB endpoint (leave blank for real AWS) |
| `DYNAMODB_TABLE_SCORES` | No | `sentinel-customer-scores` | Live scores table name |
| `SAGEMAKER_ENDPOINT_NAME` | No | *(unset)* | Set this to use AWS SageMaker instead of local model |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Quick Reference — Terminal Map

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `docker compose up -d` | Start all infrastructure |
| 2 | `python -m ingestion.consumers.feature_pipeline` | Feature computation + scoring |
| 3 | `uvicorn api.main:app --reload --port 8001` | FastAPI backend |
| 4 | `python -m scripts.simulate_transactions --customers 100` | Transaction generator |
| 5 | `python -m scripts.realtime_stress_tracker` | Live stress feed (optional) |
| 6 | `cd dashboards && npm run dev` | Next.js dashboard |

---

## Stopping Everything

```bash
# Stop the simulator and feature pipeline with Ctrl+C in their terminals

# Stop all Docker services (data is preserved in volumes)
docker compose down

# Stop and DELETE all data (full reset)
docker compose down -v
```

---

*Sentinel AI — Built for the Indian retail banking context.*  
*Model: LightGBM · Features: 46 behavioural signals · Explainability: SHAP · Infra: Kafka + Redis + DynamoDB + PostgreSQL*