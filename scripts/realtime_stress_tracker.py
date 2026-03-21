"""
scripts/realtime_stress_tracker.py
──────────────────────────────────────────────────────────────────────────────
Run this ALONGSIDE simulate_transactions.py --mode realtime

Watches PostgreSQL transactions table in real-time and logs every stress
signal as it arrives. Press Ctrl+C for a full judge-ready summary report.

Run:
  python -m scripts.realtime_stress_tracker

Shows:
  - Live feed of stress transactions as they happen
  - Customer ID, signal type, amount, timestamp
  - Full sorted report on exit (most stressed first)
  - Verification SQL to cross-check in PostgreSQL
  - Saves stress_signal_report.json for judges
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from config.settings import get_settings

settings = get_settings()

# ── Stress signal definitions ─────────────────────────────────────────────────
STRESS_SIGNALS = {
    "failed_auto_debit": {
        "condition":    lambda row: (row["txn_type"] == "auto_debit"
                                    and row["payment_status"] == "failed"),
        "label":        "⚠  Failed Auto-Debit (EMI missed)",
        "severity":     "HIGH",
        "score_impact": "model-derived",
    },
    "lending_app_upi": {
        "condition":    lambda row: row["merchant_category"] == "lending_app",
        "label":        "⚠  Borrowed from Lending App",
        "severity":     "HIGH",
        "score_impact": "model-derived",
    },
    "large_atm_withdrawal": {
        "condition":    lambda row: (row["txn_type"] == "atm_withdrawal"
                                    and float(row["amount"] or 0) >= 5000),
        "label":        "⚡ Large ATM Withdrawal (cash hoarding)",
        "severity":     "MEDIUM",
        "score_impact": "model-derived",
    },
    "failed_utility": {
        "condition":    lambda row: (row["txn_type"] == "utility_payment"
                                    and row["payment_status"] == "failed"),
        "label":        "⚡ Failed Utility Payment",
        "severity":     "MEDIUM",
        "score_impact": "model-derived",
    },
    "large_savings_drain": {
        "condition":    lambda row: (row["txn_type"] == "savings_withdrawal"
                                    and float(row["amount"] or 0) >= 10000),
        "label":        "⚡ Large Savings Withdrawal",
        "severity":     "MEDIUM",
        "score_impact": "model-derived",
    },
}

RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

SEVERITY_COLORS = {"HIGH": RED, "MEDIUM": YELLOW, "LOW": BLUE}


def get_conn():
    return psycopg2.connect(
        settings.database_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def fmt_inr(amount: float) -> str:
    return f"Rs.{amount:,.0f}"


def fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%H:%M:%S")
    return str(ts)


def run_tracker(poll_interval: float = 2.0):
    conn       = get_conn()
    start_time = datetime.now(timezone.utc)

    # ── Use txn_timestamp for tracking — works with UUID primary keys ─────────
    # Start from NOW so we only catch real-time events, not historical data
    last_seen_ts = start_time

    # customer_id → list of stress event dicts
    stress_log: dict[str, list[dict]] = defaultdict(list)
    total_stress    = 0
    total_txns_seen = 0

    print("=" * 70)
    print(f"{BOLD}SENTINEL — Real-Time Stress Signal Tracker{RESET}")
    print(f"  Watching PostgreSQL for new stress transactions...")
    print(f"  Started at : {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Poll every : {poll_interval}s")
    print(f"  Press Ctrl+C for full judge summary report")
    print("=" * 70)
    print(f"\n{'Time':>10}  {'Customer ID':<14} {'Signal':<42} {'Amount':>12}  Sev")
    print("-" * 88)

    try:
        while True:
            with conn.cursor() as cur:
                # Use txn_timestamp > last_seen_ts (works with any PK type)
                cur.execute("""
                    SELECT
                        txn_id,
                        customer_id,
                        txn_type,
                        amount,
                        merchant_category,
                        payment_channel,
                        payment_status,
                        txn_timestamp
                    FROM transactions
                    WHERE txn_timestamp > %s
                    ORDER BY txn_timestamp ASC
                    LIMIT 500
                """, (last_seen_ts,))
                rows = cur.fetchall()

            for row in rows:
                total_txns_seen += 1

                # Advance the cursor to the latest timestamp seen
                if row["txn_timestamp"] and row["txn_timestamp"] > last_seen_ts:
                    last_seen_ts = row["txn_timestamp"]

                # Check each stress signal definition
                for signal_key, signal_def in STRESS_SIGNALS.items():
                    try:
                        triggered = signal_def["condition"](row)
                    except Exception:
                        triggered = False

                    if triggered:
                        total_stress += 1
                        event = {
                            "signal_key":   signal_key,
                            "label":        signal_def["label"],
                            "severity":     signal_def["severity"],
                            "score_impact": signal_def["score_impact"],
                            "amount":       float(row["amount"] or 0),
                            "txn_type":     row["txn_type"],
                            "category":     row["merchant_category"],
                            "status":       row["payment_status"],
                            "timestamp":    row["txn_timestamp"],
                        }
                        stress_log[row["customer_id"]].append(event)

                        color = SEVERITY_COLORS.get(signal_def["severity"], "")
                        print(
                            f"{fmt_ts(row['txn_timestamp']):>10}  "
                            f"{row['customer_id']:<14} "
                            f"{color}{signal_def['label']:<42}{RESET} "
                            f"{fmt_inr(float(row['amount'] or 0)):>12}  "
                            f"{color}{signal_def['severity']}{RESET}"
                        )
                        break  # one signal label per transaction

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        _print_summary(
            stress_log, total_stress, total_txns_seen, start_time
        )
    finally:
        conn.close()


def _print_summary(
    stress_log: dict,
    total_stress: int,
    total_txns_seen: int,
    start_time: datetime,
):
    elapsed_min = (datetime.now(timezone.utc) - start_time).seconds // 60

    print("\n\n" + "=" * 70)
    print(f"{BOLD}STRESS SIGNAL SUMMARY REPORT{RESET}")
    print(f"  Runtime             : {elapsed_min} min")
    print(f"  Transactions seen   : {total_txns_seen:,}")
    print(f"  Stress signals fired: {total_stress:,}")
    print(f"  Affected customers  : {len(stress_log):,}")
    print("=" * 70)

    if not stress_log:
        print("\n  No stress signals yet.")
        print("  Make sure: python -m scripts.simulate_transactions "
              "--mode realtime  is running in another terminal.")
        return

    # Sort by number of signals descending
    sorted_customers = sorted(
        stress_log.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    # ── Ranked table ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}CUSTOMERS RANKED BY STRESS SIGNALS:{RESET}")
    print(f"\n  {'Rank':<6} {'Customer ID':<14} {'Signals':>8} "
          f"{'HIGH':>6} {'MED':>6} {'Total Amount':>16}  Top Signal")
    print("  " + "-" * 78)

    for rank, (cid, events) in enumerate(sorted_customers, 1):
        high_cnt  = sum(1 for e in events if e["severity"] == "HIGH")
        med_cnt   = sum(1 for e in events if e["severity"] == "MEDIUM")
        total_amt = sum(e["amount"] for e in events)
        top_sig   = (events[0]["label"]
                     .replace("⚠  ", "")
                     .replace("⚡ ", ""))
        color = RED if high_cnt > 0 else YELLOW
        print(
            f"  {color}{rank:<6} {cid:<14} {len(events):>8} "
            f"{high_cnt:>6} {med_cnt:>6} "
            f"{fmt_inr(total_amt):>16}  {top_sig}{RESET}"
        )

    # ── Detailed top 10 ───────────────────────────────────────────────────────
    print(f"\n{BOLD}DETAILED BREAKDOWN — TOP 10 MOST STRESSED CUSTOMERS:{RESET}")
    print("(Use these IDs in dashboard + SQL to verify)\n")

    for rank, (cid, events) in enumerate(sorted_customers[:10], 1):
        high = sum(1 for e in events if e["severity"] == "HIGH")
        color = RED if high > 0 else YELLOW
        print(f"  {color}{BOLD}#{rank}  {cid}{RESET}  — {len(events)} signals  "
              f"(HIGH={high})")
        signal_counts: dict[str, int] = defaultdict(int)
        for e in events:
            signal_counts[e["label"]] += 1
        for label, cnt in sorted(signal_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"       {label:<46}  x{cnt}")
        last_ts   = max((e["timestamp"] for e in events if e["timestamp"]),
                        default=None)
        total_amt = sum(e["amount"] for e in events)
        print(f"       Last signal : {fmt_ts(last_ts)}  |  "
              f"Total : {fmt_inr(total_amt)}\n")

    # ── Signal type breakdown ─────────────────────────────────────────────────
    print(f"{BOLD}SIGNAL TYPE BREAKDOWN:{RESET}")
    signal_totals: dict[str, int] = defaultdict(int)
    for events in stress_log.values():
        for e in events:
            signal_totals[e["signal_key"]] += 1

    for sig_key, count in sorted(signal_totals.items(),
                                   key=lambda x: x[1], reverse=True):
        label = STRESS_SIGNALS[sig_key]["label"]
        bar   = "█" * min(count, 35)
        print(f"  {label:<46} {count:>5}  {bar}")

    # ── Verification SQL ──────────────────────────────────────────────────────
    top_ids = [cid for cid, _ in sorted_customers[:5]]
    ids_str = "', '".join(top_ids)

    print(f"\n{BOLD}VERIFICATION SQL — paste into PostgreSQL:{RESET}")
    print(f"""
  \\c sentinel_db

  -- 1. Stress signals for top stressed customers:
  SELECT
      customer_id,
      COUNT(*) FILTER (WHERE txn_type = 'auto_debit'
                       AND payment_status = 'failed')          AS failed_emis,
      COUNT(*) FILTER (WHERE merchant_category = 'lending_app') AS lending_txns,
      COUNT(*) FILTER (WHERE txn_type = 'atm_withdrawal'
                       AND amount >= 5000)                     AS large_atm,
      COUNT(*) FILTER (WHERE txn_type = 'utility_payment'
                       AND payment_status = 'failed')          AS failed_utils,
      COUNT(*) FILTER (WHERE txn_type = 'savings_withdrawal'
                       AND amount >= 10000)                    AS large_drain,
      ROUND(SUM(amount)::numeric, 0)                           AS total_amount,
      MAX(txn_timestamp)                                       AS last_txn
  FROM transactions
  WHERE customer_id IN ('{ids_str}')
  GROUP BY customer_id
  ORDER BY failed_emis DESC, lending_txns DESC;

  -- 2. No future dates check:
  SELECT MAX(txn_timestamp) AS latest, NOW() AS now_utc
  FROM transactions;

  -- 3. All failed EMIs (most critical signal):
  SELECT customer_id, COUNT(*) AS failed_count,
         SUM(amount) AS total_missed
  FROM transactions
  WHERE txn_type = 'auto_debit' AND payment_status = 'failed'
  GROUP BY customer_id
  ORDER BY failed_count DESC LIMIT 10;
""")

    # ── Judge talking points ──────────────────────────────────────────────────
    if sorted_customers:
        top_cid, top_events = sorted_customers[0]
        high = sum(1 for e in top_events if e["severity"] == "HIGH")
        print(f"{BOLD}JUDGE DEMO TALKING POINTS:{RESET}")
        print(f"""
  1. Search '{top_cid}' in dashboard
     → Pulse score should be orange/red tier
     → Click View → Score Impact tab
     → See {len(top_events)} stress transactions driving the score up

  2. Sentinel detected {total_stress} stress signals across
     {len(stress_log)} customers in {elapsed_min} minutes of real-time data

  3. Signal breakdown:
{chr(10).join(f"     - {STRESS_SIGNALS[k]['label']}: {v}x" for k, v in signal_totals.items())}

  4. These signals feed into the drift formula:
     Drift_f = (mu_short - mu_hist) / sigma_hist
     When |Drift_f| > theta → Early Stress Flag fires
     LightGBM + LSTM ensemble then computes final PD score
""")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "runtime_minutes":         elapsed_min,
        "total_transactions_seen": total_txns_seen,
        "total_stress_signals":    total_stress,
        "affected_customers":      len(stress_log),
        "signal_breakdown":        dict(signal_totals),
        "customers": [
            {
                "rank":               rank,
                "customer_id":        cid,
                "stress_signal_count":len(events),
                "high_severity":      sum(1 for e in events if e["severity"] == "HIGH"),
                "medium_severity":    sum(1 for e in events if e["severity"] == "MEDIUM"),
                "total_amount":       round(sum(e["amount"] for e in events), 2),
                "signals": [
                    {
                        "signal":    e["label"],
                        "severity":  e["severity"],
                        "amount":    e["amount"],
                        "timestamp": str(e["timestamp"]),
                    }
                    for e in events
                ],
            }
            for rank, (cid, events) in enumerate(sorted_customers, 1)
        ],
    }
    report_path = "stress_signal_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  {BOLD}✓ Full report saved → {report_path}{RESET}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Real-time stress signal tracker for Sentinel"
    )
    p.add_argument(
        "--interval", type=float, default=2.0,
        help="Poll interval in seconds (default: 2.0)"
    )
    args = p.parse_args()
    run_tracker(poll_interval=args.interval)