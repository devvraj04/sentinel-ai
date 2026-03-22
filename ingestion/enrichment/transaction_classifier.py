"""
ingestion/enrichment/transaction_classifier.py
──────────────────────────────────────────────────────────────────────────────
Infers transaction purpose from raw counterparty and platform facts.

DESIGN PRINCIPLE:
  This classifier reads the raw sender_id, receiver_id, sender_name,
  receiver_name, platform, and payment_status from the TransactionEvent
  and returns an inferred classification dict used ONLY by the feature
  pipeline for computing features. It does NOT modify the TransactionEvent.
  The model never sees these inferences directly — it sees the Z-scores
  and ratios computed from them.

Indian Banking Context:
  - Known UPI lending apps: Slice, LazyPay, KreditBee, MoneyView, etc.
  - Known salary patterns: NEFT/IMPS credits from corporate accounts
  - Known utility billers: BESCOM, BSNL, Jio, Airtel, Mahanagar Gas, etc.
  - NACH/ECS → recurring EMI auto-debits
  - BBPS → utility bill payments
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from typing import Optional


# ── Known counterparty databases (Indian context) ─────────────────────────────

LENDING_APP_VPAS = frozenset({
    "slice@upi", "lazypay@upi", "kreditbee@upi", "moneyview@upi",
    "navi@upi", "kissht@upi", "dhani@upi", "stashfin@upi",
    "cashe@upi", "lendingkart@upi", "indifi@upi", "bajajfinserv@upi",
    "moneytap@upi", "truebalance@upi", "earlyslary@upi", "fibe@upi",
    "mpokket@upi", "prefr@upi", "loanfront@upi", "paysense@upi",
})

LENDING_APP_KEYWORDS = frozenset({
    "slice", "lazypay", "kreditbee", "moneyview", "navi", "kissht",
    "dhani", "stashfin", "cashe", "lendingkart", "moneytap", "fibe",
    "mpokket", "paysense", "loanfront", "earlyslary", "prefr",
    "bajaj_finserv_loan", "home_credit", "tvs_credit",
})

UTILITY_BILLER_KEYWORDS = frozenset({
    "bescom", "bsnl", "jio", "airtel", "vodafone", "vi_postpaid",
    "mahanagar_gas", "adani_gas", "igl", "tata_power", "reliance_energy",
    "mseb", "tneb", "apspdcl", "cesc", "torrent_power",
    "hathway", "act_fibernet", "airtel_broadband", "jio_fiber",
    "dish_tv", "tata_sky", "d2h", "sun_direct",
    "bwssb", "delhi_jal_board", "municipal",
})

SALARY_KEYWORDS = frozenset({
    "salary", "payroll", "wages", "stipend", "pension",
    "tcs_payroll", "infosys_hr", "wipro_salary", "hcl_salary",
    "cognizant_pay", "accenture_pay", "capgemini_pay",
    "hdfc_salary", "sbi_salary", "icici_salary", "axis_salary",
    "gov_salary", "pension_fund", "epfo",
})

INVESTMENT_KEYWORDS = frozenset({
    "mutual_fund", "sip", "zerodha", "groww", "kuvera", "coin",
    "lic", "sbi_mf", "hdfc_mf", "icici_pru", "nippon_mf",
    "fd_withdrawal", "rd_maturity", "ppf", "nps", "sgb",
    "smallcase", "upstox", "angel_one", "motilal_oswal",
})

EMI_KEYWORDS = frozenset({
    "emi", "loan_emi", "home_loan", "car_loan", "personal_loan",
    "education_loan", "auto_loan", "nach_mandate", "ecs_debit",
    "hdfc_home_loan", "sbi_car_loan", "bajaj_personal", "axis_loan",
    "lic_housing", "pnb_housing", "kotak_prime",
})

P2P_PLATFORMS = frozenset({
    "gpay", "phonepe", "paytm", "amazonpay", "whatsapp_pay",
    "bhim", "mobikwik", "freecharge",
})

DINING_KEYWORDS = frozenset({
    "swiggy", "zomato", "dominos", "mcdonalds", "kfc", "pizza_hut",
    "starbucks", "ccd", "burger_king", "subway", "haldirams",
    "barbeque_nation", "paradise", "mtr", "saravana_bhavan",
})

GROCERY_KEYWORDS = frozenset({
    "bigbasket", "dmart", "reliance_fresh", "more_supermarket",
    "star_bazaar", "spencers", "nature_basket", "jiomart",
    "blinkit", "zepto", "dunzo", "swiggy_instamart",
})

SHOPPING_KEYWORDS = frozenset({
    "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa",
    "tata_cliq", "croma", "reliance_digital", "decathlon",
    "shoppers_stop", "lifestyle", "pantaloons", "max_fashion",
})

FUEL_KEYWORDS = frozenset({
    "hp_petrol", "ioc", "bpcl", "indian_oil", "reliance_petrol",
    "shell", "nayara", "essar",
})


# ── Classification Engine ─────────────────────────────────────────────────────

def classify(
    sender_id: str | None,
    receiver_id: str | None,
    sender_name: str | None,
    receiver_name: str | None,
    platform: str,
    payment_status: str,
    amount: float,
) -> dict:
    """
    Infer transaction purpose from raw counterparty and platform facts.

    Returns a classification dict used by build_feature_vector() ONLY.
    This classification is NOT stored on the TransactionEvent.
    The model never directly reads these fields.

    The inference chain:
    1. receiver_id/name against known VPA databases (lending apps, utilities, etc.)
    2. platform type (ECS → likely EMI, NEFT_credit → likely salary)
    3. Amount patterns (e.g. round ₹X,000 from NACH → EMI)
    4. Keyword matching on receiver_name
    """
    r_id   = (receiver_id   or "").lower()
    s_id   = (sender_id     or "").lower()
    r_name = (receiver_name or "").lower()
    s_name = (sender_name   or "").lower()
    plat   = platform.lower()

    # ── Lending app detection ─────────────────────────────────────────────────
    is_likely_lending = (
        any(k in r_id for k in LENDING_APP_VPAS) or
        any(k in r_name for k in LENDING_APP_KEYWORDS) or
        any(k in r_id for k in LENDING_APP_KEYWORDS)
    )

    # ── Salary detection ──────────────────────────────────────────────────────
    # Salary arrives via NEFT/IMPS from employer, not UPI from individual
    is_likely_salary = (
        plat in ("neft", "imps", "rtgs") and
        (any(k in s_name for k in SALARY_KEYWORDS) or
         any(k in s_id  for k in SALARY_KEYWORDS)) and
        amount > 5000  # salaries are not small amounts
    )

    # ── EMI / auto-debit detection ────────────────────────────────────────────
    # ECS/NACH debits are almost always EMIs
    is_likely_emi = (
        plat in ("ecs", "nach", "mandate") or
        any(k in r_name for k in EMI_KEYWORDS) or
        any(k in r_id   for k in EMI_KEYWORDS)
    )

    # ── Utility bill detection ────────────────────────────────────────────────
    is_likely_utility = (
        plat == "bbps" or
        any(k in r_id   for k in UTILITY_BILLER_KEYWORDS) or
        any(k in r_name for k in UTILITY_BILLER_KEYWORDS)
    )

    # ── ATM cash withdrawal ───────────────────────────────────────────────────
    is_atm = plat == "atm"

    # ── P2P transfer ──────────────────────────────────────────────────────────
    is_p2p = (
        plat == "upi" and
        not is_likely_lending and not is_likely_utility and
        not any(k in r_id for k in list(DINING_KEYWORDS) + list(GROCERY_KEYWORDS))
    )

    # ── Investment redemption ─────────────────────────────────────────────────
    is_investment_redeem = any(k in r_id + r_name for k in INVESTMENT_KEYWORDS)

    # ── Direction ────────────────────────────────────────────────────────────
    # Credit: salary, loan disbursement, P2P received, reversal
    # Debit: all outflows
    is_credit = is_likely_salary or (
        plat in ("neft", "imps", "rtgs") and
        any(k in s_name for k in SALARY_KEYWORDS)
    )

    # ── Inferred purpose ─────────────────────────────────────────────────────
    if is_likely_salary:       purpose = "salary"
    elif is_likely_emi:        purpose = "emi"
    elif is_likely_lending:    purpose = "lending_borrow"
    elif is_likely_utility:    purpose = "utility"
    elif is_atm:               purpose = "atm_cash"
    elif is_investment_redeem: purpose = "investment_redeem"
    else:                      purpose = "other"

    return {
        "inferred_purpose":       purpose,
        "inferred_direction":     "credit" if is_credit else "debit",
        "is_likely_emi":          is_likely_emi,
        "is_likely_salary":       is_likely_salary,
        "is_likely_lending":      is_likely_lending,
        "is_likely_utility":      is_likely_utility,
        "is_likely_atm":          is_atm,
        "is_likely_p2p":          is_p2p,
        "is_likely_investment":   is_investment_redeem,
        "payment_failed":         payment_status == "failed",
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _matches_any(text: str, keywords: frozenset) -> bool:
    """Check if any keyword appears in the text."""
    for kw in keywords:
        if kw in text:
            return True
    return False


def _is_p2p_counterparty(cp_id: str, cp_name: str) -> bool:
    """
    Heuristic: P2P if counterparty looks like an individual
    (contains @ for UPI VPA, not a known merchant/biller).
    """
    if not cp_id:
        return False

    # Known merchant VPAs have specific patterns
    all_merchants = (
        LENDING_APP_VPAS | UTILITY_BILLER_KEYWORDS
        | DINING_KEYWORDS | GROCERY_KEYWORDS | SHOPPING_KEYWORDS
    )
    if cp_id in all_merchants or _matches_any(cp_id, all_merchants):
        return False

    # Individual VPAs: firstname.lastname@bank or phone@upi
    if "@" in cp_id:
        local_part = cp_id.split("@")[0]
        # Phone number pattern
        if re.match(r"^\d{10}$", local_part):
            return True
        # Name pattern (contains dot or alphabetic)
        if re.match(r"^[a-z]+\.?[a-z]*$", local_part):
            return True

    return False
