"""
ingestion/enrichment/transaction_classifier.py
──────────────────────────────────────────────────────────────────────────────
Infers transaction type from raw transaction signals (counterparty, platform,
MCC, amount patterns). The producer sends RAW data — this classifier labels it.

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

class TransactionClassifier:
    """
    Infers transaction type from raw signals.

    The producer sends raw transactions with:
      - counterparty_id (e.g., "swiggy@upi")
      - counterparty_name (e.g., "Swiggy")
      - platform (UPI/NEFT/IMPS/RTGS/NACH/BBPS/POS/ATM/NetBanking)
      - merchant_category (MCC-based or keyword)
      - amount
      - payment_status

    This classifier returns:
      - txn_type: The inferred transaction type enum value
      - is_lending_app_upi: True if UPI to a lending/NBFC app
      - is_p2p_transfer: True if peer-to-peer UPI transfer
      - is_investment_txn: True if investment/FD/MF transaction
      - is_auto_debit_failed: True if NACH auto-debit bounced
    """

    @staticmethod
    def classify(
        amount: float,
        platform: str,
        counterparty_id: Optional[str],
        counterparty_name: Optional[str],
        merchant_category: str,
        payment_status: str,
        account_type: str,
        is_credit: bool = False,
    ) -> dict:
        """
        Classify a raw transaction into a typed transaction.

        Returns dict with:
          txn_type, is_lending_app_upi, is_p2p_transfer,
          is_investment_txn, is_auto_debit_failed
        """
        result = {
            "txn_type": "other",
            "is_lending_app_upi": False,
            "is_p2p_transfer": False,
            "is_investment_txn": False,
            "is_auto_debit_failed": False,
        }

        cp_id = (counterparty_id or "").lower().strip()
        cp_name = (counterparty_name or "").lower().strip()
        plat = (platform or "").lower().strip()
        cat = (merchant_category or "").lower().strip()
        status = (payment_status or "").lower().strip()
        acct = (account_type or "").lower().strip()
        combined = f"{cp_id} {cp_name}"

        # ── Rule 1: Salary credit (NEFT/IMPS credit from known employer)
        if is_credit and _matches_any(combined, SALARY_KEYWORDS):
            result["txn_type"] = "salary_credit"
            return result

        # ── Rule 2: NACH/ECS auto-debit (EMI)
        if plat in ("nach", "ecs", "ecs_debit") or _matches_any(combined, EMI_KEYWORDS):
            result["txn_type"] = "auto_debit"
            if status == "failed":
                result["is_auto_debit_failed"] = True
            return result

        # ── Rule 3: Lending app UPI
        if cp_id in LENDING_APP_VPAS or _matches_any(combined, LENDING_APP_KEYWORDS):
            result["txn_type"] = "upi_debit"
            result["is_lending_app_upi"] = True
            return result

        # ── Rule 4: ATM withdrawal
        if plat == "atm" or "atm" in combined:
            result["txn_type"] = "atm_withdrawal"
            return result

        # ── Rule 5: Utility payment (BBPS or known biller)
        if plat == "bbps" or _matches_any(combined, UTILITY_BILLER_KEYWORDS):
            result["txn_type"] = "utility_payment"
            return result

        # ── Rule 6: Investment / MF / FD
        if _matches_any(combined, INVESTMENT_KEYWORDS) or cat in ("investment", "mutual_fund"):
            result["txn_type"] = "neft_rtgs" if is_credit else "upi_debit"
            result["is_investment_txn"] = True
            return result

        # ── Rule 7: Credit card payment
        if acct == "credit_card" or cat == "credit_card":
            result["txn_type"] = "credit_card_payment"
            return result

        # ── Rule 8: P2P UPI transfer (individual recipient, not merchant)
        if plat == "upi" and _is_p2p_counterparty(cp_id, cp_name):
            if is_credit:
                result["txn_type"] = "upi_credit"
            else:
                result["txn_type"] = "upi_debit"
                result["is_p2p_transfer"] = True
            return result

        # ── Rule 9: Merchant-category based (dining/grocery/shopping/fuel)
        if _matches_any(combined, DINING_KEYWORDS) or cat == "dining":
            result["txn_type"] = "upi_debit" if plat == "upi" else "pos_debit"
            return result

        if _matches_any(combined, GROCERY_KEYWORDS) or cat in ("groceries", "grocery"):
            result["txn_type"] = "upi_debit" if plat == "upi" else "pos_debit"
            return result

        if _matches_any(combined, SHOPPING_KEYWORDS) or cat == "shopping":
            result["txn_type"] = "upi_debit" if plat == "upi" else "pos_debit"
            return result

        if _matches_any(combined, FUEL_KEYWORDS) or cat == "fuel":
            result["txn_type"] = "upi_debit" if plat == "upi" else "pos_debit"
            return result

        # ── Rule 10: Large savings withdrawal (no matching counterparty)
        if acct in ("savings", "current") and not is_credit and amount >= 5000:
            if plat in ("neft", "imps", "rtgs"):
                result["txn_type"] = "neft_rtgs"
            else:
                result["txn_type"] = "savings_withdrawal"
            return result

        # ── Rule 11: UPI credit (incoming)
        if is_credit and plat == "upi":
            result["txn_type"] = "upi_credit"
            return result

        # ── Rule 12: NEFT/IMPS/RTGS credit
        if is_credit and plat in ("neft", "imps", "rtgs"):
            result["txn_type"] = "neft_rtgs"
            return result

        # ── Default: generic UPI debit or other
        if plat == "upi":
            result["txn_type"] = "upi_debit"
        elif plat == "pos":
            result["txn_type"] = "pos_debit"
        else:
            result["txn_type"] = "other"

        return result


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
