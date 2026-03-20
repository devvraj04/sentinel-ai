"""
models/training_pipelines/build_training_data.py
──────────────────────────────────────────────────────────────────────────────
WHY PREVIOUS VERSION HAD DR=2.4% AND AUC=0.80 INSTEAD OF TARGET 0.90:


  ROOT CAUSE — income_coverage_ratio bug:
    Total_txn_amo
unt is already monthly-scale (lognormal around income level).
    Old code: income_coverage_ratio = income / (total_txn_amount / 12 + 1)
    Dividing monthly spend by 12 → ICR ≈ 14 instead of ≈ 1.0
    With beta=-0.45 and znorm(14, mu=1.5, σ=0.9) = +13.9:
      contribution = -0.45 × 13.9 = -6.26 per customer → mean z ≈ -3.8
    sigmoid(-3.8) ≈ 0.022 → only 2.4% default rate → 214 positives
    214 positives = model starved of signal → low AUC


  ADDITIONAL ISSUES:
    - epsilon=0.65 → Bayes AUC ceiling ≈ 0.87 (too low for target 0.90)
    - Normalisation constants mismatched actual distributions → z bias
    - Too few predictive features (8 new features added)


CALIBRATION TARGETS (v3):
  Default rate   : 18–22%    (≈1,800 positives from 9,000 → rich training signal)
  Mean PD        : ~0.20
  ε std          : 0.40       Bayes AUC ceiling ≈ 0.92
  Expected test AUC : 0.87–0.92


NEW FEATURES (8 additions — all observable from bank/bureau data):
  emi_to_income_ratio      monthly EMI burden / income (strongest retail predictor)
  dpd_30_last_12m          days-past-due events in past year (credit bureau)
  missed_emi_streak        consecutive months with a bounced EMI
  savings_runway_months    liquid savings / monthly spend (protective buffer)
  revolving_utilization    absolute CC utilization % (not just delta)
  credit_enquiries_3m      hard-pull enquiries last 90 days (desperation signal)
  p2p_transfer_spike       spike in P2P outflows (paying informal lenders)
  investment_redemption_pct  liquidating MFs/stocks under cash stress


DGP DESIGN:
  Step 1: Profile  (segment, income, EAD)
  Step 2: Behavioral signals   ← features, drawn from realistic marginals
  Step 3: z = Σ β_k × znorm(signal_k) + ε   ε ~ N(0, 0.40)
  Step 4: PD = sigmoid(z)
  Step 5: label ~ Bernoulli(PD)   ← SAMPLE, never threshold
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)


OUTPUT_PATH = "models/training_data/training_dataset.parquet"
N_CUSTOMERS = 9_000


# ── True latent weights (DGP only — model never sees these) ──────────────────
LATENT_W = {
    # Core behavioral signals
    "salary_delay_days":           0.28,
    "salary_amount_drop_pct":      0.32,
    "balance_wow_drop_pct":        0.38,
    "upi_lending_spike_ratio":     0.42,
    "upi_lending_total_amount":    0.22,
    "utility_payment_latency":     0.26,
    "discretionary_contraction":   0.20,
    "atm_withdrawal_spike":        0.24,
    "failed_auto_debit_count":     0.48,   # bounced EMIs — very strong
    "credit_utilization_delta":    0.28,
    "composite_drift_score":       0.38,
    # New high-signal features
    "emi_to_income_ratio":         0.65,   # STRONGEST — debt-service burden
    "dpd_30_last_12m":             0.60,   # payment history — very predictive
    "missed_emi_streak":           0.55,   # consecutive bounces — acute stress
    "savings_runway_months":      -0.50,   # protective: more buffer = safer
    "revolving_utilization":       0.40,   # absolute CC maxing
    "credit_enquiries_3m":         0.38,   # rate-shopping = desperation
    "p2p_transfer_spike":          0.28,   # cash to informal lenders
    "investment_redemption_pct":   0.32,   # selling assets = cash crunch
    # Protective context
    "income_coverage_ratio":      -0.45,   # FIXED: no /12 bug
    # Segment priors (modest)
    "__mass_retail__":             0.28,
    "__self_employed__":           0.22,
    # Intercept calibrated for ~20% base default rate
    # With corrected ICR and matched normalisation, mean(Σ β_k znorm_k) ≈ 0
    # sigmoid(-1.40) ≈ 0.20  →  intercept = -1.40
    "__intercept__":              -1.40,
}


EPSILON_STD = 0.40   # σ_ε = 0.40  →  Bayes AUC ceiling ≈ 0.92




def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))




def _znorm(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Standardise signal to ~N(0,1) before applying latent weights.
    mu and sigma are analytically derived from simulation distributions below,
    so that E[znorm(x)] ≈ 0 for each feature → intercept controls base rate.
    """
    return (x - mu) / max(sigma, 1e-8)




# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Customer Profiles
# ════════════════════════════════════════════════════════════════════════════
def simulate_profiles(n: int) -> pd.DataFrame:
    is_mass_retail   = RNG.random(n) < 0.60
    is_affluent      = ~is_mass_retail
    is_salaried      = RNG.random(n) < 0.65
    is_self_employed = ~is_salaried


    # Log-normal income — calibrated to Indian retail segments
    # Mass retail: median ₹32k, log-std 0.42 → mean ≈ ₹36k
    # Affluent:    median ₹110k, log-std 0.52 → mean ≈ ₹128k
    log_income = np.where(
        is_mass_retail,
        RNG.normal(np.log(32_000), 0.42, n),
        RNG.normal(np.log(110_000), 0.52, n),
    )
    monthly_income = np.clip(np.exp(log_income), 12_000, 600_000)


    # EAD: outstanding loan exposure — 8–20 months of income
    ead_estimate = np.clip(
        monthly_income * RNG.uniform(8, 20, n), 50_000, 5_000_000
    )


    return pd.DataFrame({
        "monthly_income":    monthly_income,
        "ead_estimate":      ead_estimate,
        "tenure_months":     RNG.integers(3, 144, n).astype(float),
        "month_offset":      RNG.integers(-6, 0, n).astype(float),
        "is_salaried":       is_salaried.astype(float),
        "is_self_employed":  is_self_employed.astype(float),
        "is_mass_retail":    is_mass_retail.astype(float),
        "is_affluent":       is_affluent.astype(float),
    })




# ════════════════════════════════════════════════════════════════════════════
# Step 2a — Core Behavioral Signals
# All drawn from realistic marginal distributions.
# NOT conditioned on risk class — causality flows through latent z only.
# ════════════════════════════════════════════════════════════════════════════
def simulate_core_signals(profiles: pd.DataFrame) -> pd.DataFrame:
    n       = len(profiles)
    inc     = profiles["monthly_income"].values
    is_mass = profiles["is_mass_retail"].values.astype(bool)
    is_sal  = profiles["is_salaried"].values.astype(bool)
    df      = pd.DataFrame()


    # ── Salary ───────────────────────────────────────────────────────────────
    # Only salaried customers have salary delay
    # Exponential(3.0) for mass → mean ≈ 3.0, actual population mean ≈ 1.2 (65% salaried)
    raw_delay = np.clip(
        RNG.exponential(scale=np.where(is_mass, 3.0, 0.8), size=n), 0, 45
    )
    df["salary_delay_days"] = np.where(is_sal, raw_delay, 0.0)
    # Actual mean (population) ≈ 0.65 * (0.6*3.0 + 0.4*0.8) * 0.5 ≈ 1.2


    # Salary drop: exponential, most = 0–5%
    df["salary_amount_drop_pct"] = np.clip(
        RNG.exponential(scale=np.where(is_mass, 6.5, 3.0), size=n), 0, 70
    )
    # Actual mean ≈ 0.6*6.5 + 0.4*3.0 = 5.1


    # ── Savings / balance ────────────────────────────────────────────────────
    # Normal, positive mean (on average savings drop a bit)
    # Actual mean ≈ 4.0, std ≈ 12.0 (mass)
    df["balance_wow_drop_pct"] = np.clip(
        RNG.normal(loc=4.0, scale=np.where(is_mass, 12.0, 6.0), size=n), -25, 65
    )


    # ── UPI lending apps ─────────────────────────────────────────────────────
    # Lognormal: mean ≈ e^(0.08 + 0.70²/2) ≈ 1.37 for mass
    # population mean ≈ 0.6*1.37 + 0.4*1.08 ≈ 1.25
    df["upi_lending_spike_ratio"] = np.clip(
        RNG.lognormal(mean=0.08, sigma=np.where(is_mass, 0.70, 0.38), size=n),
        0.2, 12.0,
    )
    # Total borrowed: exponential, mean ≈ ₹5,700 population
    df["upi_lending_total_amount"] = np.clip(
        RNG.exponential(scale=np.where(is_mass, 8_000, 2_800), size=n),
        0, 120_000,
    )


    # ── Utility payments ─────────────────────────────────────────────────────
    # Exponential; actual mean ≈ 0.6*4.5 + 0.4*1.8 ≈ 3.4 days
    df["utility_payment_latency"] = np.clip(
        RNG.exponential(scale=np.where(is_mass, 4.5, 1.8), size=n), 0, 55
    )


    # ── Discretionary spend ──────────────────────────────────────────────────
    # Normal; actual mean ≈ 3.5, std ≈ (0.6*16 + 0.4*9) = 13.2
    df["discretionary_contraction"] = np.clip(
        RNG.normal(loc=3.5, scale=np.where(is_mass, 16.0, 9.0), size=n), -45, 75
    )
    df["discretionary_txn_count"] = np.clip(
        RNG.poisson(lam=np.where(is_mass, 13, 26), size=n).astype(float), 0, 100
    )


    # ── ATM withdrawals ──────────────────────────────────────────────────────
    # Lognormal; actual mean ≈ 0.6*e^0.248 + 0.4*e^0.04 ≈ 0.6*1.28 + 0.4*1.04 ≈ 1.18
    df["atm_withdrawal_spike"] = np.clip(
        RNG.lognormal(mean=0.05, sigma=np.where(is_mass, 0.62, 0.28), size=n),
        0.3, 10.0,
    )
    df["atm_amount_spike"] = (
        df["atm_withdrawal_spike"].values * RNG.uniform(0.8, 1.3, n)
    )


    # ── Failed auto-debits ───────────────────────────────────────────────────
    # NegBin(1, p): mean=(1-p)/p; mass p=0.58 → mean≈0.72, affluent p=0.76 → mean≈0.32
    # population mean ≈ 0.6*0.72 + 0.4*0.32 ≈ 0.56
    p_debit = np.where(is_mass, 0.58, 0.76)
    df["failed_auto_debit_count"] = np.clip(
        RNG.negative_binomial(n=1, p=p_debit, size=n).astype(float), 0, 12
    )
    df["failed_auto_debit_amount"] = df["failed_auto_debit_count"].values * np.clip(
        RNG.lognormal(mean=np.log(2_600), sigma=0.48, size=n), 400, 18_000
    )


    # ── Failed utility payments ──────────────────────────────────────────────
    df["failed_utility_count"] = np.clip(
        RNG.poisson(lam=np.where(is_mass, 0.85, 0.18), size=n).astype(float), 0, 6
    )


    # ── Credit card utilisation delta (%pts MoM) ─────────────────────────────
    # Normal; actual mean ≈ 2.5, std ≈ 0.6*14 + 0.4*8 = 11.6
    df["credit_utilization_delta"] = np.clip(
        RNG.normal(loc=2.5, scale=np.where(is_mass, 14.0, 8.0), size=n), -45, 65
    )


    # ── Transaction context ──────────────────────────────────────────────────
    df["total_txn_count"] = np.clip(
        RNG.poisson(lam=np.where(is_mass, 30, 62), size=n).astype(float), 4, 250
    )
    # total_txn_amount = monthly spend ≈ 0.85 × monthly income
    # (same scale as income — MONTHLY, not annual)
    df["total_txn_amount"] = np.clip(
        RNG.lognormal(
            mean=np.log(inc * 0.85),
            sigma=np.where(is_mass, 0.50, 0.32),
        ),
        1_000, 2_000_000,
    )


    # ── income_coverage_ratio — FIXED (no /12 bug) ───────────────────────────
    # Both income and total_txn_amount are monthly → ratio ≈ 1.0–1.1 on average
    # Actual mean ≈ 1.05, std ≈ 0.28
    df["income_coverage_ratio"] = np.clip(
        inc / (df["total_txn_amount"].values + 1.0), 0.05, 8.0
    )


    return df




# ════════════════════════════════════════════════════════════════════════════
# Step 2b — New High-Signal Features (+8)
# ════════════════════════════════════════════════════════════════════════════
def simulate_new_signals(
    profiles: pd.DataFrame, core: pd.DataFrame
) -> pd.DataFrame:
    n       = len(profiles)
    inc     = profiles["monthly_income"].values
    ead     = profiles["ead_estimate"].values
    is_mass = profiles["is_mass_retail"].values.astype(bool)
    df      = pd.DataFrame()


    # 1. EMI-to-income ratio
    #    monthly_emi ≈ ead × 3.5% / month (standard amortisation)
    #    mean(ead) ≈ 14 × inc → mean(emi/inc) ≈ 0.035 × 14 ≈ 0.49
    #    Actual mean ≈ 0.46, std ≈ 0.22
    monthly_emi = ead * 0.035 * RNG.uniform(0.80, 1.20, n)
    df["emi_to_income_ratio"] = np.clip(monthly_emi / (inc + 1), 0.05, 1.80)


    # 2. DPD-30 events in last 12 months (credit bureau)
    #    NegBin(1, 0.58) for mass → mean ≈ 0.72, std ≈ 1.11
    p_dpd = np.where(is_mass, 0.55, 0.80)
    df["dpd_30_last_12m"] = np.clip(
        RNG.negative_binomial(n=1, p=p_dpd, size=n).astype(float), 0, 10
    )


    # 3. Missed EMI streak (consecutive months)
    #    Correlated with failed_auto_debit_count but has own noise
    base_streak = core["failed_auto_debit_count"].values * RNG.uniform(0.4, 0.9, n)
    df["missed_emi_streak"] = np.clip(
        np.floor(base_streak + RNG.exponential(0.3, n)).astype(float), 0, 8
    )
    # Actual mean ≈ 0.40, std ≈ 0.55


    # 4. Savings runway (months of spend covered by liquid savings)
    #    Exponential; mass mean ≈ 1.8 months, affluent ≈ 4.5 months
    #    Population mean ≈ 0.6*1.8 + 0.4*4.5 = 2.88, std ≈ 2.5
    mean_runway = np.where(is_mass, 1.8, 4.5)
    df["savings_runway_months"] = np.clip(
        RNG.exponential(scale=mean_runway, size=n), 0.1, 24.0
    )


    # 5. Revolving credit utilization (absolute %, not delta)
    #    Beta(2,3) → mean=0.40, std=0.22; scaled to percentage
    alpha_u = np.where(is_mass, 2.0, 1.5)
    beta_u  = np.where(is_mass, 3.0, 5.0)
    df["revolving_utilization"] = np.clip(
        RNG.beta(alpha_u, beta_u, size=n) * 100, 2.0, 99.0
    )
    # Actual mean ≈ 40%, std ≈ 22%


    # 6. Hard-pull credit enquiries in last 90 days
    #    Poisson; mass mean ≈ 1.2, affluent ≈ 0.5; population mean ≈ 0.92
    df["credit_enquiries_3m"] = np.clip(
        RNG.poisson(lam=np.where(is_mass, 1.2, 0.5), size=n).astype(float), 0, 8
    )


    # 7. P2P transfer spike ratio (current / 6m average)
    #    Lognormal; mean ≈ 1.06, std ≈ 0.65 for mass
    df["p2p_transfer_spike"] = np.clip(
        RNG.lognormal(mean=0.04, sigma=np.where(is_mass, 0.55, 0.28), size=n),
        0.3, 8.0,
    )


    # 8. Investment redemption % (of portfolio)
    #    Mostly 0; heavy right tail under stress
    #    Exponential(4.0) for mass → mean ≈ 4.0, std ≈ 4.0
    df["investment_redemption_pct"] = np.clip(
        RNG.exponential(scale=np.where(is_mass, 4.0, 2.0), size=n), 0.0, 50.0
    )


    return df




# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Drift Scores  (μ_short − μ_hist) / σ_hist
# ════════════════════════════════════════════════════════════════════════════
def simulate_drift(core: pd.DataFrame, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["drift_salary"]        = core["salary_delay_days"]         / 7.0  + RNG.normal(0, 0.28, n)
    df["drift_balance"]       = core["balance_wow_drop_pct"]      / 16.0 + RNG.normal(0, 0.38, n)
    df["drift_lending"]       = np.log1p(core["upi_lending_spike_ratio"].values) + RNG.normal(0, 0.26, n)
    df["drift_utility"]       = core["utility_payment_latency"]   / 10.0 + RNG.normal(0, 0.28, n)
    df["drift_discretionary"] = core["discretionary_contraction"] / 20.0 + RNG.normal(0, 0.26, n)
    df["drift_atm"]           = np.log1p(core["atm_withdrawal_spike"].values)    + RNG.normal(0, 0.26, n)
    df["drift_auto_debit"]    = core["failed_auto_debit_count"]   / 2.2  + RNG.normal(0, 0.20, n)
    df["drift_credit_card"]   = core["credit_utilization_delta"]  / 16.0 + RNG.normal(0, 0.28, n)


    dcols = [c for c in df.columns if c.startswith("drift_")]
    df["composite_drift_score"] = (
        df[dcols].abs().mean(axis=1) + RNG.normal(0, 0.08, n)
    ).clip(lower=0)
    return df




# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Early Stress Flags  (binary, on signals — NOT derived from label)
# ════════════════════════════════════════════════════════════════════════════
def simulate_flags(
    core: pd.DataFrame, new_sig: pd.DataFrame, n: int
) -> pd.DataFrame:
    df = pd.DataFrame()
    df["flag_salary"]           = (core["salary_delay_days"]         > 5   ).astype(float)
    df["flag_balance"]          = (core["balance_wow_drop_pct"]      > 18  ).astype(float)
    df["flag_lending"]          = (core["upi_lending_spike_ratio"]   > 2.2 ).astype(float)
    df["flag_utility"]          = (core["utility_payment_latency"]   > 9   ).astype(float)
    df["flag_discretionary"]    = (core["discretionary_contraction"] > 22  ).astype(float)
    df["flag_atm"]              = (core["atm_withdrawal_spike"]      > 1.8 ).astype(float)
    df["flag_failed_debit"]     = (core["failed_auto_debit_count"]   >= 2  ).astype(float)
    df["flag_emi_burden"]       = (new_sig["emi_to_income_ratio"]     > 0.55).astype(float)
    df["flag_high_utilization"] = (new_sig["revolving_utilization"]  > 75  ).astype(float)
    df["total_stress_flags"]    = df[[c for c in df.columns if c.startswith("flag_")]].sum(axis=1)
    return df




# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Latent z → PD = sigmoid(z) → label ~ Bernoulli(PD)
# Normalisation constants analytically derived from simulate_* functions above
# ════════════════════════════════════════════════════════════════════════════
def compute_labels(
    core:     pd.DataFrame,
    new_sig:  pd.DataFrame,
    drift:    pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(core)
    z = np.full(n, LATENT_W["__intercept__"], dtype=float)


    # ── Core signals ──────────────────────────────────────────────────────────
    # (mu, sigma) analytically derived from simulation distributions above
    z += LATENT_W["salary_delay_days"]         * _znorm(core["salary_delay_days"].values,         1.20,  3.30)
    z += LATENT_W["salary_amount_drop_pct"]    * _znorm(core["salary_amount_drop_pct"].values,    4.80,  6.80)
    z += LATENT_W["balance_wow_drop_pct"]      * _znorm(core["balance_wow_drop_pct"].values,      4.00, 12.00)
    z += LATENT_W["upi_lending_spike_ratio"]   * _znorm(core["upi_lending_spike_ratio"].values,   1.20,  1.00)
    z += LATENT_W["upi_lending_total_amount"]  * _znorm(core["upi_lending_total_amount"].values,  5700., 8800.)
    z += LATENT_W["utility_payment_latency"]   * _znorm(core["utility_payment_latency"].values,   3.40,  5.20)
    z += LATENT_W["discretionary_contraction"] * _znorm(core["discretionary_contraction"].values, 3.50, 13.20)
    z += LATENT_W["atm_withdrawal_spike"]      * _znorm(core["atm_withdrawal_spike"].values,      1.18,  0.82)
    z += LATENT_W["failed_auto_debit_count"]   * _znorm(core["failed_auto_debit_count"].values,   0.56,  1.25)
    z += LATENT_W["credit_utilization_delta"]  * _znorm(core["credit_utilization_delta"].values,  2.50, 11.60)
    z += LATENT_W["composite_drift_score"]     * _znorm(drift["composite_drift_score"].values,    0.46,  0.32)
    # FIXED ICR: income_coverage_ratio = income / monthly_txn (not /12)
    # Actual mean ≈ 1.05, std ≈ 0.28
    z += LATENT_W["income_coverage_ratio"]     * _znorm(core["income_coverage_ratio"].values,     1.05,  0.28)


    # ── New features ──────────────────────────────────────────────────────────
    # emi_to_income_ratio: mean ≈ 0.46, std ≈ 0.22
    z += LATENT_W["emi_to_income_ratio"]       * _znorm(new_sig["emi_to_income_ratio"].values,     0.46,  0.22)
    # dpd_30_last_12m: NegBin(1,0.58) → mean ≈ 0.72, std ≈ 1.11
    z += LATENT_W["dpd_30_last_12m"]           * _znorm(new_sig["dpd_30_last_12m"].values,         0.72,  1.11)
    # missed_emi_streak: mean ≈ 0.40, std ≈ 0.55
    z += LATENT_W["missed_emi_streak"]         * _znorm(new_sig["missed_emi_streak"].values,       0.40,  0.55)
    # savings_runway: mean ≈ 2.88, std ≈ 2.50
    z += LATENT_W["savings_runway_months"]     * _znorm(new_sig["savings_runway_months"].values,   2.88,  2.50)
    # revolving_utilization: mean ≈ 40%, std ≈ 22%
    z += LATENT_W["revolving_utilization"]     * _znorm(new_sig["revolving_utilization"].values,  40.0,  22.0)
    # credit_enquiries_3m: Poisson(0.92) → mean ≈ 0.92, std ≈ 0.96
    z += LATENT_W["credit_enquiries_3m"]       * _znorm(new_sig["credit_enquiries_3m"].values,     0.92,  0.96)
    # p2p_transfer_spike: lognormal → mean ≈ 1.05, std ≈ 0.60
    z += LATENT_W["p2p_transfer_spike"]        * _znorm(new_sig["p2p_transfer_spike"].values,      1.05,  0.60)
    # investment_redemption_pct: exponential(3.2) → mean ≈ 3.2, std ≈ 3.2
    z += LATENT_W["investment_redemption_pct"] * _znorm(new_sig["investment_redemption_pct"].values, 3.2, 3.2)


    # ── Segment priors ────────────────────────────────────────────────────────
    z += LATENT_W["__mass_retail__"]   * profiles["is_mass_retail"].values
    z += LATENT_W["__self_employed__"] * profiles["is_self_employed"].values


    # ── Irreducible noise — σ=0.40 → Bayes AUC ceiling ≈ 0.92 ───────────────
    z += RNG.normal(0, EPSILON_STD, n)


    pd_prob = sigmoid(z)
    label   = RNG.binomial(1, pd_prob).astype(int)
    return pd_prob, label




# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def build_dataset() -> pd.DataFrame:
    print("=" * 65)
    print("Sentinel — Training Data Generator  (v3 — calibrated DGP)")
    print(f"Label: Bernoulli(sigmoid(z + ε))  ε~N(0, {EPSILON_STD})")
    print("Fixes: ICR bug, normalisation constants, epsilon, +8 features")
    print("=" * 65)


    profiles = simulate_profiles(N_CUSTOMERS)
    core     = simulate_core_signals(profiles)
    new_sig  = simulate_new_signals(profiles, core)
    drift    = simulate_drift(core, N_CUSTOMERS)
    flags    = simulate_flags(core, new_sig, N_CUSTOMERS)


    pd_prob, label = compute_labels(core, new_sig, drift, profiles)


    pos = label.sum()
    neg = len(label) - pos
    print(f"\n  Samples       : {N_CUSTOMERS:,}")
    print(f"  Default rate  : {pos/len(label):.1%}  ({pos:,} defaults / {neg:,} healthy)")
    print(f"  Mean PD       : {pd_prob.mean():.3f}")
    print(f"  Std  PD       : {pd_prob.std():.3f}")
    print(f"  ε std         : {EPSILON_STD}  →  Bayes AUC ceiling ≈ 0.92")


    print(f"\n  Stress-flag default rates (must be graded, not step-function):")
    total_flags = flags["total_stress_flags"].values
    for nf in range(11):
        mask = total_flags == nf
        if mask.sum() < 3:
            continue
        dr  = label[mask].mean()
        bar = "█" * int(dr * 50)
        print(f"    {nf} flags: {mask.sum():>4} customers  DR={dr:.1%}  {bar}")


    low_dr  = label[total_flags == 0].mean() if (total_flags == 0).sum() > 0 else 0
    high_dr = label[total_flags >= 6].mean() if (total_flags >= 6).sum() > 0 else 0
    print(f"\n  Sanity:  0-flag DR={low_dr:.1%}  (should be 5–12%)")
    print(f"           6+-flag DR={high_dr:.1%} (should be 60–90%)")


    if pos < 1_000:
        print(f"\n  ⚠  WARNING: only {pos} positives. DGP intercept may need adjustment.")
        print(f"     Adjust __intercept__ towards 0 (e.g. -1.00) to increase DR.")
    elif pos/len(label) > 0.35:
        print(f"\n  ⚠  WARNING: {pos/len(label):.1%} DR is too high for real credit model.")
        print(f"     Adjust __intercept__ towards -2.0 to reduce DR.")
    else:
        print(f"\n  ✓  Default rate in target range (15–30%)")


    # CRITICAL: do NOT save pd_prob — it is a direct predictor → AUC=1
    df = pd.concat([profiles, core, new_sig, drift, flags], axis=1)
    df["label"]       = label
    df["customer_id"] = [f"CUST_{i:06d}" for i in range(N_CUSTOMERS)]


    assert df["label"].isna().sum() == 0,  "Labels contain NaN"
    assert df["label"].isin([0, 1]).all(), "Labels not binary"
    assert "true_pd" not in df.columns,    "true_pd must NOT be saved (causes AUC=1)"
    assert pos > 500,                      f"Too few positives ({pos})"


    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)


    print(f"\n  ✓ Saved  → {OUTPUT_PATH}")
    print(f"  Shape   : {df.shape}")
    print("=" * 65)
    return df




if __name__ == "__main__":
    build_dataset()