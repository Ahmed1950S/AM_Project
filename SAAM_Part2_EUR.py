"""
SAAM Part II — Carbon-Aware Portfolio Allocation (EUR / Scope 1+2)

This script CONTINUES from Part I and assumes the following variables are in
memory (run Part I in the same Python session before this script):
    universe, mv_w_dict, ri_m, ret_m, mv_y, mv_m, co2_s1, co2_s2, rev,
    isin_name, static, START_YEAR, END_YEAR, OUT, monthly_all,
    months_of, estim_window, fill_oos_returns, estimate_cov,
    rp_mv, rp_vw, rf_mon, compute_perf

Strategy: Region = EUR, Scope = Scope 1 + Scope 2 (sum), θ_NZ = 10% / year
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

print("\n" + "=" * 65)
print("SAAM Part II — Carbon-Aware Portfolios (EUR / Scope 1+2)")
print("=" * 65)

# =============================================================================
# 15. CARBON DATA PREPARATION
# =============================================================================
print("\n[14] Preparing carbon data ...")

# Total emissions = Scope 1 + Scope 2 (assigned strategy)
# Note: Part I universe filter ensures both are present for in-sample firms,
# so simple addition (NaN if either NaN) is correct here.
co2_tot = co2_s1 + co2_s2

# Revenue: data is in thousands USD, project requires divide by 1000 → millions USD
rev_m = rev / 1000.0

# Firm-level carbon intensity (tCO2 / million USD revenue)
# (Computed once for reuse)
with np.errstate(divide="ignore", invalid="ignore"):
    CI_firm = co2_tot / rev_m

# Diagnostics: data coverage in our investment years
years_part2 = list(range(START_YEAR, END_YEAR + 1))
diag = []
for Y in years_part2:
    isins = universe[Y]
    diag.append({
        "Y": Y,
        "Universe": len(isins),
        "Emissions": int(co2_tot.loc[isins, Y].notna().sum()) if Y in co2_tot.columns else 0,
        "Revenue":   int(rev_m.loc[isins, Y].notna().sum())   if Y in rev_m.columns   else 0,
        "Cap_yr":    int(mv_y.loc[isins, Y].notna().sum())    if Y in mv_y.columns    else 0,
    })
diag_df = pd.DataFrame(diag).set_index("Y")
print(diag_df.to_string())

# --- Verification 1: emissions coverage = 100% (universe filter invariant) ---
miss_E = int((diag_df["Universe"] - diag_df["Emissions"]).sum())
assert miss_E == 0, f"Emissions missing in universe (Part I filter broken): {miss_E}"
print(f"   ✓ Emissions: 100% coverage in universe (filter invariant holds)")

# --- Diagnostic: revenue / cap coverage gaps (the issue flagged in advance) ---
miss_R = int((diag_df["Universe"] - diag_df["Revenue"]).sum())
miss_C = int((diag_df["Universe"] - diag_df["Cap_yr"]).sum())
print(f"   Coverage gaps:  Revenue missing {miss_R} firm-yrs, Cap missing {miss_C}")


# =============================================================================
# 16. CARBON METRICS — BASELINE PORTFOLIOS (P^(mv)_oos and P^(vw))
# =============================================================================
print("\n[15] Computing baseline portfolio carbon metrics ...")


def carbon_metrics(weights, Y):
    """
    Compute (WACI, CF) for portfolio weights at year-end Y.

    weights : pd.Series indexed by ISIN; should sum to ~1.
    Returns dict with:
        WACI   : tCO2 / million USD revenue (PCAF metric)
        CF     : tCO2 / million USD invested (ownership-attributed footprint)
        cov_W  : fraction of weight with valid CI used in WACI
        cov_F  : fraction of weight with valid (E/Cap) used in CF
    """
    isins = list(weights.index)
    w = weights.values.astype(float)

    # Pull aligned arrays
    E = co2_tot.loc[isins, Y].values.astype(float)   # tonnes
    R = rev_m.loc[isins, Y].values.astype(float)     # M$ revenue
    C = mv_y.loc[isins, Y].values.astype(float)      # M$ market cap

    # Per-firm carbon intensity: NaN where revenue is missing or non-positive
    with np.errstate(divide="ignore", invalid="ignore"):
        CI = np.where((R > 0) & np.isfinite(R), E / R, np.nan)
    valid_W = np.isfinite(CI)
    WACI = float(np.nansum(w * np.where(valid_W, CI, 0.0)))
    cov_W = float(w[valid_W].sum())

    # Per-firm emissions per dollar of cap
    with np.errstate(divide="ignore", invalid="ignore"):
        ED = np.where((C > 0) & np.isfinite(C), E / C, np.nan)
    valid_F = np.isfinite(ED)
    CF = float(np.nansum(w * np.where(valid_F, ED, 0.0)))
    cov_F = float(w[valid_F].sum())

    return {"WACI": WACI, "CF": CF, "cov_W": cov_W, "cov_F": cov_F}


def vw_weights(Y):
    """End-of-year value weights for the year-Y universe."""
    isins = universe[Y]
    cap = mv_y.loc[isins, Y].fillna(0.0)
    s = cap.sum()
    if s <= 0:
        return pd.Series(np.ones(len(isins)) / len(isins), index=isins)
    return cap / s


# Time series of carbon metrics
mv_carbon_rows, vw_carbon_rows = [], []
for Y in years_part2:
    w_mv = mv_w_dict[Y]
    w_vw = vw_weights(Y)
    mv_carbon_rows.append({"Y": Y, **carbon_metrics(w_mv, Y)})
    vw_carbon_rows.append({"Y": Y, **carbon_metrics(w_vw, Y)})

mv_carbon = pd.DataFrame(mv_carbon_rows).set_index("Y")
vw_carbon = pd.DataFrame(vw_carbon_rows).set_index("Y")

print("\n   Min-Variance portfolio P^(mv)_oos:")
print(mv_carbon.round(2).to_string())
print("\n   Value-Weighted portfolio P^(vw):")
print(vw_carbon.round(2).to_string())


# =============================================================================
# 16b. VERIFICATION OF CARBON METRICS
# =============================================================================
print("\n[15b] Verification of carbon metrics ...")

# --- A. VW carbon footprint via the alternative aggregate formula ---
# The project shows: CF(P^(vw))_Y = (1/Cap_Y) * Σ E_i,Y
# Weighted form Σ w_i (E_i/Cap_i) with w_i = Cap_i/Cap_Y simplifies to the same
# value ONLY if we have Cap_i for ALL firms with valid emissions. Verify:
print("   A. VW CF: weighted form vs aggregate form")
for Y in years_part2:
    isins = universe[Y]
    cap_Y = mv_y.loc[isins, Y].fillna(0.0)
    E_Y = co2_tot.loc[isins, Y].fillna(0.0)
    cf_agg = float(E_Y.sum() / cap_Y.sum()) if cap_Y.sum() > 0 else np.nan
    cf_w = vw_carbon.loc[Y, "CF"]
    rel_err = abs(cf_agg - cf_w) / max(abs(cf_w), 1e-12)
    if rel_err > 1e-8:
        print(f"      Y={Y}: aggregate={cf_agg:.4f} vs weighted={cf_w:.4f}  (rel err {rel_err:.2e})")
print("      ✓ done — relative errors above 1e-8 listed (none expected)")

# --- B. Coverage health ---
print("   B. Coverage of WACI/CF (fraction of weight with valid metric)")
print(f"      MV WACI cov: min={mv_carbon['cov_W'].min():.3f}, mean={mv_carbon['cov_W'].mean():.3f}")
print(f"      MV CF   cov: min={mv_carbon['cov_F'].min():.3f}, mean={mv_carbon['cov_F'].mean():.3f}")
print(f"      VW WACI cov: min={vw_carbon['cov_W'].min():.3f}, mean={vw_carbon['cov_W'].mean():.3f}")
print(f"      VW CF   cov: min={vw_carbon['cov_F'].min():.3f}, mean={vw_carbon['cov_F'].mean():.3f}")
if min(mv_carbon["cov_W"].min(), vw_carbon["cov_W"].min()) < 0.98:
    print("      ⚠ coverage < 98% somewhere — consider tightening the Part I universe filter")
else:
    print("      ✓ coverage ≥ 98% throughout — universe filter is fine in practice")

# --- C. Plausibility: WACI/CF magnitudes for European equities ---
print("   C. Plausibility of magnitudes (European equities, 2014–2025)")
print(f"      VW WACI mean = {vw_carbon['WACI'].mean():.0f} tCO2/M$ rev")
print(f"      VW CF   mean = {vw_carbon['CF'].mean():.0f} tCO2/M$ invested")
# Typical ranges for European equity benchmarks (MSCI Europe, STOXX 600):
#   WACI ≈ 100–250 tCO2/M$ revenue; CF ≈ 80–250 tCO2/M$ invested.
plausible = (50 <= vw_carbon['WACI'].mean() <= 600) and (30 <= vw_carbon['CF'].mean() <= 500)
print(f"      {'✓' if plausible else '⚠'} {'in' if plausible else 'OUTSIDE'} plausible range")

# --- D. Direction: low-vol typically tilts toward low-carbon firms? ---
mv_lower = (mv_carbon["WACI"] < vw_carbon["WACI"]).sum()
print(f"   D. MV WACI < VW WACI in {mv_lower} of {len(years_part2)} years")
print("      (low-vol tilts to staples/healthcare; expect MV < VW typically)")


# =============================================================================
# 17. TOP CARBON CONTRIBUTORS
# =============================================================================
print("\n[16] Top carbon contributors ...")

def top_n_by_CI(Y, n=10):
    """Top n firms in universe by raw carbon intensity."""
    isins = universe[Y]
    CI = (co2_tot.loc[isins, Y] / rev_m.loc[isins, Y]).dropna()
    CI = CI.sort_values(ascending=False)
    rows = []
    for rk, isin in enumerate(CI.head(n).index, 1):
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        rows.append({
            "Rank": rk, "ISIN": isin,
            "Name": isin_name.get(isin, isin),
            "Country": cty[0] if len(cty) else "",
            "CI (tCO2/M$rev)": round(float(CI.loc[isin]), 1),
            "VW weight (%)": round(float(vw_weights(Y).get(isin, 0)) * 100, 3),
        })
    return pd.DataFrame(rows)


def top_n_by_contrib(Y, weights, n=10, label="WACI"):
    """Top n firms by contribution (w_i * CI_i) to portfolio WACI."""
    isins = list(weights.index)
    CI = (co2_tot.loc[isins, Y] / rev_m.loc[isins, Y])
    contrib = (weights * CI).dropna().sort_values(ascending=False)
    rows = []
    for rk, isin in enumerate(contrib.head(n).index, 1):
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        rows.append({
            "Rank": rk, "ISIN": isin,
            "Name": isin_name.get(isin, isin),
            "Country": cty[0] if len(cty) else "",
            "Weight (%)": round(float(weights.loc[isin]) * 100, 3),
            "CI (tCO2/M$rev)": round(float(CI.loc[isin]), 1),
            f"Contrib to {label}": round(float(contrib.loc[isin]), 2),
        })
    return pd.DataFrame(rows)


snapshot_years = [2013, 2018, 2024]

print("\n--- Top 10 CARBON-INTENSIVE firms in EUR universe ---")
for Y in snapshot_years:
    print(f"\n   Y={Y}")
    print(top_n_by_CI(Y, 10).to_string(index=False))

print("\n--- Top 10 CONTRIBUTORS to VW WACI (where the benchmark's carbon comes from) ---")
for Y in snapshot_years:
    print(f"\n   Y={Y}")
    print(top_n_by_contrib(Y, vw_weights(Y), 10, label="VW WACI").to_string(index=False))


# =============================================================================
# 18. PLOT — CARBON METRICS TIME SERIES
# =============================================================================
print("\n[17] Plotting baseline carbon metrics ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
C1, C2 = "steelblue", "darkorange"

ax = axes[0]
ax.plot(mv_carbon.index, mv_carbon["WACI"], "o-", color=C1, lw=1.8,
        label=r"Min-Var $P^{(mv)}_{oos}$")
ax.plot(vw_carbon.index, vw_carbon["WACI"], "s--", color=C2, lw=1.8,
        label=r"Val-Wgt $P^{(vw)}$")
ax.set_title("WACI — Weighted-Average Carbon Intensity")
ax.set_xlabel("Year")
ax.set_ylabel(r"tCO$_2$ / M\$ revenue")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(mv_carbon.index, mv_carbon["CF"], "o-", color=C1, lw=1.8,
        label=r"Min-Var $P^{(mv)}_{oos}$")
ax.plot(vw_carbon.index, vw_carbon["CF"], "s--", color=C2, lw=1.8,
        label=r"Val-Wgt $P^{(vw)}$")
ax.set_title("CF — Carbon Footprint (ownership-attributed)")
ax.set_xlabel("Year")
ax.set_ylabel(r"tCO$_2$ / M\$ invested")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part2_carbon_baseline.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: SAAM_Part2_carbon_baseline.{pdf,png}")

print("\n[Section 3.1 complete] — proceed to 3.2: P^(mv)_oos(0.5)")
