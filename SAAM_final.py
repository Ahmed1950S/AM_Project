import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
REGION = "EUR"
START_YEAR = 2013
END_YEAR = 2024
ESTIM_YEARS = 10
MIN_OBS = 36
STALE_THR = 0.30
LOW_FLOOR = 0.50
LW_SHRINK_FLOOR = 0.01
DATA_PATH = "Data_2026/"
TEMPLATE_PATH = "Data_2026/Template_for_Part_I-SAAM.xlsx"
OUT = "Output_2026/"

print("=" * 65)
print("SAAM Part I — Minimum Variance Portfolio (EUR region)")
print("=" * 65)

# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================
print("\n[1] Loading raw data ...")

static = pd.read_excel(DATA_PATH + "Static_2025.xlsx")
ri_m_raw = pd.read_excel(DATA_PATH + "DS_RI_T_USD_M_2025.xlsx", sheet_name="RI")
mv_m_raw = pd.read_excel(DATA_PATH + "DS_MV_T_USD_M_2025.xlsx", sheet_name="MV")
ri_y_raw = pd.read_excel(DATA_PATH + "DS_RI_T_USD_Y_2025.xlsx", sheet_name="RI")
mv_y_raw = pd.read_excel(DATA_PATH + "DS_MV_T_USD_Y_2025.xlsx", sheet_name="MV")
co2_s1_raw = pd.read_excel(DATA_PATH + "DS_CO2_SCOPE_1_Y_2025.xlsx", sheet_name="Scope1")
co2_s2_raw = pd.read_excel(DATA_PATH + "DS_CO2_SCOPE_2_Y_2025.xlsx", sheet_name="Scope2")
rev_raw = pd.read_excel(DATA_PATH + "DS_REV_Y_2025.xlsx", sheet_name="REV")
rf_raw = pd.read_excel(DATA_PATH + "Risk_Free_Rate_2025.xlsx",
                       sheet_name="F-F_Research_Data_Factors")

print(f"   Static: {static.shape}, RI monthly: {ri_m_raw.shape}")

# =============================================================================
# 2. CLEAN DATASTREAM FORMAT
# =============================================================================
print("\n[2] Cleaning Datastream format ...")


def clean_ds(df, valid_isins):
    """Drop error rows, filter to valid ISINs, set ISIN as index, coerce to numeric."""
    df = df.dropna(subset=["ISIN"]).copy()
    df = df[~df["ISIN"].astype(str).str.contains("ER:", na=False)]
    df = df[df["ISIN"].isin(valid_isins)]
    names = df.set_index("ISIN")["NAME"]
    data_cols = [c for c in df.columns if c not in ["NAME", "ISIN"] and c is not None]
    data = df.set_index("ISIN")[data_cols].apply(pd.to_numeric, errors="coerce")
    new_cols = []
    for c in data.columns:
        if hasattr(c, "year") and not isinstance(c, (int, np.integer)):
            new_cols.append(pd.Timestamp(c))
        else:
            new_cols.append(c)
    data.columns = new_cols
    return data, names


eur_isins = set(static.loc[static["Region"] == REGION, "ISIN"])
print(f"   EUR ISINs in static: {len(eur_isins)}")

ri_m, firm_names = clean_ds(ri_m_raw, eur_isins)
mv_m, _ = clean_ds(mv_m_raw, eur_isins)
ri_y, _ = clean_ds(ri_y_raw, eur_isins)
mv_y, _ = clean_ds(mv_y_raw, eur_isins)
co2_s1, _ = clean_ds(co2_s1_raw, eur_isins)
co2_s2, _ = clean_ds(co2_s2_raw, eur_isins)
rev, _ = clean_ds(rev_raw, eur_isins)

isin_name = firm_names.to_dict()
print(f"   EUR firms loaded: {ri_m.shape[0]}")

# =============================================================================
# 3. BUILD DATE LISTS  (precomputed as dicts for O(1) lookup)
# =============================================================================
monthly_all = sorted([c for c in ri_m.columns if isinstance(c, pd.Timestamp)
                      and pd.Timestamp("2000-01-01") <= c <= pd.Timestamp("2025-12-31")])
annual_all = sorted([c for c in ri_y.columns if isinstance(c, (int, np.integer))])

ri_m = ri_m[monthly_all]
mv_m = mv_m[[c for c in monthly_all if c in mv_m.columns]]
ri_y = ri_y[[c for c in annual_all if c in ri_y.columns]]
mv_y = mv_y[[c for c in annual_all if c in mv_y.columns]]
co2_s1 = co2_s1[[c for c in annual_all if c in co2_s1.columns]]
co2_s2 = co2_s2[[c for c in annual_all if c in co2_s2.columns]]
rev = rev[[c for c in annual_all if c in rev.columns]]

# Pre-index monthly_all for O(1) lookups
_monthly_idx = {t: i for i, t in enumerate(monthly_all)}


def months_of(Y):
    return [c for c in monthly_all if c.year == Y]


def estim_window(Y):
    return [c for c in monthly_all
            if (Y - ESTIM_YEARS + 1, 1) <= (c.year, c.month) <= (Y, 12)]


# Precompute as dicts — eliminates repeated list comprehensions
_months_of = {Y: months_of(Y) for Y in range(START_YEAR - 1, END_YEAR + 2)}
_estim_window = {Y: estim_window(Y) for Y in range(START_YEAR, END_YEAR + 1)}

ew2013 = _estim_window[2013]
print(f"   Estimation window Dec 2013: {len(ew2013)} months "
      f"({ew2013[0].strftime('%Y-%m')} to {ew2013[-1].strftime('%Y-%m')})")
assert len(ew2013) == 120, f"Expected 120, got {len(ew2013)}"

# =============================================================================
# 4. RISK-FREE RATE
# =============================================================================
print("\n[3] Processing risk-free rate ...")

rf_raw.columns = ["YYYYMM", "RF_pct"]
rf_raw = rf_raw.dropna(subset=["YYYYMM"])
rf_raw["date"] = pd.to_datetime(rf_raw["YYYYMM"].astype(int).astype(str), format="%Y%m")
rf_raw["date"] = rf_raw["date"] + pd.offsets.MonthEnd(0)
rf_mon = rf_raw.set_index("date")["RF_pct"] / 100
rf_mon = rf_mon.squeeze()
rf_mon.name = "RF"

print(f"   RF range: {rf_mon.index.min().strftime('%Y-%m')} to "
      f"{rf_mon.index.max().strftime('%Y-%m')}")
print(f"   RF sample 2014-01: {rf_mon.loc['2014-01'].values[0]:.6f} (monthly)")

_rf_ann_check = rf_mon.loc["2014-01-01":"2025-12-31"].mean() * 12
assert 0.005 < _rf_ann_check < 0.05, (
    f"Annualised avg rf = {_rf_ann_check:.4%} — outside plausible range."
)
print(f"   Sanity: annualised avg rf over 2014-2025 = {_rf_ann_check*100:.4f}%")

# =============================================================================
# 5. PRICE CLEANING
# =============================================================================
print("\n[4] Cleaning prices ...")

n_low = ((ri_m > 0) & (ri_m < LOW_FLOOR)).sum().sum()
ri_m[ri_m < LOW_FLOOR] = np.nan
print(f"   Prices < {LOW_FLOOR} set to NaN: {n_low}")


def forward_fill_internal(row):
    fv = row.first_valid_index()
    lv = row.last_valid_index()
    if fv is None or lv is None:
        return row
    cols = list(row.index)
    start = cols.index(fv)
    end = cols.index(lv)
    row.iloc[start:end + 1] = row.iloc[start:end + 1].ffill()
    return row


ri_before = ri_m.isna().sum().sum()
ri_m = ri_m.apply(forward_fill_internal, axis=1)
n_filled = ri_before - ri_m.isna().sum().sum()
print(f"   Forward-filled internal gaps: {n_filled} observations")

cutoff = pd.Timestamp("2025-12-31")
last_valid = {}
for isin in ri_m.index:
    lv = ri_m.loc[isin].last_valid_index()
    if lv is not None and isinstance(lv, pd.Timestamp) and lv < cutoff:
        last_valid[isin] = lv
print(f"   Firms with early last price (potential delistings): {len(last_valid)}")

ret_m = ri_m.pct_change(axis=1)

for isin, ddate in last_valid.items():
    if ddate not in _monthly_idx:
        continue
    idx = _monthly_idx[ddate]
    if idx + 1 < len(monthly_all):
        ret_m.at[isin, monthly_all[idx + 1]] = -1.0
        for k in range(idx + 2, len(monthly_all)):
            ret_m.at[isin, monthly_all[k]] = np.nan

ret_m = ret_m.iloc[:, 1:]

n_delist = (ret_m == -1.0).sum().sum()
print(f"   -100% delisting returns applied: {n_delist}")

co2_s1 = co2_s1.ffill(axis=1)
co2_s2 = co2_s2.ffill(axis=1)
rev = rev.ffill(axis=1)

# =============================================================================
# 6. INVESTMENT SET (UNIVERSE) CONSTRUCTION
# =============================================================================
print("\n[5] Building investment sets ...")


def get_universe(Y):
    """
    Build investment set for allocation at end of year Y.
    Filters:
      1. Valid price at end of year Y (monthly RI)
      2. Sufficient return observations in 10-year window
      3. No stale prices (zero-return fraction < threshold)
      4. CO2 Scope 1 + Scope 2 both available at end of year Y
    """
    dec_cols = [c for c in monthly_all if c.year == Y and c.month == 12]
    if not dec_cols:
        return []
    dec_Y = dec_cols[0]

    win = _estim_window[Y]
    win_ret = [c for c in win if c in ret_m.columns]
    R_win = ret_m.reindex(columns=win_ret)

    out = []
    for isin in ri_m.index:
        if pd.isna(ri_m.at[isin, dec_Y]):
            continue
        if isin not in R_win.index:
            continue
        row = R_win.loc[isin]
        n_valid = row.notna().sum()
        if n_valid < MIN_OBS:
            continue
        n_zero = ((row == 0) | (row.abs() < 1e-10)).sum()
        if (n_zero / n_valid) > STALE_THR:
            continue
        has_s1 = Y in co2_s1.columns and pd.notna(co2_s1.at[isin, Y]) if isin in co2_s1.index else False
        has_s2 = Y in co2_s2.columns and pd.notna(co2_s2.at[isin, Y]) if isin in co2_s2.index else False
        if not (has_s1 and has_s2):
            continue
        out.append(isin)

    return out


universe = {}
for Y in range(START_YEAR, END_YEAR + 1):
    universe[Y] = get_universe(Y)
    print(f"   {Y}: {len(universe[Y]):3d} firms")

years_part2 = list(range(START_YEAR, END_YEAR + 1))

# =============================================================================
# 7. COVARIANCE ESTIMATION  (+  result cache)
# =============================================================================
# Cache: _cov_cache[Y] = (mu, Sigma) — computed once, reused in all optimizers
# and verification loops (was ~7 calls/year → 1)
_cov_cache = {}


def estimate_cov(isins, win_cols):
    """
    Pairwise-complete covariance + Ledoit-Wolf shrinkage to constant-correlation
    target (OAS formula). See lecture 5 and LW (2004) / Chen et al. (2010).
    """
    win_in = [c for c in win_cols if c in ret_m.columns]
    R = ret_m.loc[isins, win_in]
    N = len(isins)

    mu = R.mean(axis=1).values

    R_vals = R.values
    not_nan = ~np.isnan(R_vals)
    mu_full = np.nanmean(R_vals, axis=1)
    R_demeaned = R_vals - mu_full[:, None]
    R_demeaned_zero = np.where(not_nan, R_demeaned, 0.0)
    n_obs = not_nan.sum(axis=1)
    var = np.where(n_obs > 1,
                   np.sum(R_demeaned_zero ** 2, axis=1) / n_obs,
                   0.0)

    R_zero = np.where(not_nan, R_vals, 0.0)
    not_nan_f = not_nan.astype(np.float64)

    count_ij = not_nan_f @ not_nan_f.T
    sum_i_ij = R_zero @ not_nan_f.T
    sum_j_ij = not_nan_f @ R_zero.T

    safe_count = np.maximum(count_ij, 1)
    mean_i_ij = sum_i_ij / safe_count
    mean_j_ij = sum_j_ij / safe_count

    cross_ij = R_zero @ R_zero.T
    cov_ij = (cross_ij / safe_count) - mean_i_ij * mean_j_ij

    sum_sq_i_ij = (R_zero ** 2) @ not_nan_f.T
    var_i_ij = np.maximum(sum_sq_i_ij / safe_count - mean_i_ij ** 2, 0)

    sum_sq_j_ij = not_nan_f @ (R_zero ** 2).T
    var_j_ij = np.maximum(sum_sq_j_ij / safe_count - mean_j_ij ** 2, 0)

    denom = np.sqrt(var_i_ij * var_j_ij)
    corr_ij = np.where(denom > 1e-20, cov_ij / denom, 0.0)

    std_own = np.sqrt(var)
    Sig = corr_ij * np.outer(std_own, std_own)
    np.fill_diagonal(Sig, var)
    Sig = (Sig + Sig.T) / 2

    # Ledoit-Wolf (OAS) shrinkage
    std_diag = np.sqrt(np.diag(Sig))
    std_diag_safe = np.where(std_diag > 1e-20, std_diag, 1e-20)
    corr_mat = np.clip(Sig / np.outer(std_diag_safe, std_diag_safe), -1.0, 1.0)
    np.fill_diagonal(corr_mat, 1.0)
    rho_bar = (corr_mat.sum() - N) / (N * (N - 1))
    F = rho_bar * np.outer(std_diag, std_diag)
    np.fill_diagonal(F, var)

    tr_Sig = np.trace(Sig)
    tr_Sig2 = np.trace(Sig @ Sig)
    T_eff = max(np.median(count_ij[np.triu_indices(N, k=1)]), 2)
    numerator = (1.0 - 2.0 / N) * tr_Sig2 + tr_Sig ** 2
    denominator = (T_eff + 1.0 - 2.0 / N) * (tr_Sig2 - tr_Sig ** 2 / N)
    delta = 0.5 if abs(denominator) < 1e-20 else max(min(numerator / denominator, 1.0), LW_SHRINK_FLOOR)

    Sig_shrunk = delta * F + (1.0 - delta) * Sig
    eigvals, eigvecs = np.linalg.eigh(Sig_shrunk)
    if eigvals[0] < 1e-10:
        eigvals = np.maximum(eigvals, 1e-10)
        Sig_shrunk = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Sig_shrunk = (Sig_shrunk + Sig_shrunk.T) / 2

    return mu, Sig_shrunk


def get_cov(Y):
    """Cached covariance for year Y (computed once, reused across all sections)."""
    if Y not in _cov_cache:
        _cov_cache[Y] = estimate_cov(universe[Y], _estim_window[Y])
    return _cov_cache[Y]


# =============================================================================
# 8. UTILITY FUNCTIONS (cached)
# =============================================================================

def fill_oos_returns(eligible, next_months):
    """
    Fill missing OOS returns with delisting detection.
    Delisted → -100%, then NaN; leading NaN (inactive) → 0%.
    """
    R_oos = ret_m.loc[eligible].reindex(columns=next_months).copy()

    for isin in eligible:
        delisted = False
        for k, t in enumerate(next_months):
            if delisted:
                R_oos.at[isin, t] = np.nan
                continue
            if pd.isna(R_oos.at[isin, t]):
                t_idx = _monthly_idx.get(t, None)
                prev_t = monthly_all[t_idx - 1] if t_idx and t_idx > 0 else None
                had_price_prev = (prev_t is not None
                                  and prev_t in ri_m.columns
                                  and isin in ri_m.index
                                  and pd.notna(ri_m.at[isin, prev_t]))
                if had_price_prev:
                    R_oos.at[isin, t] = -1.0
                    delisted = True
                else:
                    R_oos.at[isin, t] = 0.0

    return R_oos.fillna(0.0)


# Cache: _oos_cache[Y] = R_oos for universe[Y] × months_of(Y+1)
# Called once per year instead of 4x (MV, MV05, VW05, VWNZ)
_oos_cache = {}


def get_oos_returns(Y):
    if Y not in _oos_cache:
        _oos_cache[Y] = fill_oos_returns(universe[Y], _months_of[Y + 1])
    return _oos_cache[Y]


# Cache: value-weighted benchmark weights
_vww_cache = {}


def vw_weights(Y):
    if Y not in _vww_cache:
        isins = universe[Y]
        cap = mv_y.loc[isins, Y].fillna(0.0)
        s = cap.sum()
        _vww_cache[Y] = (cap / s if s > 0
                         else pd.Series(np.ones(len(isins)) / len(isins), index=isins))
    return _vww_cache[Y]


# =============================================================================
# 9. MIN-VARIANCE OPTIMISATION
# =============================================================================
print("\n[6] Rolling min-variance optimisation ...")

_drifted_w = {}


def min_var_weights(Sigma, isins, Y):
    """
    min α'Σα  s.t. α'e = 1, α >= 0
    Warm-starts from drifted weights of the previous year.
    """
    N = Sigma.shape[0]
    if Y in _drifted_w:
        prev = _drifted_w[Y]
        w0 = np.array([prev.get(i, 0.0) for i in isins])
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(N) / N
    else:
        w0 = np.ones(N) / N

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=w0,
        jac=lambda w: 2.0 * (Sigma @ w),
        method="SLSQP",
        bounds=[(0.0, None)] * N,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not res.success:
        print(f"   WARNING: optimizer did not converge for Y={Y}: {res.message}")
    return res.x


mv_w_dict = {}
mv_ret = {}

for Y in range(START_YEAR, END_YEAR + 1):
    eligible = universe[Y]
    N = len(eligible)
    if N == 0:
        print(f"   Y={Y}: 0 firms → skip")
        continue

    mu, Sig = get_cov(Y)          # ← cached: subsequent sections reuse this
    w = min_var_weights(Sig, eligible, Y)
    mv_w_dict[Y] = pd.Series(w, index=eligible)

    ea_vol = np.sqrt(float(w @ Sig @ w) * 12) * 100

    next_months = _months_of[Y + 1]
    R_next = get_oos_returns(Y)   # ← cached OOS returns
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    mv_ret[Y + 1] = pd.Series(port_ret, index=next_months)
    _drifted_w[Y + 1] = dict(zip(eligible, ww))

    n_nonzero = (w > 1e-6).sum()
    print(f"   Y={Y}: {N:3d} firms, non-zero wgts: {n_nonzero:3d}, "
          f"ex-ante ann.σ: {ea_vol:.2f}%")

rp_mv = pd.concat(mv_ret).droplevel(0).sort_index()
rp_mv.index = pd.DatetimeIndex(rp_mv.index)

# =============================================================================
# 10. VALUE-WEIGHTED BENCHMARK
# =============================================================================
print("\n[7] Value-weighted benchmark ...")

vw_ret = {}
for Y in range(START_YEAR, END_YEAR + 1):
    eligible = universe[Y]
    next_months = _months_of[Y + 1]
    R_oos_vw = get_oos_returns(Y)     # ← reuses cached OOS returns
    port = []

    for t in next_months:
        idx = _monthly_idx.get(t)
        if idx is None or idx == 0:
            port.append(np.nan)
            continue
        prev_t = monthly_all[idx - 1]
        cap = (mv_m.loc[eligible, prev_t].fillna(0)
               if prev_t in mv_m.columns else pd.Series(0.0, index=eligible))
        tot = cap.sum()
        if tot <= 0:
            port.append(0.0)
            continue
        r_t = R_oos_vw[t].values if t in R_oos_vw.columns else np.zeros(len(eligible))
        port.append(float((cap / tot).values @ r_t))

    vw_ret[Y + 1] = pd.Series(port, index=next_months)

rp_vw = pd.concat(vw_ret).droplevel(0).sort_index()
rp_vw.index = pd.DatetimeIndex(rp_vw.index)

# =============================================================================
# 11. PERFORMANCE STATISTICS
# =============================================================================
print("\n[8] Performance statistics ...")


def compute_perf(rp, rf_s, label):
    """
    Ann. return: arithmetic 12×mean (Lecture 5, slide 12),
                 geometric (1+R_cum)^(12/T)−1 (slide 13).
    Sharpe uses arithmetic: SR^(y) = √12 × SR^(m).
    """
    rp = rp.dropna()
    rf = rf_s.reindex(rp.index).ffill().fillna(0)
    T = len(rp)
    mu_arith = 12 * rp.mean()
    mu_geom = (1 + rp).prod() ** (12 / T) - 1
    sig_ann = rp.std() * np.sqrt(12)
    rf_ann = 12 * rf.mean()
    SR = (mu_arith - rf_ann) / sig_ann
    cum = (1 + rp).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Portfolio": label,
        "Ann. Return Arith. (%)": round(mu_arith * 100, 2),
        "Ann. Return Geom. (%)": round(mu_geom * 100, 2),
        "Ann. Vol (%)": round(sig_ann * 100, 2),
        "Sharpe": round(SR, 3),
        "Min Mo. (%)": round(rp.min() * 100, 2),
        "Max Mo. (%)": round(rp.max() * 100, 2),
        "Max DD (%)": round(mdd * 100, 2),
    }


stats_df = pd.DataFrame([
    compute_perf(rp_vw, rf_mon, "Val-Wgt P^(vw)"),
    compute_perf(rp_mv, rf_mon, "Min-Var P_oos^(mv)"),
]).set_index("Portfolio")

print("\n", stats_df.to_string())

# =============================================================================
# 12. VERIFICATION CHECKS
# =============================================================================
print("\n[9] Verification checks ...")

for Y, w in mv_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6, f"Y={Y}: weights sum to {w.sum()}"
print("   ✓ All weight vectors sum to 1")

for Y, w in mv_w_dict.items():
    assert (w >= -1e-8).all(), f"Y={Y}: negative weights found"
print("   ✓ All weights non-negative")

expected_months = sum(len(_months_of[Y + 1]) for Y in range(START_YEAR, END_YEAR + 1))
actual_mv = len(rp_mv.dropna())
actual_vw = len(rp_vw.dropna())
print(f"   Min-Var returns: {actual_mv} months (expected ~{expected_months})")
print(f"   Val-Wgt returns: {actual_vw} months")

assert rp_mv.isna().sum() == 0, "NaN in min-var returns"
assert rp_vw.isna().sum() == 0, "NaN in VW returns"
print("   ✓ No NaN in return series")

# =============================================================================
# 13. TOP HOLDINGS
# =============================================================================
print("\n[10] Top 10 holdings (Min-Var):")

for Y in [2013, 2018, 2024]:
    if Y not in mv_w_dict:
        continue
    w = mv_w_dict[Y].sort_values(ascending=False)
    n_nz = (w > 1e-6).sum()
    print(f"\n   Dec {Y} (non-zero: {n_nz}/{len(w)}):")
    for rk, (isin, wt) in enumerate(w.head(10).items(), 1):
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        cty = cty[0] if len(cty) else ""
        print(f"   {rk:2d}. {isin_name.get(isin, isin):<40s} {cty:>3s}  {wt * 100:6.2f}%")

# =============================================================================
# 14. FIGURES
# =============================================================================
print("\n[11] Generating figures ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SAAM Part I — EUR: Min-Var vs Value-Weighted (2014–2025)",
             fontsize=13, fontweight="bold")
C1, C2 = "steelblue", "darkorange"
fmt = mdates.DateFormatter("%Y")
loc = mdates.YearLocator(2)

ax = axes[0, 0]
cm = (1 + rp_mv.dropna()).cumprod()
cv = (1 + rp_vw.dropna()).cumprod()
ax.plot(cm.index, cm.values, color=C1, lw=1.8, label=r"Min-Var $P_{oos}^{(mv)}$")
ax.plot(cv.index, cv.values, color=C2, lw=1.8, ls="--", label=r"Val-Wgt $P^{(vw)}$")
ax.set_title("Cumulative Return (base=1, Jan 2014)")
ax.set_ylabel("Cumulative return")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[0, 1]
rv_mv = rp_mv.rolling(12).std() * np.sqrt(12) * 100
rv_vw = rp_vw.rolling(12).std() * np.sqrt(12) * 100
ax.plot(rp_mv.index, rv_mv.values, color=C1, lw=1.6, label="Min-Var")
ax.plot(rp_vw.index, rv_vw.values, color=C2, lw=1.6, ls="--", label="Val-Wgt")
ax.set_title("Rolling 12m Annualised Volatility (%)")
ax.set_ylabel("Vol (%)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[1, 0]


def drawdown_series(rp):
    c = (1 + rp.dropna()).cumprod()
    return (c - c.cummax()) / c.cummax() * 100


dd_mv = drawdown_series(rp_mv)
dd_vw = drawdown_series(rp_vw)
ax.fill_between(dd_mv.index, dd_mv.values, 0, alpha=0.45, color=C1, label="Min-Var")
ax.fill_between(dd_vw.index, dd_vw.values, 0, alpha=0.30, color=C2, label="Val-Wgt")
ax.set_title("Drawdown from Peak (%)")
ax.set_ylabel("Drawdown (%)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[1, 1]
yrs = sorted(universe.keys())
ax.bar(yrs, [len(universe[y]) for y in yrs], color=C1, alpha=0.8,
       edgecolor="white", width=0.6)
ax.set_title("EUR Investment Set Size by Year")
ax.set_ylabel("Eligible firms"); ax.set_xticks(yrs)
ax.set_xticklabels(yrs, rotation=45); ax.grid(axis="y", alpha=0.3)
for y in yrs:
    ax.text(y, len(universe[y]) + 5, str(len(universe[y])), ha="center", fontsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part1_EUR_figures.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Figures saved.")

# =============================================================================
# 15. EXCEL EXPORT — Official Template Format
# =============================================================================
print("\n[12] Exporting Excel (official template format) ...")
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XlImage


def template_stats(rp, rf_s):
    rp = rp.dropna()
    rf = rf_s.reindex(rp.index).ffill().fillna(0)
    T = len(rp)
    mu_arith = 12 * rp.mean()
    mu_geom = (1 + rp).prod() ** (12 / T) - 1
    sig_ann = rp.std() * np.sqrt(12)
    rf_ann = 12 * rf.mean()
    SR = (mu_arith - rf_ann) / sig_ann
    return {"ann_avg_ret": mu_arith, "ann_vol": sig_ann, "ann_cum_ret": mu_geom,
            "sharpe": SR, "min_mo": rp.min(), "max_mo": rp.max()}


vw_stats = template_stats(rp_vw, rf_mon)
mv_stats = template_stats(rp_mv, rf_mon)

wb = load_workbook(TEMPLATE_PATH)
ws = wb["Sheet1"]

stat_keys = ["ann_avg_ret", "ann_vol", "ann_cum_ret", "sharpe", "min_mo", "max_mo"]
for i, key in enumerate(stat_keys):
    ws.cell(row=3 + i, column=2, value=round(vw_stats[key], 8))
    ws.cell(row=3 + i, column=3, value=round(mv_stats[key], 8))

cum_plot_path = f"{OUT}SAAM_Part1_EUR_cumulative.png"
fig_cum, ax_cum = plt.subplots(figsize=(7, 4))
cm = (1 + rp_mv.dropna()).cumprod()
cv = (1 + rp_vw.dropna()).cumprod()
ax_cum.plot(cm.index, cm.values, color="steelblue", lw=1.8, label=r"Min-Var $P_{oos}^{(mv)}$")
ax_cum.plot(cv.index, cv.values, color="darkorange", lw=1.8, ls="--", label=r"Val-Wgt $P^{(vw)}$")
ax_cum.set_title("Cumulative Return (base=1, Jan 2014)")
ax_cum.set_ylabel("Cumulative return"); ax_cum.legend(fontsize=9); ax_cum.grid(alpha=0.3)
ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_cum.xaxis.set_major_locator(mdates.YearLocator(2))
fig_cum.tight_layout()
fig_cum.savefig(cum_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig_cum)

img = XlImage(cum_plot_path)
img.width = 500; img.height = 280
ws.add_image(img, "B9")

vw_by_ym = {(d.year, d.month): v for d, v in rp_vw.items()}
mv_by_ym = {(d.year, d.month): v for d, v in rp_mv.items()}

for row_idx in range(3, 3 + 144):
    date_cell = ws.cell(row=row_idx, column=5).value
    if date_cell is None:
        continue
    dt = pd.Timestamp(date_cell)
    ym = (dt.year, dt.month)
    vw_val = vw_by_ym.get(ym, np.nan)
    mv_val = mv_by_ym.get(ym, np.nan)
    ws.cell(row=row_idx, column=6, value=round(float(vw_val), 8) if not np.isnan(vw_val) else None)
    ws.cell(row=row_idx, column=7, value=round(float(mv_val), 8) if not np.isnan(mv_val) else None)

xlsx_out = f"{OUT}SAAM_Part1_EUR_template.xlsx"
wb.save(xlsx_out)
print(f"   Template saved: {xlsx_out}")

xlsx_ext = f"{OUT}SAAM_Part1_EUR_results.xlsx"
with pd.ExcelWriter(xlsx_ext, engine="openpyxl") as writer:
    stats_df.to_excel(writer, sheet_name="Summary_Stats")
    ro = pd.DataFrame({"Min-Var": rp_mv, "Value-Weighted": rp_vw})
    ro.index = ro.index.strftime("%Y-%m")
    ro.to_excel(writer, sheet_name="Monthly_Returns")
    wd = pd.DataFrame(mv_w_dict).T.fillna(0)
    wd.index.name = "Year"
    wd.rename(columns=isin_name, inplace=True)
    wd.to_excel(writer, sheet_name="MV_Weights")
    rows = []
    for Y in sorted(mv_w_dict):
        for rk, (i, wt) in enumerate(mv_w_dict[Y].sort_values(ascending=False).head(10).items(), 1):
            cty = static.loc[static["ISIN"] == i, "Country"].values
            rows.append({"Year": Y, "Rank": rk, "ISIN": i,
                         "Name": isin_name.get(i, i),
                         "Country": cty[0] if len(cty) else "",
                         "Weight (%)": round(wt * 100, 3)})
    pd.DataFrame(rows).to_excel(writer, sheet_name="Top10_Holdings", index=False)

print(f"   Extended results saved: {xlsx_ext}")
print("\n" + "=" * 65)
print("DONE — outputs in", OUT)


# =============================================================================
# PART II — Carbon-Aware Portfolio Allocation
# =============================================================================
from scipy.optimize import linprog

print("\n" + "=" * 65)
print("SAAM Part II — Carbon-Aware Portfolios (EUR / Scope 1+2)")
print("=" * 65)

# =============================================================================
# 16. CARBON DATA PREPARATION
# =============================================================================
print("\n[14] Preparing carbon data ...")

co2_tot = co2_s1 + co2_s2
rev_m = rev / 1000.0          # thousands → millions USD

with np.errstate(divide="ignore", invalid="ignore"):
    CI_firm = co2_tot / rev_m

diag = []
for Y in years_part2:
    isins = universe[Y]
    diag.append({
        "Y": Y,
        "Universe": len(isins),
        "Emissions": int(co2_tot.loc[isins, Y].notna().sum()) if Y in co2_tot.columns else 0,
        "Revenue": int(rev_m.loc[isins, Y].notna().sum()) if Y in rev_m.columns else 0,
        "Cap_yr": int(mv_y.loc[isins, Y].notna().sum()) if Y in mv_y.columns else 0,
    })
diag_df = pd.DataFrame(diag).set_index("Y")
print(diag_df.to_string())

miss_E = int((diag_df["Universe"] - diag_df["Emissions"]).sum())
assert miss_E == 0, f"Emissions missing in universe: {miss_E}"
print(f"   ✓ Emissions: 100% coverage in universe")

miss_R = int((diag_df["Universe"] - diag_df["Revenue"]).sum())
miss_C = int((diag_df["Universe"] - diag_df["Cap_yr"]).sum())
print(f"   Coverage gaps: Revenue missing {miss_R} firm-yrs, Cap missing {miss_C}")

# ─── Analysis helpers ─────────────────────────────────────────────────────────


def _firm_ci(isin, Y):
    """Carbon intensity (tCO2/M$rev) for a firm-year; NaN if missing."""
    if (isin in co2_tot.index and Y in co2_tot.columns
            and isin in rev_m.index and Y in rev_m.columns
            and pd.notna(rev_m.loc[isin, Y]) and rev_m.loc[isin, Y] > 0):
        return float(co2_tot.loc[isin, Y] / rev_m.loc[isin, Y])
    return np.nan


def most_avoided(w_port: pd.Series, w_bench: pd.Series,
                  Y: int, n: int = 10, title: str = "") -> pd.DataFrame:
    """
    Firms most underweighted in w_port vs w_bench.
    'Most avoided' = had weight in benchmark, zeroed or cut heavily in portfolio.
    Sorted by active weight (most negative first).
    """
    all_isins = w_bench.index[w_bench > 1e-6]   # only firms that exist in benchmark
    rows = []
    for isin in all_isins:
        wb = float(w_bench.get(isin, 0.0))
        wp = float(w_port.get(isin, 0.0))
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        rows.append({
            "Name": isin_name.get(isin, isin),
            "Country": cty[0] if len(cty) else "",
            "Bench (%)": round(wb * 100, 3),
            "Port (%)": round(wp * 100, 3),
            "Active (pp)": round((wp - wb) * 100, 3),
            "CI (tCO2/M$rev)": round(_firm_ci(isin, Y), 1)
            if not np.isnan(_firm_ci(isin, Y)) else np.nan,
        })

    df = (pd.DataFrame(rows)
          .sort_values("Active (pp)")
          .head(n)
          .reset_index(drop=True))

    if title:
        print(f"\n   MOST AVOIDED POSITIONS — {title}  (Y={Y}, top {n}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 17. CARBON METRICS — BASELINE PORTFOLIOS
# =============================================================================
print("\n[15] Computing baseline portfolio carbon metrics ...")


def carbon_metrics(weights, Y):
    """WACI (tCO2/M$rev) and CF (tCO2/M$inv) for a portfolio at year-end Y."""
    isins = list(weights.index)
    w = weights.values.astype(float)
    E = co2_tot.loc[isins, Y].values.astype(float)
    R = rev_m.loc[isins, Y].values.astype(float)
    C = mv_y.loc[isins, Y].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        CI = np.where((R > 0) & np.isfinite(R), E / R, np.nan)
    valid_W = np.isfinite(CI)
    WACI = float(np.nansum(w * np.where(valid_W, CI, 0.0)))
    cov_W = float(w[valid_W].sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        ED = np.where((C > 0) & np.isfinite(C), E / C, np.nan)
    valid_F = np.isfinite(ED)
    CF = float(np.nansum(w * np.where(valid_F, ED, 0.0)))
    cov_F = float(w[valid_F].sum())

    return {"WACI": WACI, "CF": CF, "cov_W": cov_W, "cov_F": cov_F}


mv_carbon_rows, vw_carbon_rows = [], []
for Y in years_part2:
    mv_carbon_rows.append({"Y": Y, **carbon_metrics(mv_w_dict[Y], Y)})
    vw_carbon_rows.append({"Y": Y, **carbon_metrics(vw_weights(Y), Y)})

mv_carbon = pd.DataFrame(mv_carbon_rows).set_index("Y")
vw_carbon = pd.DataFrame(vw_carbon_rows).set_index("Y")

print("\n   Min-Variance portfolio P^(mv)_oos:")
print(mv_carbon.round(2).to_string())
print("\n   Value-Weighted portfolio P^(vw):")
print(vw_carbon.round(2).to_string())

# =============================================================================
# 18. VERIFICATION OF CARBON METRICS
# =============================================================================
print("\n[15b] Verification of carbon metrics ...")

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
print("      ✓ done")

print(f"   B. Coverage — MV WACI: min={mv_carbon['cov_W'].min():.3f}, mean={mv_carbon['cov_W'].mean():.3f}")
print(f"      MV CF: min={mv_carbon['cov_F'].min():.3f}   VW WACI: min={vw_carbon['cov_W'].min():.3f}")

print(f"   C. VW WACI mean = {vw_carbon['WACI'].mean():.0f} tCO2/M$rev  |  "
      f"VW CF mean = {vw_carbon['CF'].mean():.0f} tCO2/M$inv")

mv_lower = (mv_carbon["WACI"] < vw_carbon["WACI"]).sum()
print(f"   D. MV WACI < VW WACI in {mv_lower} of {len(years_part2)} years "
      f"(expect MV < VW — low-vol tilts to low-carbon sectors)")

# =============================================================================
# 19. TOP CARBON CONTRIBUTORS
# =============================================================================
print("\n[16] Top carbon contributors ...")


def top_n_by_CI(Y, n=10):
    isins = universe[Y]
    CI = (co2_tot.loc[isins, Y] / rev_m.loc[isins, Y]).dropna().sort_values(ascending=False)
    rows = []
    for rk, isin in enumerate(CI.head(n).index, 1):
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        rows.append({"Rank": rk, "ISIN": isin, "Name": isin_name.get(isin, isin),
                     "Country": cty[0] if len(cty) else "",
                     "CI (tCO2/M$rev)": round(float(CI.loc[isin]), 1),
                     "VW weight (%)": round(float(vw_weights(Y).get(isin, 0)) * 100, 3)})
    return pd.DataFrame(rows)


def top_n_by_contrib(Y, weights, n=10, label="WACI"):
    isins = list(weights.index)
    CI = (co2_tot.loc[isins, Y] / rev_m.loc[isins, Y])
    contrib = (weights * CI).dropna().sort_values(ascending=False)
    rows = []
    for rk, isin in enumerate(contrib.head(n).index, 1):
        cty = static.loc[static["ISIN"] == isin, "Country"].values
        rows.append({"Rank": rk, "ISIN": isin, "Name": isin_name.get(isin, isin),
                     "Country": cty[0] if len(cty) else "",
                     "Weight (%)": round(float(weights.loc[isin]) * 100, 3),
                     "CI (tCO2/M$rev)": round(float(CI.loc[isin]), 1),
                     f"Contrib to {label}": round(float(contrib.loc[isin]), 2)})
    return pd.DataFrame(rows)


snapshot_years = [2013, 2018, 2024]

print("\n--- Top 10 CARBON-INTENSIVE firms in EUR universe ---")
for Y in snapshot_years:
    print(f"\n   Y={Y}")
    print(top_n_by_CI(Y, 10).to_string(index=False))

print("\n--- Top 10 CONTRIBUTORS to VW WACI ---")
for Y in snapshot_years:
    print(f"\n   Y={Y}")
    print(top_n_by_contrib(Y, vw_weights(Y), 10, label="VW WACI").to_string(index=False))

# =============================================================================
# 20. PLOT — CARBON METRICS TIME SERIES
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
ax.set_xlabel("Year"); ax.set_ylabel(r"tCO$_2$ / M\$ revenue")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
ax.plot(mv_carbon.index, mv_carbon["CF"], "o-", color=C1, lw=1.8,
        label=r"Min-Var $P^{(mv)}_{oos}$")
ax.plot(vw_carbon.index, vw_carbon["CF"], "s--", color=C2, lw=1.8,
        label=r"Val-Wgt $P^{(vw)}$")
ax.set_title("CF — Carbon Footprint (ownership-attributed)")
ax.set_xlabel("Year"); ax.set_ylabel(r"tCO$_2$ / M\$ invested")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part2_carbon_baseline.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: SAAM_Part2_carbon_baseline.{pdf,png}")

print("\n[Section 3.1 complete]")

# =============================================================================
# SECTION 3.2 — Active Investor: MV with CF ≤ 0.5 × CF(MV)
# =============================================================================
print("\n" + "=" * 65)
print("Section 3.2 — Active Investor: MV with CF ≤ 0.5 × CF(MV)")
print("=" * 65)

# =============================================================================
# 21. CF TARGETS
# =============================================================================
cf_target_mv = (mv_carbon["CF"] * 0.5).rename("target_CF").copy()
print("\n[18] CF targets for P^(mv)_oos(0.5):")
print(cf_target_mv.round(3).to_string())

# =============================================================================
# 22. PER-FIRM CF VECTOR
# =============================================================================


def cf_vector(isins, Y):
    """c_i = E_i,Y / Cap_i,Y  (inf where cap missing — forces α_i = 0)."""
    E = co2_tot.loc[isins, Y].values.astype(float)
    C = mv_y.loc[isins, Y].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where((C > 0) & np.isfinite(C) & np.isfinite(E), E / C, np.inf)
    return c, np.isfinite(c)


# =============================================================================
# 23. FEASIBILITY CHECK (cached — reused in 3.3 and 4.1)
# =============================================================================
_mfcf_cache = {}


def get_mfcf(Y):
    """Cached minimum achievable CF for the year-Y universe (LP)."""
    if Y not in _mfcf_cache:
        isins = universe[Y]
        c, valid = cf_vector(isins, Y)
        N = len(isins)
        if not valid.any():
            _mfcf_cache[Y] = np.nan
        else:
            c_lp = np.where(valid, c, 1e20)
            res = linprog(c=c_lp, A_eq=np.ones((1, N)), b_eq=[1.0],
                          bounds=[(0.0, 1.0)] * N, method="highs")
            _mfcf_cache[Y] = float(c_lp @ res.x) if res.success else np.nan
    return _mfcf_cache[Y]


def feasibility_check(cf_targets, label):
    adj = {}
    infeas = []
    for Y in years_part2:
        cf_min = get_mfcf(Y)
        target_raw = float(cf_targets.loc[Y])
        if cf_min > target_raw:
            target_eff = cf_min * 1.001
            infeas.append({"Y": Y, "target_raw": target_raw,
                           "cf_min": cf_min, "target_eff": target_eff})
            adj[Y] = target_eff
        else:
            adj[Y] = target_raw
    if infeas:
        print(f"\n   ⚠ [{label}] Infeasibility — targets relaxed:")
        print(pd.DataFrame(infeas).round(3).to_string(index=False))
    else:
        print(f"\n   ✓ [{label}] All targets feasible")
    return adj


adj_targets = feasibility_check(cf_target_mv, "MV05")

# =============================================================================
# 24. CONSTRAINED MIN-VARIANCE OPTIMIZER
# =============================================================================


def min_var_cf_constrained(Sigma, c, target, isins, Y, prev_drift=None):
    """min α'Σα  s.t. α'e=1, c'α≤target, α≥0, α_i=0 for invalid cap."""
    N = Sigma.shape[0]
    valid = np.isfinite(c)
    invalid = ~valid

    if prev_drift is not None:
        w0 = np.array([prev_drift.get(i, 0.0) for i in isins])
        w0[invalid] = 0.0
        s = w0.sum()
        w0 = w0 / s if s > 0 else np.where(valid, 1.0, 0.0) / max(valid.sum(), 1)
    else:
        w0 = np.where(valid, 1.0, 0.0) / max(valid.sum(), 1)

    bounds = [(0.0, 0.0) if invalid[i] else (0.0, 1.0) for i in range(N)]
    c_safe = np.where(valid, c, 0.0)
    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0,        "jac": lambda w: np.ones(N)},
        {"type": "ineq", "fun": lambda w: target - c_safe @ w,  "jac": lambda w: -c_safe},
    ]
    return minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=w0, jac=lambda w: 2.0 * (Sigma @ w),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )


# =============================================================================
# 25. ROLLING LOOP — P^(mv)_oos(0.5)
# =============================================================================
print("\n[19] Rolling MV(0.5) optimization ...")

mv05_w_dict = {}
mv05_ret = {}
_mv05_drift = {}
solver_log = []

for Y in years_part2:
    eligible = universe[Y]
    target = adj_targets[Y]
    _, Sig = get_cov(Y)      # ← zero-cost cache hit
    c, valid = cf_vector(eligible, Y)

    res = min_var_cf_constrained(Sig, c, target, eligible, Y,
                                  prev_drift=_mv05_drift.get(Y))
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w /= w.sum()

    cf_real = float(np.where(valid, c, 0.0) @ w)
    violates = cf_real > target * (1 + 1e-6)
    if violates:
        print(f"   ⚠ Y={Y}: CF constraint violated ({cf_real:.3f} > {target:.3f})")

    solver_log.append({"Y": Y, "ok": bool(res.success), "iter": int(res.nit),
                        "ann_var": float(w @ Sig @ w), "target": target,
                        "cf_real": cf_real, "binds": abs(cf_real - target) / max(abs(target), 1e-12) < 1e-3,
                        "violates": bool(violates), "n_active": int((w > 1e-6).sum())})

    mv05_w_dict[Y] = pd.Series(w, index=eligible)

    R_next = get_oos_returns(Y)   # ← cache hit
    next_months = _months_of[Y + 1]
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    mv05_ret[Y + 1] = pd.Series(port_ret, index=next_months)
    _mv05_drift[Y + 1] = dict(zip(eligible, ww))

rp_mv05 = pd.concat(mv05_ret).droplevel(0).sort_index()
rp_mv05.index = pd.DatetimeIndex(rp_mv05.index)

solver_df = pd.DataFrame(solver_log).set_index("Y")
print("\n   Solver / constraint diagnostics:")
print(solver_df.round(3).to_string())

# =============================================================================
# 26. VERIFICATION — MV05
# =============================================================================
print("\n[20] Verification — P^(mv)_oos(0.5) ...")

for Y, w in mv05_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-8).all()
print("   ✓ Weights ∈ [0,1], sum to 1")

mv05_carbon = pd.DataFrame(
    [{"Y": Y, **carbon_metrics(mv05_w_dict[Y], Y)} for Y in years_part2]
).set_index("Y")

cf_check = pd.DataFrame({
    "target_raw": cf_target_mv, "target_eff": pd.Series(adj_targets),
    "CF_realized": mv05_carbon["CF"], "WACI_realized": mv05_carbon["WACI"],
})
cf_check["slack_vs_eff"] = cf_check["target_eff"] - cf_check["CF_realized"]
print("\n   CF realized vs target:")
print(cf_check.round(3).to_string())

violations = cf_check[cf_check["CF_realized"] > cf_check["target_eff"] * (1 + 1e-3)]
if len(violations):
    print(f"\n   ⚠ {len(violations)} year(s) with constraint violation > 0.1%")
else:
    print("   ✓ CF constraint satisfied every year")

# Variance cost of constraint (uses cache — no recomputation)
var_rows = []
for Y in years_part2:
    _, Sig = get_cov(Y)
    w_unc = mv_w_dict[Y].values
    w_con = mv05_w_dict[Y].values
    var_rows.append({"Y": Y, "var_MV": float(w_unc @ Sig @ w_unc),
                     "var_MV05": float(w_con @ Sig @ w_con)})
var_df = pd.DataFrame(var_rows).set_index("Y")
var_df["delta_pp"] = (np.sqrt(var_df["var_MV05"]) - np.sqrt(var_df["var_MV"])) * np.sqrt(12) * 100
print("\n   Ex-ante volatility cost of carbon constraint:")
print((var_df.assign(
    sigma_MV=np.sqrt(var_df["var_MV"]) * np.sqrt(12) * 100,
    sigma_MV05=np.sqrt(var_df["var_MV05"]) * np.sqrt(12) * 100,
)[["sigma_MV", "sigma_MV05", "delta_pp"]]).round(3).to_string())

# =============================================================================
# 27. PERFORMANCE — MV05
# =============================================================================
print("\n[21] Performance — MV vs MV(0.5) ...")
stats = pd.DataFrame([
    compute_perf(rp_mv,    rf_mon, "P^(mv)_oos"),
    compute_perf(rp_mv05,  rf_mon, "P^(mv)_oos(0.5)"),
    compute_perf(rp_vw,    rf_mon, "P^(vw) (benchmark)"),
]).set_index("Portfolio")
print(stats.to_string())

# =============================================================================
# 28. COMPOSITION SHIFTS + SECTOR DECOMPOSITION + MOST AVOIDED — Section 3.2
# =============================================================================
print("\n[22] Composition shifts — top firms excluded / overweighted vs MV ...")


def composition_diff(Y, top_n=10):
    w_mv  = mv_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    w_mv5 = mv05_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    diff = (w_mv5 - w_mv).sort_values()
    drops, adds = diff.head(top_n), diff.tail(top_n).iloc[::-1]

    def name(i): return isin_name.get(i, i)
    print(f"\n   --- Y={Y} ---")
    print(f"   Most REMOVED (Δ weight, pp):")
    for isin, dlt in drops.items():
        ci_i = _firm_ci(isin, Y)
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")
    print(f"   Most ADDED (Δ weight, pp):")
    for isin, dlt in adds.items():
        ci_i = _firm_ci(isin, Y)
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")


for Y in snapshot_years:
    composition_diff(Y, top_n=5)

# ── NEW: Sector decomposition + most avoided (MV05 active position vs MV) ────
print("\n" + "─" * 55)
print("SECTION 3 — Most Avoided Positions (Active Investor: MV05 vs MV)")
print("─" * 55)

for Y in snapshot_years:
    w_mv  = mv_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    w_mv5 = mv05_w_dict[Y].reindex(universe[Y]).fillna(0.0)

    most_avoided(w_mv5, w_mv, Y, n=10,
                  title=f"MV(0.5) vs MV — firms cut most by carbon constraint")

# =============================================================================
# 29. PLOTS — Section 3.2
# =============================================================================
print("\n[23] Plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section 3.2 — Min-Var vs Min-Var(0.5)", fontsize=13, fontweight="bold")
C1, C2, C3 = "steelblue", "crimson", "darkorange"
fmt = mdates.DateFormatter("%Y"); loc = mdates.YearLocator(2)

ax = axes[0, 0]
for rp, color, ls, lab in [(rp_mv, C1, "-",  r"MV $P^{(mv)}_{oos}$"),
                            (rp_mv05, C2, "-",  r"MV(0.5) $P^{(mv)}_{oos}(0.5)$"),
                            (rp_vw, C3, "--", r"VW $P^{(vw)}$")]:
    cum = (1 + rp.dropna()).cumprod()
    ax.plot(cum.index, cum.values, color=color, ls=ls, lw=1.8, label=lab)
ax.set_title("Cumulative Return (base=1, Jan 2014)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[0, 1]


def dd(rp):
    c = (1 + rp.dropna()).cumprod()
    return (c - c.cummax()) / c.cummax() * 100


ax.fill_between(dd(rp_mv).index,   dd(rp_mv).values,   0, alpha=0.45, color=C1, label="MV")
ax.fill_between(dd(rp_mv05).index, dd(rp_mv05).values, 0, alpha=0.45, color=C2, label="MV(0.5)")
ax.set_title("Drawdown from Peak (%)"); ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[1, 0]
ax.plot(mv_carbon.index,   mv_carbon["WACI"],   "o-", color=C1, label="MV")
ax.plot(mv05_carbon.index, mv05_carbon["WACI"], "s-", color=C2, label="MV(0.5)")
ax.plot(vw_carbon.index,   vw_carbon["WACI"],   "x--", color=C3, label="VW")
ax.set_title("WACI evolution"); ax.set_ylabel("tCO₂ / M$ rev")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(mv_carbon.index,   mv_carbon["CF"],   "o-", color=C1, label="MV")
ax.plot(mv05_carbon.index, mv05_carbon["CF"], "s-", color=C2, label="MV(0.5)")
ax.plot(cf_target_mv.index, cf_target_mv.values, "k:", lw=1, label="0.5×CF(MV) target")
ax.plot(vw_carbon.index,   vw_carbon["CF"],   "x--", color=C3, label="VW")
ax.set_title("CF evolution + 50% target"); ax.set_ylabel("tCO₂ / M$ inv")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part2_section32.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: SAAM_Part2_section32.{pdf,png}")
print("\n[Section 3.2 complete]")

# =============================================================================
# SECTION 3.3 — Passive Investor: TE-Min with CF ≤ 0.5 × CF(VW)
# =============================================================================
print("\n" + "=" * 65)
print("Section 3.3 — Passive Investor: TE-Min with CF ≤ 0.5 × CF(VW)")
print("=" * 65)

cf_target_vw = (vw_carbon["CF"] * 0.5).rename("target_CF").copy()
print("\n[24] CF targets for P^(vw)_oos(0.5):")
print(cf_target_vw.round(3).to_string())

adj_targets_vw = feasibility_check(cf_target_vw, "VW05")  # reuses _mfcf_cache


def min_te_cf_constrained(Sigma, w_bench, c, target, isins, warm_start=None):
    """
    min (α−w_bench)'Σ(α−w_bench)  s.t. α'e=1, c'α≤target, α≥0.
    """
    N = Sigma.shape[0]
    valid = np.isfinite(c)
    invalid = ~valid
    w0 = (warm_start if warm_start is not None else w_bench.copy())
    w0[invalid] = 0.0
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.where(valid, 1.0, 0.0) / max(valid.sum(), 1)

    bounds = [(0.0, 0.0) if invalid[i] else (0.0, 1.0) for i in range(N)]
    c_safe = np.where(valid, c, 0.0)
    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0,         "jac": lambda w: np.ones(N)},
        {"type": "ineq", "fun": lambda w: target - c_safe @ w,   "jac": lambda w: -c_safe},
    ]
    return minimize(
        fun=lambda w: float((w - w_bench) @ Sigma @ (w - w_bench)),
        x0=w0, jac=lambda w: 2.0 * (Sigma @ (w - w_bench)),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 3000},
    )


print("\n[25] Rolling VW(0.5) optimization ...")

vw05_w_dict = {}
vw05_ret = {}
solver_log_vw = []

for Y in years_part2:
    eligible = universe[Y]
    target = adj_targets_vw[Y]
    _, Sig = get_cov(Y)        # ← cache hit
    c, valid = cf_vector(eligible, Y)
    w_bench = vw_weights(Y).reindex(eligible).fillna(0.0).values

    res = min_te_cf_constrained(Sig, w_bench, c, target, eligible)
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w /= w.sum()

    cf_real = float(np.where(valid, c, 0.0) @ w)
    te2 = float((w - w_bench) @ Sig @ (w - w_bench))
    violates = cf_real > target * (1 + 1e-6)
    if violates:
        print(f"   ⚠ Y={Y}: CF violated ({cf_real:.3f} > {target:.3f})")
    if not res.success:
        print(f"   ⚠ Y={Y}: SLSQP did not converge — {res.message}")

    solver_log_vw.append({"Y": Y, "ok": bool(res.success), "iter": int(res.nit),
                           "ann_TE_pct": np.sqrt(max(te2, 0.0)) * np.sqrt(12) * 100,
                           "target": target, "cf_real": cf_real,
                           "binds": abs(cf_real - target) / max(abs(target), 1e-12) < 1e-3,
                           "violates": bool(violates), "n_active": int((w > 1e-6).sum())})

    vw05_w_dict[Y] = pd.Series(w, index=eligible)

    R_next = get_oos_returns(Y)   # ← cache hit
    next_months = _months_of[Y + 1]
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    vw05_ret[Y + 1] = pd.Series(port_ret, index=next_months)

rp_vw05 = pd.concat(vw05_ret).droplevel(0).sort_index()
rp_vw05.index = pd.DatetimeIndex(rp_vw05.index)

solver_df_vw = pd.DataFrame(solver_log_vw).set_index("Y")
print("\n   Solver / TE diagnostics:")
print(solver_df_vw.round(3).to_string())


# =============================================================================
# DIAGNOSTIC — Eigenvalue along active direction (run after §3.3)
# =============================================================================
print("\n" + "="*65)
print("DIAGNOSTIC — Σ along active direction, Y=2018")
print("="*65)

Y_diag = 2018
isins = universe[Y_diag]
N = len(isins)
win_in = [c for c in _estim_window[Y_diag] if c in ret_m.columns]
R = ret_m.loc[isins, win_in].values

# Active direction: the actual VW(0.5) tilt your code produced
w_active = (vw05_w_dict[Y_diag] - vw_weights(Y_diag).reindex(isins).fillna(0.0)).values

# (a) Pairwise-complete sample (no shrinkage) — rebuild from scratch
not_nan = ~np.isnan(R)
R_zero = np.where(not_nan, R, 0.0)
nn_f = not_nan.astype(float)
count = nn_f @ nn_f.T
safe = np.maximum(count, 1)
mu_ij_i = (R_zero @ nn_f.T) / safe
mu_ij_j = (nn_f @ R_zero.T) / safe
Sigma_sample = (R_zero @ R_zero.T) / safe - mu_ij_i * mu_ij_j
Sigma_sample = (Sigma_sample + Sigma_sample.T) / 2

# (b) LW-CC — your current Σ
_, Sigma_cc = get_cov(Y_diag)

# Active variance along the realised tilt
def avar(S, a):
    return float(a @ S @ a)

v_sample = avar(Sigma_sample, w_active)
v_cc     = avar(Sigma_cc,     w_active)

ann_te_sample = np.sqrt(max(v_sample, 0)) * np.sqrt(12) * 100
ann_te_cc     = np.sqrt(max(v_cc,     0)) * np.sqrt(12) * 100

# Realized TE for that year
y2018_months = _months_of[Y_diag + 1]
realised = (rp_vw05.loc[y2018_months] - rp_vw.loc[y2018_months]).std() * np.sqrt(12) * 100

print(f"  Active-direction annualized TE, Y={Y_diag}:")
print(f"    Sample Σ  : {ann_te_sample:.3f}%")
print(f"    LW-CC Σ   : {ann_te_cc:.3f}%")
print(f"    Realised  : {realised:.3f}%   (single-year, noisy)")
print(f"    Sample / LW-CC ratio = {ann_te_sample/ann_te_cc:.1f}×")

# =============================================================================
# 30. VERIFICATION — VW05
# =============================================================================
print("\n[26] Verification — P^(vw)_oos(0.5) ...")

for Y, w in vw05_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-8).all()
print("   ✓ Weights ∈ [0,1], sum to 1")

vw05_carbon = pd.DataFrame(
    [{"Y": Y, **carbon_metrics(vw05_w_dict[Y], Y)} for Y in years_part2]
).set_index("Y")
cf_check_vw = pd.DataFrame({
    "target_eff": pd.Series(adj_targets_vw), "CF_realized": vw05_carbon["CF"],
    "WACI_realized": vw05_carbon["WACI"],
})
cf_check_vw["slack"] = cf_check_vw["target_eff"] - cf_check_vw["CF_realized"]
print("\n   CF realized vs target:")
print(cf_check_vw.round(3).to_string())
violations = cf_check_vw[cf_check_vw["CF_realized"] > cf_check_vw["target_eff"] * 1.001]
print(f"   {'⚠ ' + str(len(violations)) + ' violations' if len(violations) else '✓ CF constraint satisfied every year'}")

te_ep = (rp_vw05.dropna() - rp_vw.dropna()).std() * np.sqrt(12) * 100
print(f"\n   Realized annualized TE of VW(0.5) vs VW = {te_ep:.2f}%")

# =============================================================================
# 31. PERFORMANCE — VW05
# =============================================================================
print("\n[27] Performance — VW vs VW(0.5) ...")
stats33 = pd.DataFrame([
    compute_perf(rp_vw,    rf_mon, "P^(vw) (benchmark)"),
    compute_perf(rp_vw05,  rf_mon, "P^(vw)_oos(0.5)"),
    compute_perf(rp_mv05,  rf_mon, "P^(mv)_oos(0.5) [reference]"),
]).set_index("Portfolio")
print(stats33.to_string())

ar = (rp_vw05 - rp_vw).dropna()
ir = ar.mean() * 12 / (ar.std() * np.sqrt(12)) if ar.std() > 0 else np.nan
print(f"\n   IR (VW(0.5) vs VW) = {ir:.3f}  |  AR={ar.mean()*12*100:.2f}%  |  TE={te_ep:.2f}%")

# =============================================================================
# 32. COMPOSITION SHIFTS + SECTOR DECOMPOSITION + MOST AVOIDED — Section 3.3
# =============================================================================
print("\n[28] Composition shifts — VW(0.5) vs VW ...")


def composition_diff_vw(Y, top_n=5):
    w_vw  = vw_weights(Y).reindex(universe[Y]).fillna(0.0)
    w_vw5 = vw05_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    diff = (w_vw5 - w_vw).sort_values()

    def name(i): return isin_name.get(i, i)
    print(f"\n   --- Y={Y} ---")
    print(f"   Most REMOVED:")
    for isin, dlt in diff.head(top_n).items():
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   CI = {_firm_ci(isin,Y) if pd.isna(_firm_ci(isin,Y)) else round(_firm_ci(isin,Y),0):>6}")
    print(f"   Most ADDED:")
    for isin, dlt in diff.tail(top_n).iloc[::-1].items():
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   CI = {_firm_ci(isin,Y) if pd.isna(_firm_ci(isin,Y)) else round(_firm_ci(isin,Y),0):>6}")


for Y in snapshot_years:
    composition_diff_vw(Y)

# ── NEW: Sector decomposition + most avoided (VW05 active position vs VW) ────
print("\n" + "─" * 55)
print("SECTION 3 — Most Avoided Positions (Passive Investor: VW05 vs VW)")
print("─" * 55)

for Y in snapshot_years:
    w_vw  = vw_weights(Y).reindex(universe[Y]).fillna(0.0)
    w_vw5 = vw05_w_dict[Y].reindex(universe[Y]).fillna(0.0)

    most_avoided(w_vw5, w_vw, Y, n=10,
                  title=f"VW(0.5) vs VW — firms cut most by carbon constraint")

# =============================================================================
# 33. PLOTS — Section 3.3
# =============================================================================
print("\n[29] Plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section 3.3 — VW vs VW(0.5)", fontsize=13, fontweight="bold")
C_VW, C_VW5, C_MV5 = "darkorange", "seagreen", "crimson"
fmt = mdates.DateFormatter("%Y"); loc = mdates.YearLocator(2)

ax = axes[0, 0]
for rp, color, ls, lab in [
    (rp_vw,   C_VW,  "--", r"VW $P^{(vw)}$"),
    (rp_vw05, C_VW5, "-",  r"VW(0.5) $P^{(vw)}_{oos}(0.5)$"),
    (rp_mv05, C_MV5, ":",  r"MV(0.5) [ref]"),
]:
    cum = (1 + rp.dropna()).cumprod()
    ax.plot(cum.index, cum.values, color=color, ls=ls, lw=1.8, label=lab)
ax.set_title("Cumulative Return (base=1, Jan 2014)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[0, 1]
ar_cum = (1 + ar).cumprod()
ax.plot(ar_cum.index, (ar_cum - 1) * 100, color=C_VW5, lw=1.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_title(f"Cumulative active return: VW(0.5) − VW (IR={ir:.2f})")
ax.set_ylabel("Cumulative active return (%)"); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[1, 0]
ax.plot(vw_carbon.index,   vw_carbon["WACI"],   "x--", color=C_VW,  label="VW")
ax.plot(vw05_carbon.index, vw05_carbon["WACI"], "s-",  color=C_VW5, label="VW(0.5)")
ax.set_title("WACI evolution"); ax.set_ylabel("tCO₂ / M$ rev")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(vw_carbon.index,   vw_carbon["CF"],   "x--", color=C_VW,  label="VW")
ax.plot(vw05_carbon.index, vw05_carbon["CF"], "s-",  color=C_VW5, label="VW(0.5)")
ax.plot(cf_target_vw.index, cf_target_vw.values, "k:", lw=1, label="0.5×CF(VW) target")
ax.set_title("CF evolution + 50% target"); ax.set_ylabel("tCO₂ / M$ inv")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part2_section33.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: SAAM_Part2_section33.{pdf,png}")
print("\n[Section 3.3 complete]")

# =============================================================================
# SECTION 4.1 — Net-Zero Glide Path
# =============================================================================
print("\n" + "=" * 65)
print("Section 4.1 — Net-Zero Glide Path: TE-Min with annual 10% cut")
print("=" * 65)

THETA = 0.10
CF_VW_2013 = float(vw_carbon.loc[2013, "CF"])


def nz_target(Y):
    return (1.0 - THETA) ** (Y - 2013 + 1) * CF_VW_2013


cf_target_nz = pd.Series({Y: nz_target(Y) for Y in years_part2}, name="target_CF")

target_compare = pd.DataFrame({
    "VW_realized":   vw_carbon["CF"],
    "0.5xVW_target": cf_target_vw,
    "NZ_target":     cf_target_nz,
})
target_compare["NZ_pct_of_2013"] = target_compare["NZ_target"] / CF_VW_2013 * 100
target_compare["NZ_vs_VW"] = target_compare["NZ_target"] / target_compare["VW_realized"]
target_compare["NZ_tighter_than_0.5xVW"] = target_compare["NZ_target"] < target_compare["0.5xVW_target"]

print("\n[30] Glide path comparison:")
print(target_compare.round(2).to_string())
print(f"\n   Years where NZ is tighter than 0.5×VW: "
      f"{int(target_compare['NZ_tighter_than_0.5xVW'].sum())} / {len(years_part2)}")

adj_targets_nz = feasibility_check(cf_target_nz, "VWNZ")   # reuses _mfcf_cache

print("\n[31] Rolling VW(NZ) optimization ...")

vwnz_w_dict = {}
vwnz_ret = {}
solver_log_nz = []

for Y in years_part2:
    eligible = universe[Y]
    target = adj_targets_nz[Y]
    _, Sig = get_cov(Y)        # ← cache hit
    c, valid = cf_vector(eligible, Y)
    w_bench = vw_weights(Y).reindex(eligible).fillna(0.0).values

    res = min_te_cf_constrained(Sig, w_bench, c, target, eligible)
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w /= w.sum()

    cf_real = float(np.where(valid, c, 0.0) @ w)
    te2 = float((w - w_bench) @ Sig @ (w - w_bench))
    violates = cf_real > target * (1 + 1e-6)
    if violates:
        print(f"   ⚠ Y={Y}: CF violated ({cf_real:.3f} > {target:.3f})")
    if not res.success:
        print(f"   ⚠ Y={Y}: SLSQP did not converge — {res.message}")

    solver_log_nz.append({"Y": Y, "ok": bool(res.success), "iter": int(res.nit),
                           "ann_TE_pct": np.sqrt(max(te2, 0.0)) * np.sqrt(12) * 100,
                           "target": target, "cf_real": cf_real,
                           "binds": abs(cf_real - target) / max(abs(target), 1e-12) < 1e-3,
                           "violates": bool(violates)})

    vwnz_w_dict[Y] = pd.Series(w, index=eligible)

    R_next = get_oos_returns(Y)   # ← cache hit
    next_months = _months_of[Y + 1]
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    vwnz_ret[Y + 1] = pd.Series(port_ret, index=next_months)

rp_vwnz = pd.concat(vwnz_ret).droplevel(0).sort_index()
rp_vwnz.index = pd.DatetimeIndex(rp_vwnz.index)

solver_df_nz = pd.DataFrame(solver_log_nz).set_index("Y")
print("\n   Solver / TE diagnostics:")
print(solver_df_nz.round(3).to_string())

# =============================================================================
# 34. VERIFICATION — VWNZ
# =============================================================================
print("\n[32] Verification — P^(vw)_oos(NZ) ...")

for Y, w in vwnz_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-8).all()
print("   ✓ Weights ∈ [0,1], sum to 1")

vwnz_carbon = pd.DataFrame(
    [{"Y": Y, **carbon_metrics(vwnz_w_dict[Y], Y)} for Y in years_part2]
).set_index("Y")
cf_check_nz = pd.DataFrame({
    "target_eff": pd.Series(adj_targets_nz), "CF_realized": vwnz_carbon["CF"],
    "WACI_realized": vwnz_carbon["WACI"],
    "pct_of_2013_CF": vwnz_carbon["CF"] / CF_VW_2013 * 100,
})
cf_check_nz["slack"] = cf_check_nz["target_eff"] - cf_check_nz["CF_realized"]
print("\n   Realized CF vs target (and as % of 2013 baseline):")
print(cf_check_nz.round(3).to_string())

final_pct = cf_check_nz.loc[2024, "pct_of_2013_CF"]
expected_pct = (1 - THETA) ** (2024 - 2013 + 1) * 100
print(f"\n   2024 CF as % of 2013 baseline = {final_pct:.1f}%  (target {expected_pct:.1f}%)")
print(f"   {'✓' if final_pct <= expected_pct + 0.5 else '⚠'} on/under glide path")

te_ep_nz = (rp_vwnz.dropna() - rp_vw.dropna()).std() * np.sqrt(12) * 100
print(f"\n   Realized annualized TE of VW(NZ) vs VW = {te_ep_nz:.2f}%")

# =============================================================================
# 35. PERFORMANCE — All Passive
# =============================================================================
print("\n[33] Performance — passive investor portfolios ...")
stats41 = pd.DataFrame([
    compute_perf(rp_vw,    rf_mon, "P^(vw)"),
    compute_perf(rp_vw05,  rf_mon, "P^(vw)_oos(0.5)"),
    compute_perf(rp_vwnz,  rf_mon, "P^(vw)_oos(NZ)"),
]).set_index("Portfolio")
print(stats41.to_string())

ar_nz = (rp_vwnz - rp_vw).dropna()
te_nz_realized = ar_nz.std() * np.sqrt(12)
ir_nz = (ar_nz.mean() * 12) / te_nz_realized if te_nz_realized > 0 else np.nan
print(f"\n   IR (VW(NZ) vs VW)  = {ir_nz:.3f}  AR={ar_nz.mean()*12*100:+.2f}%  TE={te_nz_realized*100:.2f}%")
print(f"   IR (VW(0.5) vs VW) = {ir:.3f}  AR={ar.mean()*12*100:+.2f}%  TE={te_ep:.2f}%")

# =============================================================================
# 36. CARBON METRICS — ALL PASSIVE PORTFOLIOS
# =============================================================================
print("\n[34] CF / WACI evolution — VW vs VW(0.5) vs VW(NZ)")
all_carbon = pd.DataFrame({
    "VW_CF": vw_carbon["CF"], "VW05_CF": vw05_carbon["CF"], "VWNZ_CF": vwnz_carbon["CF"],
    "VW_WACI": vw_carbon["WACI"], "VW05_WACI": vw05_carbon["WACI"], "VWNZ_WACI": vwnz_carbon["WACI"],
})
print(all_carbon.round(2).to_string())
print(f"\n   Cumulative CF — VW: {vw_carbon['CF'].sum():.1f} | "
      f"VW(0.5): {vw05_carbon['CF'].sum():.1f} ({vw05_carbon['CF'].sum()/vw_carbon['CF'].sum()*100:.1f}%) | "
      f"VW(NZ): {vwnz_carbon['CF'].sum():.1f} ({vwnz_carbon['CF'].sum()/vw_carbon['CF'].sum()*100:.1f}%)")

# =============================================================================
# 37. COMPOSITION SHIFTS + SECTOR DECOMPOSITION + MOST AVOIDED — Section 4.1
# =============================================================================
print("\n[35] Composition shifts — VW(NZ) vs VW ...")


def composition_diff_vwnz(Y, top_n=5):
    w_vw = vw_weights(Y).reindex(universe[Y]).fillna(0.0)
    w_nz = vwnz_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    diff = (w_nz - w_vw).sort_values()

    def name(i): return isin_name.get(i, i)
    print(f"\n   --- Y={Y}  (target={adj_targets_nz[Y]:.1f}, realized={vwnz_carbon.loc[Y,'CF']:.1f}) ---")
    print(f"   Most REMOVED:")
    for isin, dlt in diff.head(top_n).items():
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   CI = {_firm_ci(isin,Y) if pd.isna(_firm_ci(isin,Y)) else round(_firm_ci(isin,Y),0):>6}")
    print(f"   Most ADDED:")
    for isin, dlt in diff.tail(top_n).iloc[::-1].items():
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   CI = {_firm_ci(isin,Y) if pd.isna(_firm_ci(isin,Y)) else round(_firm_ci(isin,Y),0):>6}")


for Y in snapshot_years:
    composition_diff_vwnz(Y)

# ── NEW: Sector decomposition + most avoided (VWNZ active position vs VW) ────
print("\n" + "─" * 55)
print("SECTION 4 — Most Avoided Positions (Net-Zero: VWNZ vs VW)")
print("─" * 55)

for Y in snapshot_years:
    w_vw = vw_weights(Y).reindex(universe[Y]).fillna(0.0)
    w_nz = vwnz_w_dict[Y].reindex(universe[Y]).fillna(0.0)

    most_avoided(w_nz, w_vw, Y, n=10,
                  title=f"VW(NZ) vs VW — firms cut most by net-zero glide path")

# =============================================================================
# 38. PLOTS — Section 4.1
# =============================================================================
print("\n[36] Plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section 4.1 — VW vs VW(0.5) vs VW(NZ)", fontsize=13, fontweight="bold")
C_VW, C_VW5, C_NZ = "darkorange", "seagreen", "purple"
fmt = mdates.DateFormatter("%Y"); loc = mdates.YearLocator(2)

ax = axes[0, 0]
for rp, color, ls, lab in [
    (rp_vw,   C_VW,  "--", r"VW $P^{(vw)}$"),
    (rp_vw05, C_VW5, "-",  r"VW(0.5)"),
    (rp_vwnz, C_NZ,  "-",  r"VW(NZ)"),
]:
    cum = (1 + rp.dropna()).cumprod()
    ax.plot(cum.index, cum.values, color=color, ls=ls, lw=1.8, label=lab)
ax.set_title("Cumulative Return (base=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

ax = axes[0, 1]
ax.plot(vw_carbon.index,   vw_carbon["CF"],   "x--", color=C_VW,  label="VW CF")
ax.plot(vw05_carbon.index, vw05_carbon["CF"], "s-",  color=C_VW5, label="VW(0.5) CF")
ax.plot(vwnz_carbon.index, vwnz_carbon["CF"], "o-",  color=C_NZ,  label="VW(NZ) CF")
ax.plot(cf_target_nz.index, cf_target_nz.values, ":", color=C_NZ, lw=1, label="NZ target")
ax.plot(cf_target_vw.index, cf_target_vw.values, ":", color=C_VW5, lw=1, label="0.5×VW target")
ax.set_title("CF evolution + glide paths"); ax.set_ylabel("tCO₂ / M$ inv")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(vw_carbon.index,   vw_carbon["WACI"],   "x--", color=C_VW,  label="VW")
ax.plot(vw05_carbon.index, vw05_carbon["WACI"], "s-",  color=C_VW5, label="VW(0.5)")
ax.plot(vwnz_carbon.index, vwnz_carbon["WACI"], "o-",  color=C_NZ,  label="VW(NZ)")
ax.set_title("WACI evolution"); ax.set_ylabel("tCO₂ / M$ rev")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
ar05_cum = (1 + (rp_vw05 - rp_vw).dropna()).cumprod()
arnz_cum = (1 + (rp_vwnz - rp_vw).dropna()).cumprod()
ax.plot(ar05_cum.index, (ar05_cum - 1) * 100, color=C_VW5, lw=1.5,
        label=f"VW(0.5) − VW (IR={ir:+.2f})")
ax.plot(arnz_cum.index, (arnz_cum - 1) * 100, color=C_NZ, lw=1.5,
        label=f"VW(NZ) − VW (IR={ir_nz:+.2f})")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Cumulative active return vs VW"); ax.set_ylabel("Cumulative active return (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT}SAAM_Part2_section41.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: SAAM_Part2_section41.{pdf,png}")

print("\n[Section 4.1 complete] — Part II finished.")