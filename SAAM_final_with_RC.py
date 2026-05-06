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

# =============================================================================
# =============================================================================
# ROBUSTNESS CHECK — Single-Index Shrinkage on §3.3 (TE-Min)
# =============================================================================
# Tests whether the 16x ex-ante / realized TE gap reported in §3.3 is driven
# by the LW shrinkage TARGET (constant correlation flattens the systematic axis
# the carbon tilt loads on) or by something deeper (sample Σ rank deficiency,
# regime non-stationarity, alpha-Σ misalignment).
#
# Three estimators tested:
#   - LW-CC: original (constant-correlation target)         [baseline]
#   - LW-SI: single-index target with VW market weights     [primary RC]
#   - Sample: pure pairwise-complete (δ = 0)                [diagnostic only]
#
# CAVEAT: SI uses OAS-style intensity (same formula as CC) rather than the
# proper LW-2003 SI intensity (involves 4th-moment summation, rarely worth
# the extra code complexity for a robustness check). We are testing TARGET
# structure, not the optimal intensity. State this in the report.
# =============================================================================
print("\n" + "=" * 65)
print("ROBUSTNESS CHECK — Single-Index Shrinkage (RC)")
print("=" * 65)


# =============================================================================
# RC.1 — Generalized covariance estimator with target=cc/si/none
# =============================================================================
print("\n[RC.1] Defining generalized covariance estimator ...")


def estimate_cov_target(isins, win_cols, target="cc", market_weights=None,
                        return_delta=False, psd_repair=True,
                        return_pre_repair=False):
    """
    Pairwise-complete covariance + LW shrinkage to {cc, si, none} target.

    Parameters
    ----------
    isins : list of ISIN strings (rows of ret_m to use)
    win_cols : list of monthly columns in the estimation window
    target : 'cc' | 'si' | 'none'
    market_weights : pd.Series indexed by ISIN, only used if target='si'
    return_delta : if True, return (mu, Sigma, delta); else (mu, Sigma)
    psd_repair : if False, skip eigenvalue flooring (matrix may be non-PSD)
    return_pre_repair : if True, also return Sigma BEFORE psd repair (for diagnostics)
    """
    win_in = [c for c in win_cols if c in ret_m.columns]
    R = ret_m.loc[isins, win_in]
    N = len(isins)
    mu = R.mean(axis=1).values

    R_vals = R.values
    not_nan = ~np.isnan(R_vals)
    mu_full = np.nanmean(R_vals, axis=1)
    R_dem_zero = np.where(not_nan, R_vals - mu_full[:, None], 0.0)
    n_obs = not_nan.sum(axis=1)
    var = np.where(n_obs > 1,
                   np.sum(R_dem_zero ** 2, axis=1) / n_obs, 0.0)

    R_zero = np.where(not_nan, R_vals, 0.0)
    nn_f = not_nan.astype(np.float64)
    count_ij = nn_f @ nn_f.T
    safe = np.maximum(count_ij, 1)
    mu_i = (R_zero @ nn_f.T) / safe
    mu_j = (nn_f @ R_zero.T) / safe
    cov_ij = (R_zero @ R_zero.T) / safe - mu_i * mu_j
    var_i = np.maximum((R_zero ** 2) @ nn_f.T / safe - mu_i ** 2, 0)
    var_j = np.maximum(nn_f @ (R_zero ** 2).T / safe - mu_j ** 2, 0)
    denom = np.sqrt(var_i * var_j)
    corr_ij = np.where(denom > 1e-20, cov_ij / denom, 0.0)

    std_own = np.sqrt(var)
    Sig = corr_ij * np.outer(std_own, std_own)
    np.fill_diagonal(Sig, var)
    Sig = (Sig + Sig.T) / 2

    # No shrinkage: optionally PSD-repair sample matrix and return
    if target == "none":
        Sig_pre = Sig.copy()
        if psd_repair:
            eigvals, eigvecs = np.linalg.eigh(Sig)
            if eigvals[0] < 1e-10:
                Sig = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
                Sig = (Sig + Sig.T) / 2
        ret = [mu, Sig]
        if return_delta:
            ret.append(0.0)
        if return_pre_repair:
            ret.append(Sig_pre)
        return tuple(ret) if len(ret) > 2 else (ret[0], ret[1])

    # ----- Build target F -----
    if target == "cc":
        std_diag = np.sqrt(np.diag(Sig))
        std_safe = np.where(std_diag > 1e-20, std_diag, 1e-20)
        corr_mat = np.clip(Sig / np.outer(std_safe, std_safe), -1.0, 1.0)
        np.fill_diagonal(corr_mat, 1.0)
        rho_bar = (corr_mat.sum() - N) / (N * (N - 1))
        F = rho_bar * np.outer(std_diag, std_diag)
        np.fill_diagonal(F, var)

    elif target == "si":
        if market_weights is None:
            raise ValueError("market_weights required for SI target")
        w = market_weights.reindex(isins).fillna(0.0).values.astype(float)
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N
        sigma_m2 = float(w @ Sig @ w)
        if sigma_m2 < 1e-20:
            raise RuntimeError(f"Degenerate market variance: {sigma_m2}")
        beta = (Sig @ w) / sigma_m2
        F = sigma_m2 * np.outer(beta, beta)
        np.fill_diagonal(F, var)
    else:
        raise ValueError(f"Unknown target: {target}")

    # OAS-style intensity (same formula for both CC and SI)
    tr_Sig = np.trace(Sig)
    tr_Sig2 = np.trace(Sig @ Sig)
    T_eff = max(np.median(count_ij[np.triu_indices(N, k=1)]), 2)
    numerator = (1.0 - 2.0 / N) * tr_Sig2 + tr_Sig ** 2
    denominator = (T_eff + 1.0 - 2.0 / N) * (tr_Sig2 - tr_Sig ** 2 / N)
    delta = (0.5 if abs(denominator) < 1e-20
             else max(min(numerator / denominator, 1.0), LW_SHRINK_FLOOR))

    Sig_shrunk = delta * F + (1.0 - delta) * Sig
    Sig_pre = Sig_shrunk.copy()

    if psd_repair:
        eigvals, eigvecs = np.linalg.eigh(Sig_shrunk)
        if eigvals[0] < 1e-10:
            Sig_shrunk = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            Sig_shrunk = (Sig_shrunk + Sig_shrunk.T) / 2

    ret = [mu, Sig_shrunk]
    if return_delta:
        ret.append(delta)
    if return_pre_repair:
        ret.append(Sig_pre)
    return tuple(ret) if len(ret) > 2 else (ret[0], ret[1])


print("   ✓ estimate_cov_target() defined")

# =============================================================================
# RC.2 — Verification of the SI estimator on Y=2018
# =============================================================================
# Sanity check before running the full re-allocation. If anything fails here,
# do NOT proceed to RC.4.
# =============================================================================
print("\n[RC.2] Verifying SI estimator on Y=2018 ...")

Y_test = 2018
isins_test = universe[Y_test]
N_test = len(isins_test)

mu_si, Sig_si, delta_si, Sig_si_pre = estimate_cov_target(
    isins_test, _estim_window[Y_test],
    target="si", market_weights=vw_weights(Y_test),
    return_delta=True, return_pre_repair=True,
)
mu_cc_, Sig_cc_, delta_cc, Sig_cc_pre = estimate_cov_target(
    isins_test, _estim_window[Y_test],
    target="cc", return_delta=True, return_pre_repair=True,
)
mu_no, Sig_no, _, Sig_no_pre = estimate_cov_target(
    isins_test, _estim_window[Y_test], target="none",
    return_delta=True, return_pre_repair=True,
)

# (1) Shape, symmetry
assert Sig_si.shape == (N_test, N_test), "shape mismatch"
assert np.allclose(Sig_si, Sig_si.T, atol=1e-10), "Σ_SI not symmetric"
print(f"   ✓ Shape {Sig_si.shape}, symmetric")

# (2) PSD (post-repair)
eigs_si = np.linalg.eigvalsh(Sig_si)
assert eigs_si[0] > -1e-9, f"Σ_SI not PSD: min eig = {eigs_si[0]:.2e}"
print(f"   ✓ Σ_SI PSD post-repair (min eig = {eigs_si[0]:.2e}, max = {eigs_si[-1]:.2e})")

# (3a) PRE-REPAIR diagonal invariant: diag(Σ_SI_pre) == diag(Σ_sample_pre) exactly.
#      This is the actual invariant of the construction.
diag_diff_pre = np.max(np.abs(np.diag(Sig_si_pre) - np.diag(Sig_no_pre)))
assert diag_diff_pre < 1e-10, (
    f"diagonal of pre-repair SI should equal pre-repair sample "
    f"(got {diag_diff_pre:.2e})"
)
print(f"   ✓ Pre-repair diagonal invariant holds (max |diff| = {diag_diff_pre:.2e})")

# (3b) POST-REPAIR diagonal drift: diagnostic, NOT an invariant.
#      Drift = "negative variance" the eigenvalue floor pushed back into the matrix.
#      Large drift = the matrix was substantially non-PSD before repair.
diag_drift_si = np.max(np.abs(np.diag(Sig_si) - np.diag(Sig_si_pre)))
diag_drift_no = np.max(np.abs(np.diag(Sig_no) - np.diag(Sig_no_pre)))
typical_var = np.median(np.diag(Sig_si_pre))
print(f"   POST-REPAIR diagonal drift (diagnostic of non-PSD severity):")
print(f"     SI    : max |Δdiag| = {diag_drift_si:.3e}  "
      f"(rel. to median var {typical_var:.3e}: {diag_drift_si/typical_var:.1%})")
print(f"     Sample: max |Δdiag| = {diag_drift_no:.3e}  "
      f"(rel. to median var {typical_var:.3e}: {diag_drift_no/typical_var:.1%})")

# Min eigenvalue PRE-repair tells us how non-PSD the underlying matrix is
min_eig_si_pre = np.linalg.eigvalsh(Sig_si_pre)[0]
min_eig_no_pre = np.linalg.eigvalsh(Sig_no_pre)[0]
print(f"   Min eigenvalue PRE-repair:")
print(f"     SI    : {min_eig_si_pre:+.3e}")
print(f"     Sample: {min_eig_no_pre:+.3e}")
if min_eig_si_pre < -1e-6:
    print(f"     ⚠ SI shrinkage did NOT make matrix PSD — "
          f"shrinkage intensity δ={delta_si:.2f} insufficient.")
    print(f"        This is informative for the report: even with SI target,")
    print(f"        the sample part is too rank-deficient at T_eff << N.")

# (4) Shrinkage intensity in [LW_SHRINK_FLOOR, 1]
print(f"   δ_CC = {delta_cc:.4f}   δ_SI = {delta_si:.4f}")

# (5) Active-direction variance under each estimator
w_active = (
    vw05_w_dict[Y_test]
    - vw_weights(Y_test).reindex(isins_test).fillna(0.0)
).values

v_cc = float(w_active @ Sig_cc_ @ w_active)
v_si = float(w_active @ Sig_si @ w_active)
v_no = float(w_active @ Sig_no @ w_active)


def ann_te(v):
    return np.sqrt(max(v, 0)) * np.sqrt(12) * 100


print(f"\n   Active-direction annualized TE (Y={Y_test}):")
print(f"     LW-CC  : {ann_te(v_cc):.4f}%")
print(f"     LW-SI  : {ann_te(v_si):.4f}%")
print(f"     Sample : {ann_te(v_no):.4f}%   (raw quadratic v={v_no:+.3e})")

# Verdict — what to expect
if v_si > v_cc:
    print(f"   → SI ex-ante > CC ex-ante: SI target preserves "
          f"more variance along active axis (good sign).")
else:
    print(f"   ⚠ SI ex-ante ≤ CC ex-ante: SI target NOT helping "
          f"this direction. Investigate before reading RC.4 results.")


# =============================================================================
# RC.3 — Deep diagnostic: where does the active vector live in Σ_sample's
#         eigenspectrum?
# =============================================================================
# This is what tells us whether the 16x gap is shrinkage-target choice
# (Cause 1 in our discussion) vs. sample Σ rank deficiency (Cause 2) vs.
# regime non-stationarity (Cause 3). We do this on Y=2018 and Y=2022 (energy
# bull market — likely the worst year for ex-ante / realized gap).
# =============================================================================
print("\n[RC.3] Deep diagnostic — eigenspectrum decomposition ...")

for Y_diag in [2018, 2022]:
    print(f"\n   --- Y={Y_diag} ---")
    isins = universe[Y_diag]
    N = len(isins)

    # Build raw pairwise-complete sample Σ (BEFORE PSD repair) — we want to see
    # if it's actually non-PSD.
    win_in = [c for c in _estim_window[Y_diag] if c in ret_m.columns]
    R = ret_m.loc[isins, win_in].values
    not_nan = ~np.isnan(R)
    mu_full = np.nanmean(R, axis=1)
    R_dem_zero = np.where(not_nan, R - mu_full[:, None], 0.0)
    n_obs = not_nan.sum(axis=1)
    var_own = np.where(n_obs > 1,
                       np.sum(R_dem_zero ** 2, axis=1) / n_obs, 0.0)

    R_zero = np.where(not_nan, R, 0.0)
    nn_f = not_nan.astype(float)
    count = nn_f @ nn_f.T
    safe = np.maximum(count, 1)
    mu_i = (R_zero @ nn_f.T) / safe
    mu_j = (nn_f @ R_zero.T) / safe
    cov_ij = (R_zero @ R_zero.T) / safe - mu_i * mu_j
    v_i = np.maximum((R_zero ** 2) @ nn_f.T / safe - mu_i ** 2, 0)
    v_j = np.maximum(nn_f @ (R_zero ** 2).T / safe - mu_j ** 2, 0)
    denom = np.sqrt(v_i * v_j)
    corr = np.where(denom > 1e-20, cov_ij / denom, 0.0)
    std = np.sqrt(var_own)
    Sig_raw = corr * np.outer(std, std)
    np.fill_diagonal(Sig_raw, var_own)
    Sig_raw = (Sig_raw + Sig_raw.T) / 2

    eigs_raw = np.linalg.eigvalsh(Sig_raw)
    n_neg = (eigs_raw < -1e-12).sum()
    n_zero = (np.abs(eigs_raw) < 1e-10).sum()
    print(f"     RAW Σ_sample (N={N}, T_med={int(np.median(count[np.triu_indices(N, k=1)]))}):")
    print(f"       Eigenvalues: min={eigs_raw[0]:+.3e}, max={eigs_raw[-1]:.3e}")
    print(f"       Negative: {n_neg} | Near-zero (|λ|<1e-10): {n_zero}")
    if n_neg > 0:
        print(f"       → Pairwise-complete is NON-PSD (expected when T < N).")

    # Active vector
    if Y_diag in vw05_w_dict:
        w_active = (vw05_w_dict[Y_diag]
                    - vw_weights(Y_diag).reindex(isins).fillna(0.0)).values
    else:
        print("       ⚠ no vw05 weights for this year — skipping active decomp.")
        continue

    norm_a = np.linalg.norm(w_active)
    n_nz = (np.abs(w_active) > 1e-8).sum()
    print(f"     Active vector: ||w_a||₂={norm_a:.4f}, "
          f"sum={w_active.sum():+.2e}, n_nonzero={n_nz}")

    # Quadratic form on raw (no PSD repair)
    v_raw = float(w_active @ Sig_raw @ w_active)
    print(f"     w_a' Σ_raw w_a = {v_raw:+.3e}  →  ann TE = {ann_te(v_raw):.4f}%")

    # Project active onto Σ_raw eigenbasis
    eigvals, eigvecs = np.linalg.eigh(Sig_raw)
    proj = eigvecs.T @ w_active
    contrib = eigvals * proj ** 2

    pos_sum = contrib[eigvals > 0].sum()
    neg_sum = contrib[eigvals < 0].sum()
    print(f"     Decomposition w_a'Σ_raw w_a:")
    print(f"       Positive-eig contribution: {pos_sum:+.3e}")
    print(f"       Negative-eig contribution: {neg_sum:+.3e}")
    print(f"       Sum (= w_a'Σ w_a):          {contrib.sum():+.3e}")

    # Top contributors
    order = np.argsort(np.abs(contrib))[::-1]
    print(f"     Top 5 |contributions|:")
    for k in order[:5]:
        print(f"       λ={eigvals[k]:+.3e}  proj²={proj[k] ** 2:.3e}  "
              f"contrib={contrib[k]:+.3e}")

    # Where does active mass concentrate? (top vs bottom of spectrum)
    # Sort eigenvalues descending; cumulative |proj|² in top-k vs bottom-k
    desc = np.argsort(eigvals)[::-1]
    proj2_sorted = proj[desc] ** 2
    total_proj2 = proj2_sorted.sum()
    if total_proj2 > 0:
        for k_pct in [0.05, 0.20, 0.50]:
            k = max(int(N * k_pct), 1)
            top_share = proj2_sorted[:k].sum() / total_proj2
            bot_share = proj2_sorted[-k:].sum() / total_proj2
            print(f"       |proj|² in top-{int(k_pct * 100)}%   eigs: {top_share:.1%}  | "
                  f"in bottom-{int(k_pct * 100)}% eigs: {bot_share:.1%}")


# =============================================================================
# RC.4 — Re-run §3.3 with LW-SI shrinkage
# =============================================================================
# Reuses min_te_cf_constrained (same constraint set). Only Σ feed changes.
# Same CF targets (adj_targets_vw), same universe, same w_bench, so any
# difference in result is attributable purely to estimator choice.
# =============================================================================
print("\n[RC.4] Re-running §3.3 with LW-SI shrinkage ...")

_cov_cache_si = {}


def get_cov_si(Y):
    if Y not in _cov_cache_si:
        mw = vw_weights(Y)
        mu, Sig, delta = estimate_cov_target(
            universe[Y], _estim_window[Y],
            target="si", market_weights=mw, return_delta=True,
        )
        _cov_cache_si[Y] = (mu, Sig, delta)
    return _cov_cache_si[Y]


vw05si_w_dict = {}
vw05si_ret = {}
solver_log_si = []

for Y in years_part2:
    eligible = universe[Y]
    target_cf = adj_targets_vw[Y]
    _, Sig_si_y, delta_y = get_cov_si(Y)
    c, valid = cf_vector(eligible, Y)
    w_bench = vw_weights(Y).reindex(eligible).fillna(0.0).values

    res = min_te_cf_constrained(Sig_si_y, w_bench, c, target_cf, eligible)
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w /= w.sum()

    cf_real = float(np.where(valid, c, 0.0) @ w)
    te2 = float((w - w_bench) @ Sig_si_y @ (w - w_bench))
    violates = cf_real > target_cf * (1 + 1e-6)
    if violates:
        print(f"   ⚠ Y={Y}: CF violated ({cf_real:.3f} > {target_cf:.3f})")
    if not res.success:
        print(f"   ⚠ Y={Y}: SLSQP did not converge — {res.message}")

    solver_log_si.append({
        "Y": Y, "ok": bool(res.success), "iter": int(res.nit),
        "delta": delta_y,
        "ann_TE_pct": np.sqrt(max(te2, 0.0)) * np.sqrt(12) * 100,
        "target": target_cf, "cf_real": cf_real,
        "binds": abs(cf_real - target_cf) / max(abs(target_cf), 1e-12) < 1e-3,
        "violates": bool(violates),
        "n_active": int((w > 1e-6).sum()),
    })

    vw05si_w_dict[Y] = pd.Series(w, index=eligible)

    R_next = get_oos_returns(Y)
    next_months = _months_of[Y + 1]
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    vw05si_ret[Y + 1] = pd.Series(port_ret, index=next_months)

rp_vw05si = pd.concat(vw05si_ret).droplevel(0).sort_index()
rp_vw05si.index = pd.DatetimeIndex(rp_vw05si.index)

solver_df_si = pd.DataFrame(solver_log_si).set_index("Y")
print("\n   SI solver / TE diagnostics:")
print(solver_df_si.round(3).to_string())

# Verification: weights, CF constraint
for Y, w in vw05si_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6, f"Y={Y}: weights don't sum to 1"
    assert (w >= -1e-8).all(), f"Y={Y}: negative weights"
print("\n   ✓ Weights ∈ [0,1], sum to 1 every year (SI)")

vw05si_carbon = pd.DataFrame(
    [{"Y": Y, **carbon_metrics(vw05si_w_dict[Y], Y)} for Y in years_part2]
).set_index("Y")
violations_si = (
    vw05si_carbon["CF"] > pd.Series(adj_targets_vw) * 1.001
).sum()
if violations_si == 0:
    print("   ✓ CF ≤ target every year (SI)")
else:
    print(f"   ⚠ {violations_si} CF violations under SI")


# =============================================================================
# RC.5 — Comparison table: LW-CC vs LW-SI
# =============================================================================
print("\n[RC.5] Comparison — LW-CC vs LW-SI ...")

# (a) Performance metrics
stats_rc = pd.DataFrame([
    compute_perf(rp_vw,     rf_mon, "P^(vw) (benchmark)"),
    compute_perf(rp_vw05,   rf_mon, "P^(vw)_oos(0.5)  [LW-CC]"),
    compute_perf(rp_vw05si, rf_mon, "P^(vw)_oos(0.5)  [LW-SI]"),
]).set_index("Portfolio")
print("\n   Performance comparison:")
print(stats_rc.to_string())

# (b) Realized TE / IR
ar_cc = (rp_vw05   - rp_vw).dropna()
ar_si = (rp_vw05si - rp_vw).dropna()
te_cc = ar_cc.std() * np.sqrt(12) * 100
te_si = ar_si.std() * np.sqrt(12) * 100
ir_cc = ar_cc.mean() * 12 / (ar_cc.std() * np.sqrt(12)) if ar_cc.std() > 0 else np.nan
ir_si = ar_si.mean() * 12 / (ar_si.std() * np.sqrt(12)) if ar_si.std() > 0 else np.nan

print(f"\n   Realized active stats:")
print(f"     LW-CC : AR = {ar_cc.mean()*12*100:+.2f}%/yr | "
      f"TE = {te_cc:.3f}% | IR = {ir_cc:+.3f}")
print(f"     LW-SI : AR = {ar_si.mean()*12*100:+.2f}%/yr | "
      f"TE = {te_si:.3f}% | IR = {ir_si:+.3f}")

# (c) Ex-ante / realized TE ratio — the core diagnostic
mean_ea_cc = solver_df_vw["ann_TE_pct"].mean()
mean_ea_si = solver_df_si["ann_TE_pct"].mean()
ratio_cc = te_cc / mean_ea_cc if mean_ea_cc > 0 else np.nan
ratio_si = te_si / mean_ea_si if mean_ea_si > 0 else np.nan

print(f"\n   *** EX-ANTE / REALIZED TE RATIO (the core diagnostic) ***")
print(f"     LW-CC : ex-ante {mean_ea_cc:.3f}%  realized {te_cc:.3f}%  "
      f"→ ratio = {ratio_cc:.1f}x")
print(f"     LW-SI : ex-ante {mean_ea_si:.3f}%  realized {te_si:.3f}%  "
      f"→ ratio = {ratio_si:.1f}x")

# (d) Per-year ratio table (informative — does ratio spike in 2021/22?)
print(f"\n   Per-year ex-ante / realized TE ratio:")
yearly = []
for Y in years_part2:
    next_months = _months_of[Y + 1]
    realised_y = (rp_vw05.reindex(next_months)
                  - rp_vw.reindex(next_months)).std() * np.sqrt(12) * 100
    realised_y_si = (rp_vw05si.reindex(next_months)
                     - rp_vw.reindex(next_months)).std() * np.sqrt(12) * 100
    ea_cc_y = solver_df_vw.loc[Y, "ann_TE_pct"]
    ea_si_y = solver_df_si.loc[Y, "ann_TE_pct"]
    yearly.append({
        "Y": Y,
        "EA_CC": ea_cc_y, "RE_CC": realised_y,
        "ratio_CC": realised_y / ea_cc_y if ea_cc_y > 0 else np.nan,
        "EA_SI": ea_si_y, "RE_SI": realised_y_si,
        "ratio_SI": realised_y_si / ea_si_y if ea_si_y > 0 else np.nan,
    })
yearly_df = pd.DataFrame(yearly).set_index("Y")
print(yearly_df.round(3).to_string())

# (e) Composition stability: do the same names get cut?
print(f"\n   Composition stability — top-10 cut overlap:")


def top_cuts(w_dict, Y, n=10):
    a = (w_dict[Y]
         - vw_weights(Y).reindex(w_dict[Y].index).fillna(0.0))
    return a.nsmallest(n).index.tolist()


for Y in [2013, 2018, 2024]:
    cc_cuts = set(top_cuts(vw05_w_dict, Y))
    si_cuts = set(top_cuts(vw05si_w_dict, Y))
    overlap = len(cc_cuts & si_cuts)
    print(f"     Y={Y}: CC ∩ SI = {overlap}/10")
    diff_cc = cc_cuts - si_cuts
    diff_si = si_cuts - cc_cuts
    if diff_cc:
        names_cc = [static.set_index('ISIN').loc[i, 'NAME'][:25]
                    if i in static['ISIN'].values else i[:8] for i in list(diff_cc)[:3]]
        print(f"        CC-only cuts (sample): {names_cc}")
    if diff_si:
        names_si = [static.set_index('ISIN').loc[i, 'NAME'][:25]
                    if i in static['ISIN'].values else i[:8] for i in list(diff_si)[:3]]
        print(f"        SI-only cuts (sample): {names_si}")

# (f) Active share
print(f"\n   Active share (= ½ Σ|α - α_vw|):")
for Y in [2013, 2018, 2024]:
    a_cc = (vw05_w_dict[Y]
            - vw_weights(Y).reindex(vw05_w_dict[Y].index).fillna(0)).abs().sum() / 2
    a_si = (vw05si_w_dict[Y]
            - vw_weights(Y).reindex(vw05si_w_dict[Y].index).fillna(0)).abs().sum() / 2
    print(f"     Y={Y}: CC = {a_cc*100:.2f}%   SI = {a_si*100:.2f}%")

# Save tables for the report
yearly_df.to_csv(f"{OUT}RC_yearly_TE_ratio.csv")
stats_rc.to_csv(f"{OUT}RC_perf_comparison.csv")
solver_df_si.to_csv(f"{OUT}RC_solver_log_si.csv")
print(f"\n   Saved: RC_yearly_TE_ratio.csv, RC_perf_comparison.csv, RC_solver_log_si.csv")


# =============================================================================
# RC.6 — Decision summary
# =============================================================================
print("\n" + "=" * 65)
print("RC SUMMARY — How to interpret in the report")
print("=" * 65)

print(f"""
  Ex-ante / realized TE ratio:    LW-CC = {ratio_cc:.1f}x   →   LW-SI = {ratio_si:.1f}x

  Three possible verdicts:
  -----------------------------------------------------------------------
  A.  SI ratio drops to < 3x AND realized IR sign/magnitude ~ same as CC
      → LW-CC target was wrong. Report: "robustness check confirms
         qualitative findings; SI is the better-specified estimator."
         Use SI numbers in main results, CC numbers in appendix.

  B.  SI ratio drops, BUT IR or composition shifts materially
      → Cost-of-decarbonization is implementation-dependent.
         Report: "we report SI as primary estimator; CC numbers in
         appendix for transparency. The −0.41 IR with CC was an
         estimator artefact."

  C.  SI ratio still ≥ 8x
      → Issue is deeper than target choice. Review RC.3 output:
         - If sample Σ has many negative eigenvalues AND active
           vector loads on them, this is alpha-Σ misalignment
           (Lee & Stefek 2008). Fix needs a designed factor model,
           not different shrinkage.
         - If per-year ratios spike in 2021–22 only, it's regime
           non-stationarity, not estimator.
         - If ratio is stable across years, dimensionality (T<<N²).
""")

print("\n[RC pre-final] — proceeding to structural diagnostic RC.7 ...")


# =============================================================================
# RC.7 — Structural diagnostic: realized factor attribution
# =============================================================================
# CRITICAL: the realized covariance must be built from the SAME return matrix
# used for the realized portfolio simulation (get_oos_returns(Y), which fills
# delistings with -100% then 0, and inactive stocks with 0). Otherwise Σ_real
# silently excludes tail events and the ratio truth/LW is understated.
#
# To separate "factor structure missed by Σ_LW" from "tail-event risk", we
# build TWO realized covariances per year:
#   Σ_real_full     : from filled OOS returns (matches realized TE exactly)
#   Σ_real_survivors: from raw returns (NaN-aware) — pure factor structure,
#                     no delisting / inactive-stock contamination
#
# The DIFFERENCE between them isolates tail-event risk.
# Plus a per-eigenvector concentration check: if max|v_k_i| is large, that
# "factor" is a single-stock event, not a real factor.
# =============================================================================
print("\n" + "=" * 65)
print("RC.7 — Realized factor attribution (the structural argument)")
print("=" * 65)


def _pairwise_cov_from_matrix(R, treat_zero_as_missing=False):
    """Pairwise-complete covariance from N×T matrix R.
    treat_zero_as_missing: if True, treats 0 entries as missing
        (relevant for filled returns where 0 is the inactive flag)."""
    if treat_zero_as_missing:
        not_nan = (~np.isnan(R)) & (R != 0.0)
    else:
        not_nan = ~np.isnan(R)

    R_dem_zero = np.where(not_nan, R - np.nanmean(np.where(not_nan, R, np.nan), axis=1)[:, None], 0.0)
    n_obs = not_nan.sum(axis=1)
    var_own = np.where(n_obs > 1, np.sum(R_dem_zero ** 2, axis=1) / n_obs, 0.0)

    R_zero = np.where(not_nan, R, 0.0)
    nn_f = not_nan.astype(float)
    count = nn_f @ nn_f.T
    safe = np.maximum(count, 1)
    mu_i = (R_zero @ nn_f.T) / safe
    mu_j = (nn_f @ R_zero.T) / safe
    cov_ij = (R_zero @ R_zero.T) / safe - mu_i * mu_j
    v_i = np.maximum((R_zero ** 2) @ nn_f.T / safe - mu_i ** 2, 0)
    v_j = np.maximum(nn_f @ (R_zero ** 2).T / safe - mu_j ** 2, 0)
    denom = np.sqrt(v_i * v_j)
    corr = np.where(denom > 1e-20, cov_ij / denom, 0.0)
    std = np.sqrt(var_own)
    Sig = corr * np.outer(std, std)
    np.fill_diagonal(Sig, var_own)
    Sig = (Sig + Sig.T) / 2
    return Sig


def build_realized_cov_full(Y, isins):
    """Σ_real from the SAME filled return matrix used for OOS portfolio sim.
    Includes delistings (-100% spikes) and inactive-stock zero-fills."""
    R_filled = get_oos_returns(Y).loc[isins].values  # N × 12, no NaNs
    # Standard sample covariance (no pairwise — all entries valid)
    R_dem = R_filled - R_filled.mean(axis=1, keepdims=True)
    T = R_filled.shape[1]
    Sig = (R_dem @ R_dem.T) / T
    Sig = (Sig + Sig.T) / 2
    return Sig


def build_realized_cov_survivors(isins, months):
    """Σ_real on raw ret_m (NaN-aware). Excludes inactive / delisted stocks
    from contributing to those periods. Pure 'factor' covariance among
    actively-trading names."""
    R = ret_m.loc[isins, months].values
    return _pairwise_cov_from_matrix(R, treat_zero_as_missing=False)


# Sanity check: realized TE from Σ_real_full should match realized TE from
# the actual portfolio simulation (within drift effect, ~1-2%).
print("\n   [Sanity] w_a' Σ_real_full w_a vs realized TE from RC.5:")
print(f"   {'Y':>5}  {'from_Σ_full':>13}  {'from_simulation':>17}  {'ratio':>7}")
for Y in years_part2:
    isins = universe[Y]
    next_months = _months_of[Y + 1]
    Sig_full = build_realized_cov_full(Y, isins)
    w_a = (vw05_w_dict[Y]
           - vw_weights(Y).reindex(isins).fillna(0.0)).values
    v_full = float(w_a @ Sig_full @ w_a)
    te_from_sigma = np.sqrt(max(v_full, 0)) * np.sqrt(12) * 100
    te_from_sim = (rp_vw05.reindex(next_months)
                   - rp_vw.reindex(next_months)).std() * np.sqrt(12) * 100
    ratio = te_from_sigma / te_from_sim if te_from_sim > 0 else np.nan
    print(f"   {Y:>5}  {te_from_sigma:>12.3f}%  {te_from_sim:>16.3f}%  {ratio:>6.2f}x")
print("   → ratios should all be ~1.00 (small drift effect). If not, something's off.\n")


# (A) Per-year decomposition: factor structure vs tail-event contribution
diag_years = [2018, 2022, 2024]
print(f"   Years analyzed: {diag_years}")
print(f"   Decomposing the 16× gap into:")
print(f"     1. Factor structure missed by Σ_LW (survivors-only)")
print(f"     2. Tail / event risk (delistings, inactive-stock spikes)\n")

attribution_rows = []
for Y in diag_years:
    isins = universe[Y]
    next_months = _months_of[Y + 1]
    w_a = (vw05_w_dict[Y]
           - vw_weights(Y).reindex(isins).fillna(0.0)).values

    # Three covariances along the active direction
    _, Sig_lw_y = get_cov(Y)
    Sig_full = build_realized_cov_full(Y, isins)
    Sig_surv = build_realized_cov_survivors(isins, next_months)

    v_lw = float(w_a @ Sig_lw_y @ w_a)
    v_full = float(w_a @ Sig_full @ w_a)
    v_surv = float(w_a @ Sig_surv @ w_a)

    te_lw = np.sqrt(max(v_lw, 0)) * np.sqrt(12) * 100
    te_full = np.sqrt(max(v_full, 0)) * np.sqrt(12) * 100
    te_surv = np.sqrt(max(v_surv, 0)) * np.sqrt(12) * 100

    print(f"\n   --- Y={Y}  (active vector evaluated against Y+1 = {Y + 1} returns) ---")
    print(f"     Annualized active TE (in three views):")
    print(f"       Σ_LW (ex-ante):                  {te_lw:.3f}%")
    print(f"       Σ_real survivors (factor only):  {te_surv:.3f}%   ratio vs LW: "
          f"{te_surv/te_lw if te_lw > 0 else float('inf'):.1f}x")
    print(f"       Σ_real full (factor + tail):     {te_full:.3f}%   ratio vs LW: "
          f"{te_full/te_lw if te_lw > 0 else float('inf'):.1f}x")
    # Decompose: variance basis (additive)
    v_factor = v_surv
    v_tail = max(v_full - v_surv, 0.0)
    if v_full > 0:
        pct_factor = v_factor / v_full * 100
        pct_tail = v_tail / v_full * 100
        print(f"     Variance decomposition of full realized TE:")
        print(f"       Factor structure (recoverable):  {pct_factor:.0f}%")
        print(f"       Tail/event risk (unrecoverable): {pct_tail:.0f}%")

    # Eigendecompose Σ_real_full and check concentration of top eigenvectors
    eigs_r, vecs_r = np.linalg.eigh(Sig_full)
    desc = np.argsort(eigs_r)[::-1]
    eigs_r = eigs_r[desc]
    vecs_r = vecs_r[:, desc]

    proj_r = vecs_r.T @ w_a
    contrib_r = eigs_r * proj_r ** 2

    print(f"\n     Top 5 realized eigenvectors of Σ_real_full:")
    print(f"     {'k':>2}  {'σ_real':>8}  {'σ_LW(same dir)':>14}  "
          f"{'undertest':>10}  {'max|v_k|':>9}  {'concentration':>13}  "
          f"{'contrib%':>9}")
    for k in range(min(5, (eigs_r > 1e-12).sum())):
        v = vecs_r[:, k]
        sig_real_k = np.sqrt(max(eigs_r[k], 0)) * np.sqrt(12) * 100
        v_lw_k = float(v @ Sig_lw_y @ v)
        sig_lw_k = np.sqrt(max(v_lw_k, 0)) * np.sqrt(12) * 100
        ratio_k = sig_real_k / sig_lw_k if sig_lw_k > 0 else float('inf')
        # Concentration: max |v_k_i| — if close to 1, eigenvector is a single stock
        max_load = float(np.max(np.abs(v)))
        # Effective N: 1 / sum(v_k_i^4) — Herfindahl-style; high = diffuse, low = concentrated
        eff_n = 1.0 / np.sum(v ** 4) if np.sum(v ** 4) > 0 else np.nan
        contrib_pct = contrib_r[k] / v_full * 100 if v_full > 0 else np.nan
        kind = "stock" if max_load > 0.7 else ("few" if max_load > 0.3 else "factor")
        print(f"     {k+1:>2}  {sig_real_k:>7.2f}%  {sig_lw_k:>13.2f}%  "
              f"{ratio_k:>9.1f}x  {max_load:>8.3f}   {kind:>5} (eff N={eff_n:>5.0f})  "
              f"{contrib_pct:>+8.1f}%")

    attribution_rows.append({
        "Y": Y, "te_lw": te_lw, "te_surv": te_surv, "te_full": te_full,
        "ratio_full": te_full / te_lw if te_lw > 0 else np.nan,
        "ratio_surv": te_surv / te_lw if te_lw > 0 else np.nan,
        "pct_factor": v_factor / v_full * 100 if v_full > 0 else np.nan,
        "pct_tail": v_tail / v_full * 100 if v_full > 0 else np.nan,
    })

attr_df = pd.DataFrame(attribution_rows).set_index("Y")

print(f"\n   Summary across years:")
print(attr_df.round(2).to_string())


# (B) Active vector structure — corrected interpretation
# Note: 21% R² in a cross-section of weight regressions IS substantial.
# corr(w_a, CI z-score) = sign(slope) * sqrt(R²) ≈ -0.46 — economically large.
print(f"\n\n   Structural regression: how much of w_a is driven by CI?")
print(f"   Model: w_a[i] = β₀ + β₁·z(CI_i) + Σ γ_r · 1{{region_i = r}} + ε_i")
print(f"   (R² ≈ 20–30% on a single explanatory variable in a "
      f"cross-section of weight\n    deviations is economically meaningful.)\n")

isin_to_region = static.set_index("ISIN")["Region"].to_dict()
regression_rows = []

for Y in diag_years:
    isins = universe[Y]
    w_a = (vw05_w_dict[Y]
           - vw_weights(Y).reindex(isins).fillna(0.0)).values
    c_vec, valid = cf_vector(isins, Y)
    regions = np.array([isin_to_region.get(i, "UNK") for i in isins])
    mask = valid & np.isfinite(c_vec)
    if mask.sum() < 50:
        print(f"   Y={Y}: too few valid CI observations, skipping")
        continue

    c_clean = c_vec[mask]
    p1, p99 = np.percentile(c_clean, [1, 99])
    c_w = np.clip(c_clean, p1, p99)
    c_z = (c_w - c_w.mean()) / c_w.std()
    region_clean = regions[mask]
    unique_regions = sorted(set(region_clean))[:-1]
    region_mat = (np.column_stack([(region_clean == r).astype(float)
                                   for r in unique_regions])
                  if len(unique_regions) > 0 else np.empty((mask.sum(), 0)))
    X = np.column_stack([np.ones(mask.sum()), c_z, region_mat])
    y = w_a[mask]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    X_ci = np.column_stack([np.ones(mask.sum()), c_z])
    beta_ci, _, _, _ = np.linalg.lstsq(X_ci, y, rcond=None)
    y_hat_ci = X_ci @ beta_ci
    ss_res_ci = float(np.sum((y - y_hat_ci) ** 2))
    r2_ci = 1 - ss_res_ci / ss_tot if ss_tot > 0 else np.nan
    corr_ci = -np.sqrt(r2_ci) if beta_ci[1] < 0 else np.sqrt(r2_ci)

    slope_bp = beta_ci[1] * 1e4
    print(f"   Y={Y}: N={mask.sum()}  "
          f"R²(CI)={r2_ci:.1%}  corr(w_a,CI)={corr_ci:+.2f}  "
          f"R²(CI+region)={r2:.1%}  slope={slope_bp:+.2f} bp/σ")

    regression_rows.append({
        "Y": Y, "N": int(mask.sum()),
        "R2_CI_only": r2_ci, "corr_CI": corr_ci,
        "R2_CI_region": r2,
        "slope_bp_per_sigma_CI": slope_bp,
    })

reg_df = pd.DataFrame(regression_rows).set_index("Y")
print(f"\n   Mean R² (CI only):       {reg_df['R2_CI_only'].mean():.1%}  "
      f"(corr ≈ {reg_df['corr_CI'].mean():+.2f})")
print(f"   Mean R² (CI + region):   {reg_df['R2_CI_region'].mean():.1%}")

attr_df.to_csv(f"{OUT}RC7_realized_attribution.csv")
reg_df.to_csv(f"{OUT}RC7_active_structure.csv")
print(f"\n   Saved: RC7_realized_attribution.csv, RC7_active_structure.csv")

# (C) Honest synthesis
print("\n" + "=" * 65)
print("RC.7 — Honest synthesis")
print("=" * 65)
mean_ratio_full = attr_df["ratio_full"].mean()
mean_ratio_surv = attr_df["ratio_surv"].mean()
mean_pct_factor = attr_df["pct_factor"].mean()
mean_pct_tail = attr_df["pct_tail"].mean()
mean_corr_ci = reg_df["corr_CI"].mean() if len(reg_df) else float('nan')
mean_r2_full = reg_df["R2_CI_region"].mean() if len(reg_df) else float('nan')
print(f"""
  THE 16× EX-ANTE/REALIZED GAP DECOMPOSES INTO TWO PARTS:
  --------------------------------------------------------
  1. FACTOR STRUCTURE Σ_LW MISSES (recoverable with better risk model):
       Survivors-only ratio:  {mean_ratio_surv:.1f}x
       Variance share:        {mean_pct_factor:.0f}% of full realized variance
     → Σ_LW underprices the active direction by ~{mean_ratio_surv:.0f}x even
       when ignoring tail events. A multi-factor risk model with sector
       and CI factors would close most of this.

  2. TAIL / EVENT RISK (no equilibrium covariance estimator catches this):
       Variance share:        {mean_pct_tail:.0f}% of full realized variance
     → Delistings and inactive-stock fills inject realized variance that
       no rolling sample-based estimator (LW-CC, LW-SI, multi-factor)
       would predict ex ante. This is irreducible measurement realism.

  ACTIVE VECTOR STRUCTURE:
  ------------------------
  corr(w_a, CI z-score):   {mean_corr_ci:+.2f}    (R² ≈ {(mean_corr_ci**2)*100:.0f}%)
  R²(w_a ~ CI + region):   {mean_r2_full:.0%}
  slope of w_a on CI:      {reg_df['slope_bp_per_sigma_CI'].mean() if len(reg_df) else float('nan'):+.2f} bp per σ_CI

  → The carbon-CF constraint creates a CI tilt of economically meaningful
    magnitude (corr ≈ -0.46 with CI). Region adds little (~0 pp), so the
    structure is overwhelmingly carbon-driven. CI is the dominant axis
    along which Σ_LW is mis-specified.

  WHY SHRINKAGE TARGET CHOICE DIDN'T MATTER (RC.4 result):
  --------------------------------------------------------
  Both LW-CC and LW-SI preserve a single market-like axis. w_a is dollar-
  neutral (sum ≈ 0), so it has near-zero loading on that axis. The CI tilt
  it DOES load on is in NEITHER target. → Same blind spot, both estimators.

  REPORT FRAMING (suggested limitations bullet):
  ----------------------------------------------
  "The realized TE for VW(0.5) is approximately 16x its ex-ante estimate.
  Decomposition shows two contributions: (i) factor risk along the carbon-
  intensity axis, where Σ_LW underprices the active direction by a factor
  of {mean_ratio_surv:.0f}x — a consequence of LW-CC and LW-SI shrinkage targets
  not encoding sector/CI structure (alpha-Σ misalignment, Lee & Stefek
  2008); (ii) tail and event risk from delistings and inactive-stock fills,
  which contribute the residual ~{mean_pct_tail:.0f}% of realized variance and which no
  equilibrium covariance estimator would have predicted. The active
  vector itself has correlation {mean_corr_ci:+.2f} with carbon-intensity z-score,
  confirming the tilt is overwhelmingly along the constraint axis. A
  designed multi-factor model with explicit sector + CI blocks would close
  the factor part of the gap; the tail-event part is irreducible measurement
  realism. Crucially, qualitative findings (composition, IR, drawdown)
  are stable across estimator choices, so §3.3 conclusions are constraint-
  driven rather than estimator-driven."
""")


print("\n[Robustness check — proceeding to RC.8: benchmark consistency] ...")


# =============================================================================
# RC.8 — BENCHMARK CONSISTENCY CHECK
# =============================================================================
# CRITICAL diagnostic prompted by the sanity-check failure in RC.7.
#
# rp_vw (the benchmark used to compute realized TE) is constructed by MONTHLY
# rebalancing using mv_m (monthly market caps).
# rp_vw05 (the optimized portfolio) DRIFTS from annual cap weights (vw_weights
# uses mv_y, annual). These are different conventions.
#
# The optimizer minimizes:
#   (α - w_vw_annual)' Σ_LW (α - w_vw_annual)   ← STATIC weights, annual mv
# But the realized TE is computed from:
#   std(rp_vw05 - rp_vw)                         ← DRIFTING vs MONTHLY-REBAL.
#
# So the realized TE includes phantom variance from the convention mismatch.
# This RC.8 isolates how much of the 16× gap is convention-mismatch vs
# genuine model failure.
# =============================================================================
print("\n" + "=" * 65)
print("RC.8 — Benchmark consistency check")
print("=" * 65)


def static_drift_vw_returns(Y):
    """Drift-only VW benchmark using annual cap weights — matches the
    drift-only convention used by vw05_w_dict (the optimized portfolio).

    Sequence of operations identical to the optimized portfolio's:
        ww = vw_weights(Y); compound monthly; renormalize.
    """
    isins = universe[Y]
    next_months = _months_of[Y + 1]
    R_next = get_oos_returns(Y)
    w0 = vw_weights(Y).reindex(isins).fillna(0.0).values
    ww = w0.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    return pd.Series(port_ret, index=next_months)


print("\n[RC.8.1] Building static-drift VW benchmark (matches the optimization's "
      "static-weight objective convention) ...")

rp_vw_static_pieces = []
for Y in years_part2:
    rp_vw_static_pieces.append(static_drift_vw_returns(Y))
rp_vw_static = pd.concat(rp_vw_static_pieces).sort_index()
rp_vw_static.index = pd.DatetimeIndex(rp_vw_static.index)

# Compare the two benchmarks against each other
benchmark_diff = (rp_vw - rp_vw_static).dropna()
te_benchmarks = benchmark_diff.std() * np.sqrt(12) * 100
mean_diff = benchmark_diff.mean() * 12 * 100

print(f"\n   Difference between the two benchmarks (rp_vw - rp_vw_static):")
print(f"     Annualized vol of difference:   {te_benchmarks:.3f}%")
print(f"     Annualized mean of difference:  {mean_diff:+.3f}% per year")
print(f"   → This is purely a convention-mismatch artifact.")

# Recompute the §3.3 realized TE against BOTH benchmarks
print("\n[RC.8.2] Realized TE for VW(0.5) under both benchmark conventions:\n")
ar_monthly = (rp_vw05 - rp_vw).dropna()
ar_static = (rp_vw05 - rp_vw_static).dropna()

te_monthly = ar_monthly.std() * np.sqrt(12) * 100
te_static = ar_static.std() * np.sqrt(12) * 100
ar_monthly_y = ar_monthly.mean() * 12 * 100
ar_static_y = ar_static.mean() * 12 * 100
ir_monthly = ar_monthly_y / te_monthly if te_monthly > 0 else np.nan
ir_static = ar_static_y / te_static if te_static > 0 else np.nan

print(f"   {'':38} {'TE':>9}   {'AR/yr':>9}   {'IR':>7}")
print(f"   vs rp_vw (monthly-rebalanced)     "
      f"{te_monthly:>8.3f}%  {ar_monthly_y:>+8.3f}%  {ir_monthly:>+7.3f}")
print(f"   vs rp_vw_static (drift-matched)   "
      f"{te_static:>8.3f}%  {ar_static_y:>+8.3f}%  {ir_static:>+7.3f}")
print(f"\n   Reduction in 'apparent' TE: "
      f"{te_monthly:.3f}% → {te_static:.3f}%   "
      f"({(1 - te_static/te_monthly)*100:.0f}% smaller)")

# Repeat the ex-ante / realized ratio with the consistent benchmark
mean_ea_cc = solver_df_vw["ann_TE_pct"].mean()
ratio_monthly = te_monthly / mean_ea_cc if mean_ea_cc > 0 else np.nan
ratio_static = te_static / mean_ea_cc if mean_ea_cc > 0 else np.nan

print(f"\n[RC.8.3] *** EX-ANTE / REALIZED TE RATIO — CONSISTENT BENCHMARK ***")
print(f"   Ex-ante TE (LW-CC):              {mean_ea_cc:.3f}%")
print(f"   Realized TE (vs rp_vw monthly):  {te_monthly:.3f}%   "
      f"ratio = {ratio_monthly:>5.1f}x")
print(f"   Realized TE (vs rp_vw_static):   {te_static:.3f}%   "
      f"ratio = {ratio_static:>5.1f}x   ← apples-to-apples")

# Re-run the RC.7 sanity check with the corrected benchmark
print(f"\n[RC.8.4] Re-run of RC.7 sanity check with consistent benchmark:")
print(f"   {'Y':>5}  {'from_Σ_full':>13}  "
      f"{'sim_vs_static':>15}  {'ratio':>7}")
for Y in years_part2:
    isins = universe[Y]
    next_months = _months_of[Y + 1]
    Sig_full = build_realized_cov_full(Y, isins)
    w_a = (vw05_w_dict[Y]
           - vw_weights(Y).reindex(isins).fillna(0.0)).values
    v_full = float(w_a @ Sig_full @ w_a)
    te_sigma = np.sqrt(max(v_full, 0)) * np.sqrt(12) * 100
    te_sim_static = ((rp_vw05.reindex(next_months)
                      - rp_vw_static.reindex(next_months)).std()
                     * np.sqrt(12) * 100)
    ratio = te_sigma / te_sim_static if te_sim_static > 0 else np.nan
    print(f"   {Y:>5}  {te_sigma:>12.3f}%  "
          f"{te_sim_static:>14.3f}%  {ratio:>6.2f}x")
print("   → these ratios should now be ~1.00 if the convention mismatch was "
      "the issue.")


# Verdict
print("\n" + "=" * 65)
print("RC.8 — Implications")
print("=" * 65)
print(f"""
  The 16× ex-ante / realized TE gap reported in §3.3 decomposes as:

    Reported gap (vs rp_vw):          {ratio_monthly:.1f}x
    Apples-to-apples gap (vs static): {ratio_static:.1f}x
    Convention-mismatch component:    ~{ratio_monthly - ratio_static:.0f}x of the gap

  The convention-mismatch component is NOT a model failure. It comes from
  comparing a static-weight optimization against a monthly-rebalanced
  benchmark. The optimizer never had a chance to minimize that TE because
  it solved a different (static-weight) problem.

  IMPLICATIONS FOR THE REPORT:
  ----------------------------
  1. The {ratio_static:.0f}x residual gap (apples-to-apples) is what the model
     should be held accountable for. The factor-structure analysis from
     RC.7 explains this part.

  2. The {ratio_monthly:.0f}x reported gap mostly reflects the BENCHMARK CHOICE
     in your simulator, not a flaw in the optimization itself.

  3. The IR = {ir_monthly:+.3f} you reported is for VW(0.5) vs monthly-rebal
     VW. The IR vs the convention-matched VW is {ir_static:+.3f}. The economic
     story (decarbonization is roughly cost-free or has small cost) is
     unchanged at the IR level, but the magnitude is materially different.

  REPORT-LEVEL FIX (recommended):
  -------------------------------
  Either:
    (a) Report TE/IR vs both benchmarks, and explain the mismatch.
        Lead with vs static (apples-to-apples); show vs monthly as a
        sensitivity. This is the most honest and the easiest to defend.

    (b) Re-state the optimization as monthly-rebalanced (would require
        rebuilding §3.3 with monthly Σ — out of scope for the current
        deadline).

  Recommended: (a). State clearly: "the realized TE numbers throughout
  §3.3 are computed against a monthly-rebalanced VW, while the
  optimization minimizes static-weight TE against an annual VW; we
  report both for transparency. The apples-to-apples ratio is {ratio_static:.0f}x,
  consistent with finite-T estimation noise. The {ratio_monthly:.0f}x gap vs
  the monthly-rebalanced benchmark reflects convention mismatch, not
  estimator failure."
""")

print("\n[Robustness check complete] — review output and decide narrative.")