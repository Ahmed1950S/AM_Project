"""
SAAM Part II — Section 3.2
Active investor: Minimum-Variance portfolio with carbon footprint
constrained to ≤ 0.5 × CF(P^(mv)_oos) each year.

Continues from Sections 3.1; assumes mv_carbon, mv_w_dict, universe,
co2_tot, mv_y, ret_m, etc. are in memory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, linprog

print("\n" + "=" * 65)
print("Section 3.2 — Active Investor: MV with CF ≤ 0.5 × CF(MV)")
print("=" * 65)

# =============================================================================
# 19. PER-YEAR CF TARGETS (locked)
# =============================================================================
# Choice #16: freeze targets as 0.5 × baseline MV CF computed in Section 3.1
cf_target_mv = (mv_carbon["CF"] * 0.5).rename("target_CF").copy()
print("\n[18] CF targets for P^(mv)_oos(0.5):")
print(cf_target_mv.round(3).to_string())


# =============================================================================
# 20. PER-FIRM CF VECTOR c_Y = E_i,Y / Cap_i,Y
# =============================================================================
def cf_vector(isins, Y):
    """
    Per-firm CF coefficients c_i = E_i,Y / Cap_i,Y for the year-Y universe.

    Returns
    -------
    c : np.ndarray, shape (N,)
        Coefficients used in the linear constraint c'α ≤ target.
        Firms with NaN/zero Cap or NaN E get c_i = +inf (effectively excluded).
    valid : np.ndarray bool, shape (N,)
        True where the CF coefficient is well-defined.
    """
    E = co2_tot.loc[isins, Y].values.astype(float)
    C = mv_y.loc[isins, Y].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where((C > 0) & np.isfinite(C) & np.isfinite(E), E / C, np.inf)
    valid = np.isfinite(c)
    return c, valid


# =============================================================================
# 21. FEASIBILITY CHECK — minimum achievable CF in each universe
# =============================================================================
def min_feasible_cf(isins, Y):
    """
    Solve  min_α  c'α   s.t.  Σα = 1,  α ≥ 0   (a simple LP).

    Returns the minimum CF achievable in the year-Y universe by any
    long-only fully-invested portfolio. This is the lower bound the CF
    constraint must respect.
    """
    c, valid = cf_vector(isins, Y)
    N = len(isins)
    if not valid.any():
        return np.nan
    # Replace inf with a huge but finite penalty so linprog stays well-posed
    c_lp = np.where(valid, c, 1e20)
    res = linprog(
        c=c_lp,
        A_eq=np.ones((1, N)), b_eq=[1.0],
        bounds=[(0.0, 1.0)] * N,
        method="highs",
    )
    if not res.success:
        return np.nan
    return float(c_lp @ res.x)


# Precompute min-feasible CF and adjusted targets
adj_targets = {}
infeas_log = []
for Y in years_part2:
    isins = universe[Y]
    cf_min = min_feasible_cf(isins, Y)
    target_raw = float(cf_target_mv.loc[Y])
    if cf_min > target_raw:
        # Original target is infeasible — relax to slightly above min
        target_eff = cf_min * 1.001
        infeas_log.append({"Y": Y, "target_raw": target_raw,
                           "cf_min": cf_min, "target_eff": target_eff})
        adj_targets[Y] = target_eff
    else:
        adj_targets[Y] = target_raw

if infeas_log:
    print("\n   ⚠ Infeasibility detected — targets relaxed:")
    print(pd.DataFrame(infeas_log).round(3).to_string(index=False))
else:
    print("\n   ✓ All raw targets feasible (no relaxation needed)")


# =============================================================================
# 22. CONSTRAINED MIN-VARIANCE OPTIMIZER
# =============================================================================
def min_var_cf_constrained(Sigma, c, target, isins, Y, prev_drift=None):
    """
    Solve:  min  α'Σα
            s.t. α'e = 1,
                 c'α ≤ target,
                 α_i ≥ 0,
                 α_i = 0 if c_i is invalid (NaN cap)

    Warm start: drifted weights of P^(mv)_oos(0.5) at end of year Y if available
    (passed via prev_drift), else equal weights on valid firms.
    """
    N = Sigma.shape[0]
    valid = np.isfinite(c)
    invalid = ~valid

    # Warm start
    if prev_drift is not None:
        w0 = np.array([prev_drift.get(i, 0.0) for i in isins])
        # Zero out invalid firms; renormalize
        w0[invalid] = 0.0
        s = w0.sum()
        w0 = w0 / s if s > 0 else np.where(valid, 1.0, 0.0) / valid.sum()
    else:
        w0 = np.where(valid, 1.0, 0.0) / max(valid.sum(), 1)

    # Bounds: zero for invalid firms (forces α_i = 0)
    bounds = [(0.0, 0.0) if invalid[i] else (0.0, 1.0) for i in range(N)]

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0,
                          "jac": lambda w: np.ones(N)},
        {"type": "ineq", "fun": lambda w: target - np.where(valid, c, 0.0) @ w,
                          "jac": lambda w: -np.where(valid, c, 0.0)},
    ]

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=w0,
        jac=lambda w: 2.0 * (Sigma @ w),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    return res


# =============================================================================
# 23. ROLLING LOOP — P^(mv)_oos(0.5)
# =============================================================================
print("\n[19] Rolling MV(0.5) optimization ...")

mv05_w_dict = {}     # year-end weights (post-rebalance)
mv05_ret = {}        # next-year monthly returns
_mv05_drift = {}     # drifted weights, year-end

solver_log = []

for Y in years_part2:
    eligible = universe[Y]
    N = len(eligible)
    target = adj_targets[Y]

    # Reuse Part 1 covariance (same Σ as MV unconstrained)
    mu, Sig = estimate_cov(eligible, estim_window(Y))
    c, valid = cf_vector(eligible, Y)

    res = min_var_cf_constrained(
        Sig, c, target, eligible, Y,
        prev_drift=_mv05_drift.get(Y),
    )
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w = w / w.sum()  # numerical clean-up

    # Verify the realized CF and constraint
    cf_real = float(np.where(valid, c, 0.0) @ w)
    binds = abs(cf_real - target) / max(abs(target), 1e-12) < 1e-3
    violates = cf_real > target * (1 + 1e-6)

    solver_log.append({
        "Y": Y,
        "ok": bool(res.success),
        "iter": int(res.nit),
        "ann_var": float(w @ Sig @ w),
        "target": target,
        "cf_real": cf_real,
        "binds": bool(binds),
        "violates": bool(violates),
        "n_active": int((w > 1e-6).sum()),
    })

    if violates:
        print(f"   ⚠ Y={Y}: CF constraint violated  ({cf_real:.3f} > {target:.3f})")

    mv05_w_dict[Y] = pd.Series(w, index=eligible)

    # Compute next-year ex-post returns (drifted weights)
    next_months = months_of(Y + 1)
    R_next = fill_oos_returns(eligible, next_months)
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    mv05_ret[Y + 1] = pd.Series(port_ret, index=next_months)
    _mv05_drift[Y + 1] = dict(zip(eligible, ww))

# Concat returns
rp_mv05 = pd.concat(mv05_ret).droplevel(0).sort_index()
rp_mv05.index = pd.DatetimeIndex(rp_mv05.index)

solver_df = pd.DataFrame(solver_log).set_index("Y")
print("\n   Solver / constraint diagnostics:")
print(solver_df.round(3).to_string())


# =============================================================================
# 24. VERIFICATION
# =============================================================================
print("\n[20] Verification — P^(mv)_oos(0.5) ...")

# A. All weight vectors sum to 1, non-negative
for Y, w in mv05_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6, f"Y={Y}: weights sum to {w.sum()}"
    assert (w >= -1e-8).all(), f"Y={Y}: negative weights"
print("   ✓ Weights ∈ [0,1], sum to 1")

# B. CF constraint satisfied (use carbon_metrics, the reporting function,
#    which must agree with the optimizer's c'α)
mv05_carbon_rows = []
for Y in years_part2:
    mv05_carbon_rows.append({"Y": Y, **carbon_metrics(mv05_w_dict[Y], Y)})
mv05_carbon = pd.DataFrame(mv05_carbon_rows).set_index("Y")

cf_check = pd.DataFrame({
    "target_raw":  cf_target_mv,
    "target_eff":  pd.Series(adj_targets),
    "CF_realized": mv05_carbon["CF"],
    "WACI_realized": mv05_carbon["WACI"],
    "feasible_raw": cf_target_mv >= pd.Series({Y: min_feasible_cf(universe[Y], Y)
                                                for Y in years_part2}),
})
cf_check["slack_vs_eff"] = cf_check["target_eff"] - cf_check["CF_realized"]
print("\n   CF realized vs target:")
print(cf_check.round(3).to_string())

violations = cf_check[cf_check["CF_realized"] > cf_check["target_eff"] * (1 + 1e-3)]
if len(violations):
    print(f"\n   ⚠ {len(violations)} year(s) with constraint violation > 0.1%")
else:
    print("   ✓ CF constraint satisfied (within 0.1%) every year")

# C. Identity: optimizer-internal c'α should equal carbon_metrics CF (sanity)
print("\n   Cross-check: optimizer c'α vs carbon_metrics CF")
max_diff = 0.0
for Y in years_part2:
    isins = universe[Y]
    c, valid = cf_vector(isins, Y)
    w = mv05_w_dict[Y].values
    cf_opt = float(np.where(valid, c, 0.0) @ w)
    cf_rep = float(mv05_carbon.loc[Y, "CF"])
    max_diff = max(max_diff, abs(cf_opt - cf_rep))
print(f"   max |c'α − carbon_metrics.CF| = {max_diff:.3e}  (should be ~0)")
assert max_diff < 1e-8, "Optimizer CF and reporting CF disagree"
print("   ✓ Identity holds")

# D. Ex-ante variance should be ≥ unconstrained MV variance (constraint adds cost)
mv_vars = []
for Y in years_part2:
    eligible = universe[Y]
    _, Sig = estimate_cov(eligible, estim_window(Y))
    w_unc = mv_w_dict[Y].values
    w_con = mv05_w_dict[Y].values
    mv_vars.append({
        "Y": Y,
        "var_MV": float(w_unc @ Sig @ w_unc),
        "var_MV05": float(w_con @ Sig @ w_con),
    })
var_df = pd.DataFrame(mv_vars).set_index("Y")
var_df["delta_pp"] = (np.sqrt(var_df["var_MV05"]) - np.sqrt(var_df["var_MV"])) * np.sqrt(12) * 100
print("\n   Ex-ante annualised volatility (%): MV vs MV(0.5)")
print((var_df.assign(
    sigma_MV=np.sqrt(var_df["var_MV"]) * np.sqrt(12) * 100,
    sigma_MV05=np.sqrt(var_df["var_MV05"]) * np.sqrt(12) * 100,
)[["sigma_MV", "sigma_MV05", "delta_pp"]]).round(3).to_string())

if (var_df["var_MV05"] < var_df["var_MV"] - 1e-10).any():
    print("   ⚠ MV(0.5) has LOWER variance than MV — adding a constraint cannot do this")
else:
    print("   ✓ MV(0.5) variance ≥ MV variance every year (cost of carbon constraint)")


# =============================================================================
# 25. PERFORMANCE COMPARISON — P^(mv)_oos vs P^(mv)_oos(0.5)
# =============================================================================
print("\n[21] Performance — MV vs MV(0.5) ...")

stats = pd.DataFrame([
    compute_perf(rp_mv,    rf_mon, "P^(mv)_oos"),
    compute_perf(rp_mv05,  rf_mon, "P^(mv)_oos(0.5)"),
    compute_perf(rp_vw,    rf_mon, "P^(vw) (benchmark)"),
]).set_index("Portfolio")
print(stats.to_string())


# =============================================================================
# 26. PORTFOLIO COMPOSITION DIFFS
# =============================================================================
print("\n[22] Composition shifts — top firms excluded / overweighted vs MV ...")

def composition_diff(Y, top_n=10):
    w_mv  = mv_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    w_mv5 = mv05_w_dict[Y].reindex(universe[Y]).fillna(0.0)
    diff = (w_mv5 - w_mv).sort_values()

    drops = diff.head(top_n)
    adds  = diff.tail(top_n).iloc[::-1]

    def name(i): return isin_name.get(i, i)
    print(f"\n   --- Y={Y} ---")
    print(f"   Most REMOVED (Δ weight, pp):")
    for isin, dlt in drops.items():
        ci_i = (co2_tot.loc[isin, Y] / rev_m.loc[isin, Y]) if (
            isin in co2_tot.index and isin in rev_m.index
            and pd.notna(rev_m.loc[isin, Y]) and rev_m.loc[isin, Y] > 0) else np.nan
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")
    print(f"   Most ADDED (Δ weight, pp):")
    for isin, dlt in adds.items():
        ci_i = (co2_tot.loc[isin, Y] / rev_m.loc[isin, Y]) if (
            isin in co2_tot.index and isin in rev_m.index
            and pd.notna(rev_m.loc[isin, Y]) and rev_m.loc[isin, Y] > 0) else np.nan
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")

for Y in [2013, 2018, 2024]:
    composition_diff(Y, top_n=5)


# =============================================================================
# 27. PLOTS — cumulative return, drawdown, WACI/CF
# =============================================================================
print("\n[23] Plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section 3.2 — Min-Var vs Min-Var(0.5)", fontsize=13, fontweight="bold")
C1, C2, C3 = "steelblue", "crimson", "darkorange"
fmt = mdates.DateFormatter("%Y"); loc = mdates.YearLocator(2)

# Cumulative returns
ax = axes[0, 0]
for rp, color, ls, lab in [(rp_mv, C1, "-",  r"MV $P^{(mv)}_{oos}$"),
                           (rp_mv05, C2, "-",  r"MV(0.5) $P^{(mv)}_{oos}(0.5)$"),
                           (rp_vw, C3, "--", r"VW $P^{(vw)}$")]:
    cum = (1 + rp.dropna()).cumprod()
    ax.plot(cum.index, cum.values, color=color, ls=ls, lw=1.8, label=lab)
ax.set_title("Cumulative Return (base=1, Jan 2014)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# Drawdown
ax = axes[0, 1]
def dd(rp):
    c = (1 + rp.dropna()).cumprod()
    return (c - c.cummax()) / c.cummax() * 100
ax.fill_between(dd(rp_mv).index,   dd(rp_mv).values,   0, alpha=0.45, color=C1, label="MV")
ax.fill_between(dd(rp_mv05).index, dd(rp_mv05).values, 0, alpha=0.45, color=C2, label="MV(0.5)")
ax.set_title("Drawdown from Peak (%)"); ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# WACI
ax = axes[1, 0]
ax.plot(mv_carbon.index,   mv_carbon["WACI"],   "o-", color=C1, label="MV")
ax.plot(mv05_carbon.index, mv05_carbon["WACI"], "s-", color=C2, label="MV(0.5)")
ax.plot(vw_carbon.index,   vw_carbon["WACI"],   "x--", color=C3, label="VW")
ax.set_title("WACI evolution"); ax.set_ylabel("tCO₂ / M$ rev")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# CF
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

print("\n[Section 3.2 complete] — proceed to 3.3: P^(vw)_oos(0.5) tracking-error min")
