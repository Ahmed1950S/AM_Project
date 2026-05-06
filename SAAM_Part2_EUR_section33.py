"""
SAAM Part II — Section 3.3
Passive investor: minimize tracking error vs P^(vw) subject to
   CF ≤ 0.5 × CF(P^(vw))   each year.

Continues from Sections 3.1 and 3.2; assumes vw_carbon, mv05_w_dict,
universe, co2_tot, mv_y, ret_m, etc. are in memory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, linprog

print("\n" + "=" * 65)
print("Section 3.3 — Passive Investor: TE-Min with CF ≤ 0.5 × CF(VW)")
print("=" * 65)

# =============================================================================
# 28. PER-YEAR CF TARGETS (locked)
# =============================================================================
# Choice #26: target = 0.5 × baseline VW CF computed in Section 3.1
cf_target_vw = (vw_carbon["CF"] * 0.5).rename("target_CF").copy()
print("\n[24] CF targets for P^(vw)_oos(0.5):")
print(cf_target_vw.round(3).to_string())


# =============================================================================
# 29. FEASIBILITY CHECK
# =============================================================================
adj_targets_vw = {}
infeas_log_vw = []
for Y in years_part2:
    isins = universe[Y]
    cf_min = min_feasible_cf(isins, Y)   # reused from 3.2
    target_raw = float(cf_target_vw.loc[Y])
    if cf_min > target_raw:
        target_eff = cf_min * 1.001
        infeas_log_vw.append({"Y": Y, "target_raw": target_raw,
                              "cf_min": cf_min, "target_eff": target_eff})
        adj_targets_vw[Y] = target_eff
    else:
        adj_targets_vw[Y] = target_raw

if infeas_log_vw:
    print("\n   ⚠ Infeasibility detected — targets relaxed:")
    print(pd.DataFrame(infeas_log_vw).round(3).to_string(index=False))
else:
    print("\n   ✓ All raw targets feasible (no relaxation needed)")


# =============================================================================
# 30. CONSTRAINED TRACKING-ERROR OPTIMIZER
# =============================================================================
def min_te_cf_constrained(Sigma, w_bench, c, target, isins,
                          warm_start=None):
    """
    Solve:  min  (α − w_bench)' Σ (α − w_bench)
            s.t. α'e = 1,
                 c'α ≤ target,
                 α_i ≥ 0,
                 α_i = 0 if c_i invalid (NaN cap)

    Choice #22: minimize TE² (smooth, equivalent argmin to TE).
    Choice #25: warm start = w_bench (passive starting point) — infeasible
        when the CF constraint binds, but SLSQP handles infeasibility OK
        with analytic Jacobians.
    """
    N = Sigma.shape[0]
    valid = np.isfinite(c)
    invalid = ~valid

    if warm_start is None:
        w0 = w_bench.copy()
    else:
        w0 = warm_start.copy()
    w0[invalid] = 0.0
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.where(valid, 1.0, 0.0) / max(valid.sum(), 1)

    bounds = [(0.0, 0.0) if invalid[i] else (0.0, 1.0) for i in range(N)]

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0,
                          "jac": lambda w: np.ones(N)},
        {"type": "ineq", "fun": lambda w: target - np.where(valid, c, 0.0) @ w,
                          "jac": lambda w: -np.where(valid, c, 0.0)},
    ]

    res = minimize(
        fun=lambda w: float((w - w_bench) @ Sigma @ (w - w_bench)),
        x0=w0,
        jac=lambda w: 2.0 * (Sigma @ (w - w_bench)),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 3000},
    )
    return res


# =============================================================================
# 31. ROLLING LOOP — P^(vw)_oos(0.5)
# =============================================================================
print("\n[25] Rolling VW(0.5) optimization ...")

vw05_w_dict = {}
vw05_ret = {}
_vw05_drift = {}
solver_log_vw = []

for Y in years_part2:
    eligible = universe[Y]
    target = adj_targets_vw[Y]

    mu, Sig = estimate_cov(eligible, estim_window(Y))
    c, valid = cf_vector(eligible, Y)              # reuse 3.2
    w_bench = vw_weights(Y).reindex(eligible).fillna(0.0).values

    res = min_te_cf_constrained(
        Sig, w_bench, c, target, eligible,
        warm_start=None,  # use w_bench
    )
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 0:
        w = w / w.sum()

    cf_real = float(np.where(valid, c, 0.0) @ w)
    te2 = float((w - w_bench) @ Sig @ (w - w_bench))
    binds = abs(cf_real - target) / max(abs(target), 1e-12) < 1e-3
    violates = cf_real > target * (1 + 1e-6)

    solver_log_vw.append({
        "Y": Y, "ok": bool(res.success), "iter": int(res.nit),
        "ann_TE_pct": np.sqrt(max(te2, 0.0)) * np.sqrt(12) * 100,
        "target": target, "cf_real": cf_real,
        "binds": bool(binds), "violates": bool(violates),
        "n_active": int((w > 1e-6).sum()),
    })

    if violates:
        print(f"   ⚠ Y={Y}: CF constraint violated  ({cf_real:.3f} > {target:.3f})")
    if not res.success:
        print(f"   ⚠ Y={Y}: SLSQP did not converge — {res.message}")

    vw05_w_dict[Y] = pd.Series(w, index=eligible)

    # Next-year monthly returns with drifted weights
    next_months = months_of(Y + 1)
    R_next = fill_oos_returns(eligible, next_months)
    ww = w.copy()
    port_ret = []
    for t in next_months:
        r_t = R_next[t].values
        rp_t = float(ww @ r_t)
        port_ret.append(rp_t)
        ww = ww * (1.0 + r_t) / max(1.0 + rp_t, 1e-12)
    vw05_ret[Y + 1] = pd.Series(port_ret, index=next_months)
    _vw05_drift[Y + 1] = dict(zip(eligible, ww))

rp_vw05 = pd.concat(vw05_ret).droplevel(0).sort_index()
rp_vw05.index = pd.DatetimeIndex(rp_vw05.index)

solver_df_vw = pd.DataFrame(solver_log_vw).set_index("Y")
print("\n   Solver / TE diagnostics:")
print(solver_df_vw.round(3).to_string())


# =============================================================================
# 32. VERIFICATION
# =============================================================================
print("\n[26] Verification — P^(vw)_oos(0.5) ...")

# A. Sum-to-1 / non-negativity
for Y, w in vw05_w_dict.items():
    assert abs(w.sum() - 1.0) < 1e-6, f"Y={Y}: weights sum {w.sum()}"
    assert (w >= -1e-8).all(), f"Y={Y}: negative weights"
print("   ✓ Weights ∈ [0,1], sum to 1")

# B. CF constraint
vw05_carbon = pd.DataFrame(
    [{"Y": Y, **carbon_metrics(vw05_w_dict[Y], Y)} for Y in years_part2]
).set_index("Y")
cf_check_vw = pd.DataFrame({
    "target_eff":  pd.Series(adj_targets_vw),
    "CF_realized": vw05_carbon["CF"],
    "WACI_realized": vw05_carbon["WACI"],
})
cf_check_vw["slack"] = cf_check_vw["target_eff"] - cf_check_vw["CF_realized"]
print("\n   CF realized vs target:")
print(cf_check_vw.round(3).to_string())
violations = cf_check_vw[cf_check_vw["CF_realized"] > cf_check_vw["target_eff"] * 1.001]
print(f"   {'⚠ ' + str(len(violations)) + ' violations' if len(violations) else '✓ CF constraint satisfied (within 0.1%) every year'}")

# C. Identity check
print("\n   Cross-check: optimizer c'α vs carbon_metrics CF")
max_diff = 0.0
for Y in years_part2:
    c, valid = cf_vector(universe[Y], Y)
    w = vw05_w_dict[Y].values
    cf_opt = float(np.where(valid, c, 0.0) @ w)
    cf_rep = float(vw05_carbon.loc[Y, "CF"])
    max_diff = max(max_diff, abs(cf_opt - cf_rep))
print(f"   max |c'α − carbon_metrics.CF| = {max_diff:.3e}")
assert max_diff < 1e-8

# D. TE properties:
#    (i) ex-ante TE > 0 everywhere (constraint binds when target < VW CF)
#   (ii) ex-ante TE(VW(0.5)) << ex-ante TE(MV(0.5)) [TE-min vs MV-min]
print("\n   D. Tracking-error sanity")
te_compare = []
for Y in years_part2:
    eligible = universe[Y]
    _, Sig = estimate_cov(eligible, estim_window(Y))
    w_bench = vw_weights(Y).reindex(eligible).fillna(0.0).values
    w_vw05  = vw05_w_dict[Y].values
    w_mv05  = mv05_w_dict[Y].reindex(eligible).fillna(0.0).values
    te_vw05 = np.sqrt(max((w_vw05 - w_bench) @ Sig @ (w_vw05 - w_bench), 0)) * np.sqrt(12) * 100
    te_mv05 = np.sqrt(max((w_mv05 - w_bench) @ Sig @ (w_mv05 - w_bench), 0)) * np.sqrt(12) * 100
    te_compare.append({"Y": Y, "TE_VW05_%": te_vw05, "TE_MV05_%": te_mv05})
te_compare_df = pd.DataFrame(te_compare).set_index("Y")
print(te_compare_df.round(3).to_string())
if (te_compare_df["TE_VW05_%"] >= te_compare_df["TE_MV05_%"] - 1e-8).any():
    print("   ⚠ Some years have TE(VW05) ≥ TE(MV05) — surprising; investigate")
else:
    print("   ✓ TE(VW05) < TE(MV05) every year (as expected: VW05 is the TE-minimizer)")

# E. Ex-post TE: empirical realized tracking error
te_ep = (rp_vw05.dropna() - rp_vw.dropna()).std() * np.sqrt(12) * 100
print(f"\n   E. Realized (ex-post) annualized TE of VW(0.5) vs VW = {te_ep:.2f}%")
print(f"      Mean ex-ante TE = {te_compare_df['TE_VW05_%'].mean():.2f}%  "
      f"(in-line with ex-post is a plausibility check)")


# =============================================================================
# 33. PERFORMANCE COMPARISON
# =============================================================================
print("\n[27] Performance — VW vs VW(0.5) ...")

stats33 = pd.DataFrame([
    compute_perf(rp_vw,    rf_mon, "P^(vw) (benchmark)"),
    compute_perf(rp_vw05,  rf_mon, "P^(vw)_oos(0.5)"),
    compute_perf(rp_mv05,  rf_mon, "P^(mv)_oos(0.5) [reference]"),
]).set_index("Portfolio")
print(stats33.to_string())

# Information ratio for VW(0.5)
ar = (rp_vw05 - rp_vw).dropna()
ir = ar.mean() * 12 / (ar.std() * np.sqrt(12)) if ar.std() > 0 else np.nan
print(f"\n   Information Ratio (VW(0.5) vs VW) = {ir:.3f}")
print(f"   Annualized active return = {ar.mean()*12*100:.2f}%")
print(f"   Annualized realized TE   = {te_ep:.2f}%")


# =============================================================================
# 34. COMPOSITION SHIFTS — VW(0.5) vs VW
# =============================================================================
print("\n[28] Composition shifts — VW(0.5) vs VW ...")

def composition_diff_vw(Y, top_n=5):
    isins = universe[Y]
    w_vw  = vw_weights(Y).reindex(isins).fillna(0.0)
    w_vw5 = vw05_w_dict[Y].reindex(isins).fillna(0.0)
    diff = (w_vw5 - w_vw).sort_values()
    drops = diff.head(top_n)
    adds  = diff.tail(top_n).iloc[::-1]

    def name(i): return isin_name.get(i, i)
    def CI(i):
        if (i in co2_tot.index and i in rev_m.index
            and pd.notna(rev_m.loc[i, Y]) and rev_m.loc[i, Y] > 0):
            return float(co2_tot.loc[i, Y] / rev_m.loc[i, Y])
        return np.nan

    print(f"\n   --- Y={Y} ---")
    print(f"   Most REMOVED (Δ weight, pp):")
    for isin, dlt in drops.items():
        ci_i = CI(isin)
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")
    print(f"   Most ADDED (Δ weight, pp):")
    for isin, dlt in adds.items():
        ci_i = CI(isin)
        print(f"     {name(isin)[:40]:<40s}  Δ = {dlt*100:+6.2f} pp   "
              f"CI = {ci_i if pd.isna(ci_i) else round(ci_i,0):>6}")

for Y in [2013, 2018, 2024]:
    composition_diff_vw(Y, top_n=5)


# =============================================================================
# 35. PLOTS
# =============================================================================
print("\n[29] Plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section 3.3 — VW vs VW(0.5)", fontsize=13, fontweight="bold")
C_VW, C_VW5, C_MV5 = "darkorange", "seagreen", "crimson"
fmt = mdates.DateFormatter("%Y"); loc = mdates.YearLocator(2)

# Cumulative
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

# Active return (VW(0.5) − VW)
ax = axes[0, 1]
ar_cum = (1 + ar).cumprod()
ax.plot(ar_cum.index, (ar_cum - 1) * 100, color=C_VW5, lw=1.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_title(f"Cumulative active return: VW(0.5) − VW (IR={ir:.2f})")
ax.set_ylabel("Cumulative active return (%)")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# WACI
ax = axes[1, 0]
ax.plot(vw_carbon.index,   vw_carbon["WACI"],   "x--", color=C_VW,  label="VW")
ax.plot(vw05_carbon.index, vw05_carbon["WACI"], "s-",  color=C_VW5, label="VW(0.5)")
ax.set_title("WACI evolution"); ax.set_ylabel("tCO₂ / M$ rev")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# CF
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

print("\n[Section 3.3 complete] — proceed to 4.1: Net-Zero glide path")
