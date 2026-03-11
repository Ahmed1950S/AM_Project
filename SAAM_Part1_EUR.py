# =============================================================================
# SAAM Project 2026 — Part I: Standard Portfolio Allocation — EUR region
# =============================================================================
import pandas as pd, numpy as np, pickle, warnings
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

REGION=      "EUR"
START_YEAR=  2013
END_YEAR=    2024
ESTIM_YEARS= 10
MIN_OBS=     36      # min valid monthly obs in estimation window
STALE_THR=   0.50    # max zero-return fraction
LOW_FLOOR=   0.50    # RI values below this → NaN
ALPHA_REG=   1e-4    # Ledoit-style diagonal regularisation for Σ (improves conditioning)
CACHE=       "/home/claude/"
OUT=         "/mnt/user-data/outputs/"

print("="*65)
print("SAAM Part I — Minimum Variance Portfolio (EUR region)")
print("="*65)

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
print("\n[1] Loading ...")
def pkl(n):
    with open(f"{CACHE}{n}.pkl","rb") as f: return pickle.load(f)

static=pkl("static"); ri_y=pkl("ri_y"); ri_m=pkl("ri_m")
mv_y=pkl("mv_y");     mv_m=pkl("mv_m"); co2s1=pkl("co2s1")
co2s2=pkl("co2s2");   rev=pkl("rev");   rf_raw=pkl("rf")

# ── 2. RF ─────────────────────────────────────────────────────────────────────
rf_raw.columns=["YYYYMM","RF_pct"]
rf_raw["date"]=pd.to_datetime(rf_raw["YYYYMM"].astype(str),format="%Y%m")
rf_mon=(1+rf_raw.set_index("date")["RF_pct"]/100)**(1/12)-1

# ── 3. HELPER ─────────────────────────────────────────────────────────────────
def load_clean(df, valid_isins):
    ic=df["ISIN"].dropna()
    out=df.loc[ic.index[ic.isin(valid_isins)]].copy()
    out.index=out["ISIN"]; out.index.name="ISIN"
    out=out.drop(columns=["ISIN"]).apply(pd.to_numeric,errors="coerce")
    out.columns=[pd.Timestamp(c) if hasattr(c,"year") and
                 not isinstance(c,(int,np.integer)) else c for c in out.columns]
    return out

# ── 4. EUR FILTER ─────────────────────────────────────────────────────────────
print(f"\n[2] Filtering EUR ...")
eur_isins=set(static.loc[static["Region"]==REGION,"ISIN"])
ri_y_e=load_clean(ri_y,eur_isins);   ri_m_e=load_clean(ri_m,eur_isins)
mv_y_e=load_clean(mv_y,eur_isins);   mv_m_e=load_clean(mv_m,eur_isins)
co2s1_e=load_clean(co2s1,eur_isins); co2s2_e=load_clean(co2s2,eur_isins)
rev_e=load_clean(rev,eur_isins)
print(f"   {ri_m_e.shape[0]} firms × {ri_m_e.shape[1]} monthly cols")

# ── 5. DATE LISTS ─────────────────────────────────────────────────────────────
monthly_all=sorted([c for c in ri_m_e.columns if isinstance(c,pd.Timestamp)
    and pd.Timestamp("2000-01-01")<=c<=pd.Timestamp("2025-12-31")])
annual_all=sorted([c for c in ri_y_e.columns if isinstance(c,(int,np.integer))])

ri_m_e=ri_m_e[monthly_all]; mv_m_e=mv_m_e[[c for c in monthly_all if c in mv_m_e.columns]]
ri_y_e=ri_y_e[[c for c in annual_all if c in ri_y_e.columns]]
mv_y_e=mv_y_e[[c for c in annual_all if c in mv_y_e.columns]]
co2s1_e=co2s1_e[[c for c in annual_all if c in co2s1_e.columns]]
co2s2_e=co2s2_e[[c for c in annual_all if c in co2s2_e.columns]]
rev_e=rev_e[[c for c in annual_all if c in rev_e.columns]]

def months_of(Y): return [c for c in monthly_all if c.year==Y]
def estim_window(Y):
    return [c for c in monthly_all
            if pd.Timestamp(f"{Y-ESTIM_YEARS+1}-01-01")<=c<=pd.Timestamp(f"{Y}-12-31")]

# ── 6. CLEAN ──────────────────────────────────────────────────────────────────
print("\n[3] Cleaning ...")
ri_m_e[ri_m_e<LOW_FLOOR]=np.nan; ri_y_e[ri_y_e<LOW_FLOOR]=np.nan

cutoff=pd.Timestamp("2025-12-31")
last_valid={}
for isin in ri_m_e.index:
    lv=ri_m_e.loc[isin].last_valid_index()
    if lv is not None and isinstance(lv,pd.Timestamp) and lv<cutoff:
        last_valid[isin]=lv
print(f"   Delisted: {len(last_valid)}")

ret_m=ri_m_e.pct_change(axis=1)
for isin,ddate in last_valid.items():
    idx=monthly_all.index(ddate)
    if idx+1<len(monthly_all): ret_m.at[isin,monthly_all[idx+1]]=-1.0
ret_m=ret_m.iloc[:,1:]

co2s1_e=co2s1_e.ffill(axis=1); co2s2_e=co2s2_e.ffill(axis=1); rev_e=rev_e.ffill(axis=1)

# ── 7. UNIVERSE ───────────────────────────────────────────────────────────────
def get_universe(Y):
    win=estim_window(Y); R_win=ret_m[[c for c in win if c in ret_m.columns]]
    out=[]
    for isin in ri_y_e.index:
        if Y not in ri_y_e.columns or pd.isna(ri_y_e.at[isin,Y]): continue
        s1=Y in co2s1_e.columns and not pd.isna(co2s1_e.at[isin,Y])
        s2=Y in co2s2_e.columns and not pd.isna(co2s2_e.at[isin,Y])
        if not(s1 or s2): continue
        if isin not in R_win.index: continue
        row=R_win.loc[isin]
        if row.notna().sum()<MIN_OBS: continue
        if (row==0).sum()/len(win)>=STALE_THR: continue
        out.append(isin)
    return out

# ── 8. COVARIANCE ─────────────────────────────────────────────────────────────
def estimate_cov(isins, win_cols):
    win_in=[c for c in win_cols if c in ret_m.columns]
    R=ret_m.loc[isins,win_in].fillna(0).values; T=R.shape[1]
    mu=R.mean(axis=1); Rd=R-mu[:,None]
    Sig=(Rd@Rd.T)/T
    # Diagonal regularisation: Σ_reg = (1-α)Σ + α·diag(Σ)
    # Shrinks toward diagonal, improves conditioning, no look-ahead bias
    Sig_reg=(1-ALPHA_REG)*Sig+ALPHA_REG*np.diag(np.diag(Sig))
    return mu, Sig_reg

# ── 9. MIN-VAR ────────────────────────────────────────────────────────────────
# Warm-start weights (equal weight or previous year weights projected)
_prev_w = {}

def min_var_weights(Sigma, isins, Y):
    N=Sigma.shape[0]
    # Warm start: use prev year weights where ISIN overlaps, else equal weight
    if Y-1 in _prev_w:
        prev=_prev_w[Y-1]; w0=np.array([prev.get(i,0.) for i in isins])
        w0=w0/w0.sum() if w0.sum()>0 else np.ones(N)/N
    else:
        w0=np.ones(N)/N

    res=minimize(lambda w:float(w@Sigma@w), w0,
                 jac=lambda w:2.*(Sigma@w), method="SLSQP",
                 bounds=[(0.,None)]*N,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1.},
                 options={"ftol":1e-10,"maxiter":1000})
    _prev_w[Y]=dict(zip(isins,res.x))
    return res.x

# ── 10. ROLLING LOOP ──────────────────────────────────────────────────────────
print("\n[4] Rolling optimisation ...")
mv_w_dict={}; mv_ret={}; univ={}

for Y in range(START_YEAR, END_YEAR+1):
    eligible=get_universe(Y); univ[Y]=eligible; N=len(eligible)
    print(f"   Y={Y}: {N:3d} firms",end="")
    if N==0: print("  → skip"); continue

    mu,Sig=estimate_cov(eligible,estim_window(Y))
    w=min_var_weights(Sig,eligible,Y)
    mv_w_dict[Y]=pd.Series(w,index=eligible)

    next_m=months_of(Y+1)
    R_next=ret_m.loc[eligible].reindex(columns=next_m).fillna(0)
    ww=w.copy(); port_ret=[]
    for t in next_m:
        r_t=R_next[t].values; rp_t=float(ww@r_t)
        port_ret.append(rp_t)
        ww=ww*(1.+r_t)/max(1.+rp_t,1e-12)
    mv_ret[Y+1]=pd.Series(port_ret,index=next_m)
    print(f"  |  ann.σ={np.sqrt(float(w@Sig@w)*12)*100:.2f}%")

rp_mv=pd.concat(mv_ret).droplevel(0).sort_index()
rp_mv.index=pd.DatetimeIndex(rp_mv.index)

# ── 11. VALUE-WEIGHTED ────────────────────────────────────────────────────────
print("\n[5] Value-weighted benchmark ...")
vw_ret={}
for Y in range(START_YEAR, END_YEAR+1):
    eligible=univ.get(Y,[]); next_m=months_of(Y+1); port=[]
    for t in next_m:
        idx=monthly_all.index(t) if t in monthly_all else None
        if idx is None or idx==0: port.append(np.nan); continue
        prev_t=monthly_all[idx-1]
        cap=mv_m_e.loc[eligible,prev_t].fillna(0) \
            if prev_t in mv_m_e.columns else pd.Series(0.,index=eligible)
        tot=cap.sum()
        if tot<=0: port.append(0.); continue
        r_t=ret_m.loc[eligible,t].fillna(0).values \
            if t in ret_m.columns else np.zeros(len(eligible))
        port.append(float((cap/tot).values@r_t))
    vw_ret[Y+1]=pd.Series(port,index=next_m)

rp_vw=pd.concat(vw_ret).droplevel(0).sort_index()
rp_vw.index=pd.DatetimeIndex(rp_vw.index)

# ── 12. PERFORMANCE ───────────────────────────────────────────────────────────
print("\n[6] Performance stats ...")
def perf(rp,rf_s,label):
    rp=rp.dropna(); rf=rf_s.reindex(rp.index).ffill().fillna(0)
    mu=(1+rp.mean())**12-1; sig=rp.std()*np.sqrt(12)
    SR=(mu-((1+rf.mean())**12-1))/sig
    cum=(1+rp).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    return {"Portfolio":label,"Ann. Return (%)":round(mu*100,2),
            "Ann. Vol (%)":round(sig*100,2),"Sharpe":round(SR,3),
            "Min Mo. (%)":round(rp.min()*100,2),"Max Mo. (%)":round(rp.max()*100,2),
            "Max DD (%)":round(mdd*100,2)}

stats_df=pd.DataFrame([perf(rp_vw,rf_mon,"Val-Wgt P^(vw)"),
                        perf(rp_mv,rf_mon,"Min-Var P_oos^(mv)")]).set_index("Portfolio")
print("\n",stats_df.to_string())

# ── 13. TOP HOLDINGS ──────────────────────────────────────────────────────────
isin_name=static.set_index("ISIN")["NAME"].to_dict()
print("\n[7] Top 10 at end-2024 (Min-Var):")
w24=mv_w_dict.get(2024,pd.Series(dtype=float)).sort_values(ascending=False)
for rk,(i,wt) in enumerate(w24.head(10).items(),1):
    print(f"   {rk:2d}. {isin_name.get(i,i):<42s} {wt*100:6.2f}%")
print(f"   Non-zero: {(w24>1e-6).sum()} / {len(w24)}")

print("\n[8] Universe by year:")
for Y in sorted(univ): print(f"   {Y}: {len(univ[Y]):3d} firms")

# ── 14. FIGURES ───────────────────────────────────────────────────────────────
print("\n[9] Figures ...")
fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle("SAAM Part I — EUR: Min-Var vs Value-Weighted (2014–2025)",
             fontsize=13,fontweight="bold")
C1,C2="steelblue","darkorange"
fmt=mdates.DateFormatter("%Y"); loc=mdates.YearLocator(2)

# Cumulative return
ax=axes[0,0]
cm=(1+rp_mv.dropna()).cumprod(); cv=(1+rp_vw.dropna()).cumprod()
ax.plot(cm.index,cm.values,color=C1,lw=1.8,label="Min-Var $P_{oos}^{(mv)}$")
ax.plot(cv.index,cv.values,color=C2,lw=1.8,ls="--",label="Val-Wgt $P^{(vw)}$")
ax.set_title("Cumulative Return (base=1, Jan 2014)"); ax.set_ylabel("Cum. return")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# Rolling volatility
ax=axes[0,1]
ax.plot(rp_mv.index,(rp_mv.rolling(12).std()*np.sqrt(12)*100).values,color=C1,lw=1.6,label="Min-Var")
ax.plot(rp_vw.index,(rp_vw.rolling(12).std()*np.sqrt(12)*100).values,color=C2,lw=1.6,ls="--",label="Val-Wgt")
ax.set_title("Rolling 12m Ann. Volatility (%)"); ax.set_ylabel("Vol (%)"); ax.legend(fontsize=9)
ax.grid(alpha=0.3); ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# Drawdown
ax=axes[1,0]
def dds(rp):
    c=(1+rp.dropna()).cumprod(); return (c-c.cummax())/c.cummax()*100
ax.fill_between(dds(rp_mv).index,dds(rp_mv).values,0,alpha=0.45,color=C1,label="Min-Var")
ax.fill_between(dds(rp_vw).index,dds(rp_vw).values,0,alpha=0.30,color=C2,label="Val-Wgt")
ax.set_title("Drawdown from Peak (%)"); ax.set_ylabel("Drawdown (%)"); ax.legend(fontsize=9)
ax.grid(alpha=0.3); ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)

# Universe size
ax=axes[1,1]; yrs=sorted(univ.keys())
ax.bar(yrs,[len(univ[y]) for y in yrs],color=C1,alpha=0.8,edgecolor="white",width=0.6)
ax.set_title("EUR Universe Size by Year"); ax.set_ylabel("Eligible firms")
ax.set_xticks(yrs); ax.set_xticklabels(yrs,rotation=45); ax.grid(axis="y",alpha=0.3)

plt.tight_layout()
for ext in ("pdf","png"):
    plt.savefig(f"{OUT}SAAM_Part1_EUR_figures.{ext}",dpi=150,bbox_inches="tight")
plt.close(); print("   Figures saved.")

# ── 15. EXCEL ─────────────────────────────────────────────────────────────────
print("\n[10] Exporting Excel ...")
xlsx=f"{OUT}SAAM_Part1_EUR_results.xlsx"
with pd.ExcelWriter(xlsx,engine="openpyxl") as writer:
    stats_df.to_excel(writer,sheet_name="Summary_Stats")

    ro=pd.DataFrame({"Min-Var":rp_mv,"Value-Weighted":rp_vw})
    ro.index=ro.index.strftime("%Y-%m"); ro.to_excel(writer,sheet_name="Monthly_Returns")

    wd=pd.DataFrame(mv_w_dict).T.fillna(0)
    wd.index.name="Year"; wd.rename(columns=isin_name,inplace=True)
    wd.to_excel(writer,sheet_name="MV_Weights")

    rows=[]
    for Y in sorted(mv_w_dict):
        for rk,(i,wt) in enumerate(mv_w_dict[Y].sort_values(ascending=False).head(10).items(),1):
            cty=static.loc[static["ISIN"]==i,"Country"].values
            rows.append({"Year":Y,"Rank":rk,"ISIN":i,"Name":isin_name.get(i,i),
                         "Country":cty[0] if len(cty) else "","Weight (%)":round(wt*100,3)})
    pd.DataFrame(rows).to_excel(writer,sheet_name="Top10_Holdings",index=False)

print(f"   Saved: {xlsx}")
print("\n"+"="*65+"\nDONE — outputs in /mnt/user-data/outputs/")
