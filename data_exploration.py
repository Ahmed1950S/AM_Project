"""
SAAM Project – Part I: Data Exploration & Cleaning
Region: EUR | CO2 Scope: 1 + 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Adjust path to where your data lives
DATA_PATH = "Data_2026/"

# ============================================================
# CELL 1: Load all raw data
# ============================================================

# Static file (firm metadata)
static = pd.read_excel(DATA_PATH + "Static_2025.xlsx")
print(f"Static: {static.shape}")
print(static.head())

# Monthly total return index (RI) and market value (MV)
ri_m = pd.read_excel(DATA_PATH + "DS_RI_T_USD_M_2025.xlsx", sheet_name="RI")
mv_m = pd.read_excel(DATA_PATH + "DS_MV_T_USD_M_2025.xlsx", sheet_name="MV")

# Yearly: RI, MV, CO2 Scope 1 & 2, Revenue
ri_y = pd.read_excel(DATA_PATH + "DS_RI_T_USD_Y_2025.xlsx", sheet_name="RI")
mv_y = pd.read_excel(DATA_PATH + "DS_MV_T_USD_Y_2025.xlsx", sheet_name="MV")
co2_s1 = pd.read_excel(DATA_PATH + "DS_CO2_SCOPE_1_Y_2025.xlsx", sheet_name="Scope1")
co2_s2 = pd.read_excel(DATA_PATH + "DS_CO2_SCOPE_2_Y_2025.xlsx", sheet_name="Scope2")
rev = pd.read_excel(DATA_PATH + "DS_REV_Y_2025.xlsx", sheet_name="REV")

# Risk-free rate
rf = pd.read_excel(DATA_PATH + "Risk_Free_Rate_2025.xlsx",
                   sheet_name="F-F_Research_Data_Factors")

print(f"\nRI monthly: {ri_m.shape}")
print(f"MV monthly: {mv_m.shape}")
print(f"CO2 Scope1: {co2_s1.shape}")
print(f"CO2 Scope2: {co2_s2.shape}")
print(f"Revenue:    {rev.shape}")
print(f"RF rate:    {rf.shape}")

# ============================================================
# CELL 2: Inspect raw structure and identify problems
# ============================================================

def inspect_df(df, name):
    """Quick inspection of a Datastream-format dataframe."""
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns (first 5): {list(df.columns[:5])}")
    print(f"Columns (last 5):  {list(df.columns[-5:])}")
    
    # Check for rows where ISIN is null (error rows from Datastream)
    null_isin = df['ISIN'].isna().sum()
    print(f"Rows with null ISIN: {null_isin}")
    
    # Check for string values in data columns (Datastream error codes)
    data_cols = [c for c in df.columns if c not in ['NAME', 'ISIN']]
    str_mask = df[data_cols].map(lambda x: isinstance(x, str))
    str_count = str_mask.sum().sum()
    if str_count > 0:
        # Show unique string values
        unique_strs = set()
        for c in data_cols:
            for v in df[c]:
                if isinstance(v, str):
                    unique_strs.add(v[:60])
        print(f"String values in data: {str_count}")
        print(f"Unique strings: {unique_strs}")
    else:
        print(f"String values in data: 0")
    
    # Count None columns (yearly files have trailing Nones from openpyxl)
    none_cols = [c for c in df.columns if c is None]
    print(f"None-named columns: {len(none_cols)}")

for df, name in [(ri_m, "RI Monthly"), (mv_m, "MV Monthly"),
                 (ri_y, "RI Yearly"), (mv_y, "MV Yearly"),
                 (co2_s1, "CO2 Scope 1"), (co2_s2, "CO2 Scope 2"),
                 (rev, "Revenue")]:
    inspect_df(df, name)

# ============================================================
# CELL 3: Clean all dataframes — unified cleaning function
# ============================================================

def clean_datastream_df(df, name=""):
    """
    Standard cleaning for Datastream-format dataframes:
    1. Drop rows with null ISIN (error rows)
    2. Drop columns with None name (trailing empty cols)
    3. Set ISIN as index
    4. Convert string error codes to NaN in data columns
    5. Convert all data columns to float
    """
    df = df.copy()
    
    # 1. Drop error rows (null ISIN)
    n_before = len(df)
    df = df.dropna(subset=['ISIN'])
    # Also drop rows where ISIN itself is an error string
    df = df[~df['ISIN'].astype(str).str.contains('ER:', na=False)]
    n_dropped = n_before - len(df)
    
    # 2. Drop None-named columns
    df = df[[c for c in df.columns if c is not None]]
    
    # 3. Store NAME separately, set ISIN as index
    names = df.set_index('ISIN')['NAME']
    
    # 4. Get data columns (everything except NAME and ISIN)
    data_cols = [c for c in df.columns if c not in ['NAME', 'ISIN']]
    data = df.set_index('ISIN')[data_cols]
    
    # 5. Replace string error codes with NaN, convert to float
    data = data.apply(pd.to_numeric, errors='coerce')
    
    print(f"  {name}: dropped {n_dropped} error rows, "
          f"{len(data)} firms × {len(data_cols)} periods, "
          f"NaN ratio: {data.isna().mean().mean():.1%}")
    
    return data, names

ri_m_data, firm_names = clean_datastream_df(ri_m, "RI Monthly")
mv_m_data, _          = clean_datastream_df(mv_m, "MV Monthly")
ri_y_data, _          = clean_datastream_df(ri_y, "RI Yearly")
mv_y_data, _          = clean_datastream_df(mv_y, "MV Yearly")
co2_s1_data, _        = clean_datastream_df(co2_s1, "CO2 Scope 1")
co2_s2_data, _        = clean_datastream_df(co2_s2, "CO2 Scope 2")
rev_data, _           = clean_datastream_df(rev, "Revenue")

# ============================================================
# CELL 4: Verify ISIN consistency across all files
# ============================================================

all_isins = [
    ("Static",    set(static['ISIN'])),
    ("RI_M",      set(ri_m_data.index)),
    ("MV_M",      set(mv_m_data.index)),
    ("RI_Y",      set(ri_y_data.index)),
    ("MV_Y",      set(mv_y_data.index)),
    ("CO2_S1",    set(co2_s1_data.index)),
    ("CO2_S2",    set(co2_s2_data.index)),
    ("Revenue",   set(rev_data.index)),
]

ref_set = all_isins[0][1]
print("ISIN consistency check (vs Static file):")
for name, isin_set in all_isins[1:]:
    missing = ref_set - isin_set
    extra = isin_set - ref_set
    match = "✓" if len(missing) == 0 and len(extra) == 0 else "✗"
    print(f"  {match} {name}: missing={len(missing)}, extra={len(extra)}")

# ============================================================
# CELL 5: Filter to EUR region
# ============================================================

eur_isins = static[static['Region'] == 'EUR']['ISIN'].values
print(f"\nEUR firms in static file: {len(eur_isins)}")

# Filter all data to EUR
ri_m_eur = ri_m_data.loc[ri_m_data.index.isin(eur_isins)]
mv_m_eur = mv_m_data.loc[mv_m_data.index.isin(eur_isins)]
ri_y_eur = ri_y_data.loc[ri_y_data.index.isin(eur_isins)]
mv_y_eur = mv_y_data.loc[mv_y_data.index.isin(eur_isins)]
co2_s1_eur = co2_s1_data.loc[co2_s1_data.index.isin(eur_isins)]
co2_s2_eur = co2_s2_data.loc[co2_s2_data.index.isin(eur_isins)]
rev_eur = rev_data.loc[rev_data.index.isin(eur_isins)]
names_eur = firm_names.loc[firm_names.index.isin(eur_isins)]

print(f"RI monthly EUR: {ri_m_eur.shape}")
print(f"MV monthly EUR: {mv_m_eur.shape}")
print(f"CO2 S1 EUR:     {co2_s1_eur.shape}")
print(f"CO2 S2 EUR:     {co2_s2_eur.shape}")
print(f"Revenue EUR:    {rev_eur.shape}")

# Verification: country distribution
eur_static = static[static['Region'] == 'EUR']
print(f"\nCountry distribution (EUR):")
print(eur_static['Country'].value_counts().head(10))

# ============================================================
# CELL 6: Clean monthly RI — low prices and delisted firms
# ============================================================

# --- 6a: Treat prices below 0.5 as missing ---
LOW_PRICE_THRESHOLD = 0.5

ri_clean = ri_m_eur.copy()
low_price_mask = (ri_clean > 0) & (ri_clean < LOW_PRICE_THRESHOLD)
n_low = low_price_mask.sum().sum()
ri_clean[low_price_mask] = np.nan
print(f"Prices set to NaN (below {LOW_PRICE_THRESHOLD}): {n_low}")

# How many firms are affected?
firms_with_low = low_price_mask.any(axis=1).sum()
print(f"Firms with at least one low-price month: {firms_with_low}")

# --- 6b: Forward-fill internal missing prices ---
# Per project instructions: "When the missing value is between two available
# years, just use the number from the previous year."
# For monthly prices: forward-fill NaN values that sit BETWEEN two valid prices.
# We do NOT fill leading NaNs (firm not yet listed) or trailing NaNs (delisted).

ri_before_ffill = ri_clean.copy()

def forward_fill_internal(row):
    """Forward-fill only internal gaps (between first and last valid)."""
    fv = row.first_valid_index()
    lv = row.last_valid_index()
    if fv is None or lv is None:
        return row
    cols = list(row.index)
    start = cols.index(fv)
    end = cols.index(lv)
    # Only ffill the interior slice
    row.iloc[start:end+1] = row.iloc[start:end+1].ffill()
    return row

ri_clean = ri_clean.apply(forward_fill_internal, axis=1)

# Report what changed
n_filled = ri_before_ffill.isna().sum().sum() - ri_clean.isna().sum().sum()
print(f"\nForward-filled internal missing prices: {n_filled} observations")

# Show which firms were affected
firms_with_gaps = []
for isin in ri_clean.index:
    diff = ri_before_ffill.loc[isin].isna().sum() - ri_clean.loc[isin].isna().sum()
    if diff > 0:
        firms_with_gaps.append((isin, names_eur.get(isin, 'N/A'), diff))

print(f"Firms with internal gaps filled: {len(firms_with_gaps)}")
for isin, name, n in sorted(firms_with_gaps, key=lambda x: -x[2]):
    print(f"  {isin} ({str(name)[:40]}): {n} months filled")

print("\nNote: forward-fill means return = 0% during gap months,")
print("      then the return on re-appearance reflects the full accumulated move.")

# --- 6c: Detect delisted firms ---
delisted_firms = names_eur[names_eur.str.contains('DEAD', case=False, na=False)]
print(f"\nDelisted EUR firms: {len(delisted_firms)}")

# Parse delisting dates from names like "CREDIT SUISSE GROUP DEAD - DELIST.14/06/23"
import re

def parse_delist_date(name):
    """Extract delisting date from Datastream name string."""
    # Pattern: DD/MM/YY at end of string
    match = re.search(r'(\d{2})/(\d{2})/(\d{2})\s*$', str(name))
    if match:
        day, month, year = match.groups()
        year_full = 2000 + int(year)
        try:
            return pd.Timestamp(year=year_full, month=int(month), day=int(day))
        except ValueError:
            return None
    return None

delist_dates = {}
for isin, name in delisted_firms.items():
    dt = parse_delist_date(name)
    if dt:
        delist_dates[isin] = dt

print(f"Successfully parsed delisting dates: {len(delist_dates)}")
print("\nSample delisting dates:")
for isin, dt in list(delist_dates.items())[:5]:
    print(f"  {isin} ({names_eur[isin][:40]}): {dt.strftime('%Y-%m-%d')}")

# ============================================================
# CELL 7: Compute monthly returns from cleaned RI
# ============================================================

# Monthly dates from the column headers
monthly_dates = ri_clean.columns  # These are datetime objects from the Excel

# Simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
returns = ri_clean.pct_change(axis=1)

# Drop the first column (Dec 1999) — no return for the first date
returns = returns.iloc[:, 1:]

print(f"Returns shape: {returns.shape}")
print(f"Date range: {returns.columns[0]} to {returns.columns[-1]}")
print(f"NaN ratio in returns: {returns.isna().mean().mean():.1%}")

# --- 7a: Handle delisted firms ---
# When a firm is delisted, the price disappears. The last available return
# should be -100% (total loss). This happens when price goes from valid to NaN.
#
# Logic: if RI is non-NaN at t-1 but NaN at t, and the firm is delisted,
# set R_t = -1 (i.e., -100%)

for isin, delist_dt in delist_dates.items():
    if isin not in returns.index:
        continue
    
    prices = ri_clean.loc[isin]
    # Find the first NaN after the last valid price, within or after delist month
    last_valid_idx = prices.last_valid_index()
    
    if last_valid_idx is None:
        continue
    
    # The month after the last valid price: that's when the -100% hits
    col_list = list(ri_clean.columns)
    last_pos = col_list.index(last_valid_idx)
    
    if last_pos + 1 < len(col_list):
        next_month = col_list[last_pos + 1]
        # Only apply if this is in the returns columns
        if next_month in returns.columns:
            returns.loc[isin, next_month] = -1.0

# Verification: check for -100% returns
n_minus100 = (returns == -1.0).sum().sum()
print(f"\n-100% returns applied (delistings): {n_minus100}")

# --- 7b: Handle returns after delisting ---
# After the -100% return, all subsequent returns should be NaN
# (you can't earn a return on a delisted stock)
for isin in delist_dates:
    if isin not in returns.index:
        continue
    row = returns.loc[isin]
    minus100_cols = row[row == -1.0].index
    if len(minus100_cols) > 0:
        delist_col = minus100_cols[0]
        col_list = list(returns.columns)
        delist_pos = col_list.index(delist_col)
        # Set everything after the -100% to NaN
        if delist_pos + 1 < len(col_list):
            returns.loc[isin, col_list[delist_pos + 1:]] = np.nan

# ============================================================
# CELL 8: Return sanity checks
# ============================================================

print("="*50)
print("RETURN DISTRIBUTION DIAGNOSTICS")
print("="*50)

# Basic stats (excluding NaN)
flat_rets = returns.values.flatten()
flat_rets = flat_rets[~np.isnan(flat_rets)]
print(f"Total return observations: {len(flat_rets):,}")
print(f"Mean monthly return:  {np.mean(flat_rets):.4f}")
print(f"Std monthly return:   {np.std(flat_rets):.4f}")
print(f"Min return:           {np.min(flat_rets):.4f}")
print(f"Max return:           {np.max(flat_rets):.4f}")
print(f"Returns < -50%:       {(flat_rets < -0.5).sum()}")
print(f"Returns > +100%:      {(flat_rets > 1.0).sum()}")
print(f"Returns == -100%:     {(flat_rets == -1.0).sum()}")

# Plot return distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram (clipped for visibility)
axes[0].hist(np.clip(flat_rets, -0.5, 0.5), bins=200, edgecolor='none', alpha=0.7)
axes[0].set_title("Distribution of Monthly Returns (clipped to ±50%)")
axes[0].set_xlabel("Return")
axes[0].set_ylabel("Frequency")
axes[0].axvline(0, color='red', linewidth=0.5, linestyle='--')

# Box plot of extreme returns by year
# Sample: annual return stats
yearly_groups = {}
for col in returns.columns:
    year = col.year if hasattr(col, 'year') else int(str(col)[:4])
    if year not in yearly_groups:
        yearly_groups[year] = []
    yearly_groups[year].extend(returns[col].dropna().tolist())

years = sorted(yearly_groups.keys())
yearly_stds = [np.std(yearly_groups[y]) for y in years]
axes[1].bar(years, yearly_stds, color='steelblue', alpha=0.7)
axes[1].set_title("Monthly Return Volatility by Year (EUR)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Std Dev of Monthly Returns")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("returns_diagnostics.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: returns_diagnostics.png")

# ============================================================
# CELL 9: Missing value patterns — RI (before vs after forward-fill)
# ============================================================

# Count internal gaps BEFORE forward-fill
def count_internal_missing(row):
    fv = row.first_valid_index()
    lv = row.last_valid_index()
    if fv is None or lv is None:
        return np.nan
    cols = list(row.index)
    start = cols.index(fv)
    end = cols.index(lv)
    interior = row.iloc[start:end+1]
    return interior.isna().sum()

internal_missing_before = ri_before_ffill.apply(count_internal_missing, axis=1)
internal_missing_after = ri_clean.apply(count_internal_missing, axis=1)

first_valid = ri_clean.apply(lambda row: row.first_valid_index(), axis=1)

print("Missing value analysis (EUR, monthly RI):")
print(f"  Firms with no valid prices at all: {first_valid.isna().sum()}")
print(f"\n  Internal gaps BEFORE forward-fill:")
print(f"    0 gaps:   {(internal_missing_before == 0).sum()}")
print(f"    1-5 gaps: {((internal_missing_before > 0) & (internal_missing_before <= 5)).sum()}")
print(f"    6+ gaps:  {(internal_missing_before > 5).sum()}")
print(f"\n  Internal gaps AFTER forward-fill:")
print(f"    0 gaps:   {(internal_missing_after == 0).sum()}")
print(f"    Any gaps: {(internal_missing_after > 0).sum()}")
print(f"    (Should be 0 — forward-fill closes all internal gaps)")

# Visualize missing data pattern BEFORE forward-fill (to document raw data quality)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Focus on 2004-2025 (the relevant period)
date_mask = [c for c in ri_before_ffill.columns 
             if hasattr(c, 'year') and c.year >= 2004]

for ax, (data, title) in zip(axes, [
    (ri_before_ffill[date_mask], "BEFORE Forward-Fill"),
    (ri_clean[date_mask], "AFTER Forward-Fill"),
]):
    presence = (~data.isna()).astype(int)
    first_appear = presence.apply(lambda row: row.values.argmax() 
                                   if row.any() else len(row), axis=1)
    presence_sorted = presence.loc[first_appear.sort_values().index]
    ax.imshow(presence_sorted.values, aspect='auto', cmap='Blues',
              interpolation='none')
    ax.set_title(f"Data Availability: EUR Monthly RI (2004-2025)\n{title}")
    ax.set_xlabel("Month index (from Jan 2004)")
    ax.set_ylabel(f"Firms (n={len(presence_sorted)})")

plt.tight_layout()
plt.savefig("missing_pattern_ri.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: missing_pattern_ri.png")

# ============================================================
# CELL 10: CO2 data — forward-fill + exploration
# ============================================================

print("="*50)
print("CO2 EMISSIONS ANALYSIS (EUR)")
print("="*50)

# Per project instructions (p.3): "When the missing value is between two
# available years or at the end of the sample, just use the number from
# the previous year." This applies to CO2 and revenue.

# --- 10a: Show RAW coverage before forward-fill ---
co2_raw_total = co2_s1_eur + co2_s2_eur  # NaN + anything = NaN
co2_raw_years = [c for c in co2_raw_total.columns if isinstance(c, (int, np.integer))]
co2_raw_yr = co2_raw_total[co2_raw_years]

print("\nCO2 coverage BEFORE forward-fill (raw S1+S2):")
for y in [2013, 2018, 2023, 2024]:
    if y in co2_raw_yr.columns:
        n = co2_raw_yr[y].notna().sum()
        print(f"  {y}: {n}/{len(co2_raw_yr)} ({100*n/len(co2_raw_yr):.1f}%)")

# --- 10b: Forward-fill CO2 and revenue ---
co2_s1_eur = co2_s1_eur.ffill(axis=1)
co2_s2_eur = co2_s2_eur.ffill(axis=1)
rev_eur = rev_eur.ffill(axis=1)

co2_total = co2_s1_eur + co2_s2_eur
co2_years = [c for c in co2_total.columns if isinstance(c, (int, np.integer))]
co2_total_yr = co2_total[co2_years]

print("\nCO2 coverage AFTER forward-fill (S1+S2):")
for y in [2013, 2018, 2023, 2024]:
    if y in co2_total_yr.columns:
        n = co2_total_yr[y].notna().sum()
        n_raw = co2_raw_yr[y].notna().sum() if y in co2_raw_yr.columns else 0
        print(f"  {y}: {n}/{len(co2_total_yr)} ({100*n/len(co2_total_yr):.1f}%) "
              f"[+{n - n_raw} from forward-fill]")

# --- 10c: Full coverage table ---
print("\nFull CO2 coverage by year (after forward-fill):")
for y in range(2010, 2025):
    if y in co2_total_yr.columns:
        n = co2_total_yr[y].notna().sum()
        print(f"  {y}: {n}/{len(co2_total_yr)} ({100*n/len(co2_total_yr):.1f}%)")

# Firms that have Scope 1 but NOT Scope 2 (or vice versa) — after forward-fill
for y in [2013, 2018, 2023]:
    if y in co2_s1_eur.columns and y in co2_s2_eur.columns:
        has_s1 = co2_s1_eur[y].notna()
        has_s2 = co2_s2_eur[y].notna()
        s1_only = (has_s1 & ~has_s2).sum()
        s2_only = (~has_s1 & has_s2).sum()
        both = (has_s1 & has_s2).sum()
        print(f"\n  {y}: S1 only={s1_only}, S2 only={s2_only}, Both={both}")

# Distribution of total emissions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y in zip(axes, [2013, 2023]):
    if y in co2_total_yr.columns:
        vals = co2_total_yr[y].dropna()
        ax.hist(np.log10(vals[vals > 0] + 1), bins=50, 
                edgecolor='none', alpha=0.7, color='darkgreen')
        ax.set_title(f"log10(Scope1 + Scope2) Distribution, {y}\n(n={len(vals)})")
        ax.set_xlabel("log10(tonnes CO2)")
        ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("co2_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: co2_distribution.png")

# Zero emissions
for y in range(2013, 2025):
    if y in co2_total_yr.columns:
        n_zero = (co2_total_yr[y] == 0).sum()
        if n_zero > 0:
            firms_zero = co2_total_yr.index[co2_total_yr[y] == 0]
            print(f"\n  Zero total emissions in {y}: {n_zero} firms")
            for isin in firms_zero:
                print(f"    {isin} ({names_eur.get(isin, 'N/A')[:40]})")

# ============================================================
# CELL 11: Revenue exploration — watch for negatives
# ============================================================

print("\n" + "="*50)
print("REVENUE ANALYSIS (EUR)")
print("="*50)

# Revenue was forward-filled above (same as CO2)
rev_years = [c for c in rev_eur.columns if isinstance(c, (int, np.integer))]
rev_yr = rev_eur[rev_years]

# Negative revenue
for y in range(2013, 2025):
    if y in rev_yr.columns:
        neg_mask = rev_yr[y] < 0
        if neg_mask.sum() > 0:
            print(f"\n  Negative revenue in {y}:")
            for isin in rev_yr.index[neg_mask]:
                val = rev_yr.loc[isin, y]
                print(f"    {isin} ({names_eur.get(isin, 'N/A')[:40]}): "
                      f"{val:,.0f} (thousands USD)")

# Revenue coverage
print(f"\nRevenue coverage (2013-2024):")
for y in range(2013, 2025):
    if y in rev_yr.columns:
        n = rev_yr[y].notna().sum()
        print(f"  {y}: {n}/{len(rev_yr)}")

# ============================================================
# CELL 12: Market capitalization exploration
# ============================================================

print("\n" + "="*50)
print("MARKET CAP ANALYSIS (EUR)")
print("="*50)

# Quick look at MV distribution at end-2013
mv_dates = [c for c in mv_m_eur.columns if hasattr(c, 'year')]
dec_2013_cols = [c for c in mv_dates if c.year == 2013 and c.month == 12]
if dec_2013_cols:
    dec2013_mv = mv_m_eur[dec_2013_cols[0]].dropna()
    print(f"MV at Dec 2013: {len(dec2013_mv)} firms with data")
    print(f"  Min:    ${dec2013_mv.min():,.0f}M")
    print(f"  Median: ${dec2013_mv.median():,.0f}M")
    print(f"  Mean:   ${dec2013_mv.mean():,.0f}M")
    print(f"  Max:    ${dec2013_mv.max():,.0f}M")
    
    # Top 10 by market cap
    top10 = dec2013_mv.nlargest(10)
    print(f"\n  Top 10 EUR firms by MV (Dec 2013):")
    for isin, mv in top10.items():
        print(f"    {isin} ({names_eur.get(isin, 'N/A')[:35]}): ${mv:,.0f}M")

# ============================================================
# CELL 13: Risk-free rate processing
# ============================================================

print("\n" + "="*50)
print("RISK-FREE RATE")
print("="*50)

# RF is in the format: YYYYMM as integer, RF as annual rate in percent
# E.g., 0.41 means 0.41% per year for that month
rf.columns = ['date_int', 'RF']
rf = rf.dropna(subset=['date_int'])
rf['date_int'] = rf['date_int'].astype(int)

# Convert to proper datetime
rf['date'] = pd.to_datetime(rf['date_int'].astype(str), format='%Y%m')
# Move to end of month to match return dates
rf['date'] = rf['date'] + pd.offsets.MonthEnd(0)
rf = rf.set_index('date')

# RF is an annual rate in percent → convert to monthly decimal
# Annual rate (decimal) = RF / 100
# Monthly rate = annual rate / 12
rf['RF_annual'] = rf['RF'] / 100.0        # annual decimal
rf['RF_monthly'] = rf['RF_annual'] / 12.0  # monthly decimal

print(f"RF range: {rf.index.min()} to {rf.index.max()}")
print(f"\nRF sample (2014):")
print(rf.loc['2014'][['RF', 'RF_annual', 'RF_monthly']].head())
print(f"\nMean annual RF (2014-2025): {rf.loc['2014':'2025']['RF_annual'].mean():.4f}")
print(f"Mean monthly RF (2014-2025): {rf.loc['2014':'2025']['RF_monthly'].mean():.6f}")

# ============================================================
# CELL 14: Stale price detection
# ============================================================

print("\n" + "="*50)
print("STALE PRICE DETECTION (EUR)")
print("="*50)

STALE_THRESHOLD = 0.3  # >30% zero returns = stale

def get_ret_cols(returns_df, y_start, m_start, y_end, m_end):
    """Select return columns by year/month to avoid business-day date mismatches."""
    return [c for c in returns_df.columns
            if hasattr(c, 'year') and
            (y_start, m_start) <= (c.year, c.month) <= (y_end, m_end)]

def detect_stale(returns_df, Y):
    """
    For each firm, compute the fraction of zero returns 
    in the 10-year window ending Dec Y. Returns (frac_zero, n_valid).
    """
    cols = get_ret_cols(returns_df, Y - 9, 1, Y, 12)
    window_rets = returns_df[cols]
    
    # Count zero returns (where return is not NaN)
    n_valid = window_rets.notna().sum(axis=1)
    n_zero = ((window_rets == 0) | (window_rets.abs() < 1e-10)).sum(axis=1)
    
    # Fraction of zero returns among valid returns
    frac_zero = n_zero / n_valid.replace(0, np.nan)
    
    return frac_zero, n_valid

# Check first window: Jan 2004 - Dec 2013
frac_zero, n_valid = detect_stale(returns, 2013)
test_cols = get_ret_cols(returns, 2004, 1, 2013, 12)
print(f"Estimation window check: {len(test_cols)} months (expect 120)")

stale_firms = frac_zero[frac_zero > STALE_THRESHOLD].dropna()
print(f"Firms with >{STALE_THRESHOLD:.0%} zero returns (2004-2013): {len(stale_firms)}")
if len(stale_firms) > 0:
    for isin in stale_firms.index:
        print(f"  {isin} ({names_eur.get(isin, 'N/A')[:40]}): "
              f"{frac_zero[isin]:.1%} zero, {n_valid[isin]} valid obs")

# Also check: firms with very few observations
low_obs = n_valid[n_valid < 36]
print(f"\nFirms with <36 valid returns in 2004-2013: {len(low_obs)}")
for isin in low_obs.index[:10]:
    print(f"  {isin} ({names_eur.get(isin, 'N/A')[:40]}): {n_valid[isin]} returns")

# ============================================================
# CELL 15: Investment set construction (preview for Dec 2013)
# ============================================================

print("\n" + "="*50)
print("INVESTMENT SET CONSTRUCTION: Dec 2013 (first allocation)")
print("="*50)

Y = 2013
ESTIMATION_MONTHS = 120  # 10 years
MIN_RETURNS = 36         # at least 3 years of data

# Get December of year Y for prices
dec_Y_cols = [c for c in ri_clean.columns 
              if hasattr(c, 'year') and c.year == Y and c.month == 12]
if not dec_Y_cols:
    print("ERROR: Dec 2013 column not found!")
else:
    dec_Y = dec_Y_cols[0]
    
    # All EUR ISINs as starting point
    candidates = set(ri_clean.index)
    print(f"Starting candidates: {len(candidates)}")
    
    # Filter 1: must have a valid price at end of year Y
    has_price = set(ri_clean[ri_clean[dec_Y].notna()].index)
    excluded_no_price = candidates - has_price
    candidates = candidates & has_price
    print(f"After price filter (end {Y}): {len(candidates)} "
          f"(removed {len(excluded_no_price)})")
    
    # Filter 2: sufficient return observations in estimation window
    ret_cols = get_ret_cols(returns, Y - 9, 1, Y, 12)
    n_valid_rets = returns.loc[list(candidates), ret_cols].notna().sum(axis=1)
    has_enough = set(n_valid_rets[n_valid_rets >= MIN_RETURNS].index)
    excluded_insuff = candidates - has_enough
    candidates = candidates & has_enough
    print(f"After min returns filter (>={MIN_RETURNS}): {len(candidates)} "
          f"(removed {len(excluded_insuff)})")
    
    # Filter 3: no stale prices
    frac_zero_cand = frac_zero.loc[frac_zero.index.isin(candidates)]
    stale = set(frac_zero_cand[frac_zero_cand > STALE_THRESHOLD].index)
    candidates = candidates - stale
    print(f"After stale price filter (>{STALE_THRESHOLD:.0%}): {len(candidates)} "
          f"(removed {len(stale)})")
    
    # Filter 4: CO2 Scope 1 + Scope 2 both available at end of year Y
    # Per project instructions (Section 2.1): "exclude firms without carbon
    # data available at the end of year Y" so that Parts 1 and 2 use the
    # same investment set and performance comparisons are apples-to-apples.
    has_co2 = set(co2_total_yr[co2_total_yr[Y].notna()].index) & candidates
    excluded_co2 = candidates - has_co2
    candidates = has_co2
    print(f"After CO2 filter (S1+S2 at {Y}): {len(candidates)} "
          f"(removed {len(excluded_co2)})")
    
    # NOTE: No revenue filter. Revenue is only needed to compute WACI
    # (a reporting metric), not the carbon footprint constraint used in
    # Part 2 optimization. Negative revenue firms (6 firm-years in EUR)
    # will be handled when displaying WACI, not by excluding them here.
    
    print(f"\n>>> FINAL INVESTMENT SET (Dec {Y}): {len(candidates)} firms")

# ============================================================
# CELL 16: Summary of investment set across all years
# ============================================================

print("\n" + "="*50)
print("INVESTMENT SET SIZE: ALL YEARS (2013-2024)")
print("="*50)

investment_sets = {}

for Y in range(2013, 2025):
    # Price at end of Y
    dec_Y_cols = [c for c in ri_clean.columns 
                  if hasattr(c, 'year') and c.year == Y and c.month == 12]
    if not dec_Y_cols:
        continue
    dec_Y = dec_Y_cols[0]
    
    candidates = set(ri_clean.index)
    
    # Filter 1: price at end of Y
    candidates &= set(ri_clean[ri_clean[dec_Y].notna()].index)
    
    # Filter 2: sufficient returns in 10-year window
    ret_cols = get_ret_cols(returns, Y - 9, 1, Y, 12)
    n_valid_rets = returns.loc[list(candidates), ret_cols].notna().sum(axis=1)
    candidates &= set(n_valid_rets[n_valid_rets >= MIN_RETURNS].index)
    
    # Filter 3: stale prices in the 10-year window
    fz, nv = detect_stale(returns, Y)
    stale = set(fz[fz > STALE_THRESHOLD].dropna().index)
    candidates -= stale
    
    # Filter 4: CO2 S1+S2 available (same investment set for Parts 1 & 2)
    if Y in co2_total_yr.columns:
        candidates &= set(co2_total_yr[co2_total_yr[Y].notna()].index)
    
    investment_sets[Y] = sorted(candidates)
    print(f"  {Y}: {len(candidates)} firms")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
years = sorted(investment_sets.keys())
sizes = [len(investment_sets[y]) for y in years]
ax.bar(years, sizes, color='steelblue', alpha=0.8)
ax.set_title("Investment Set Size by Year (EUR, with CO2 filter, no revenue filter)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Firms")
for y, s in zip(years, sizes):
    ax.text(y, s + 5, str(s), ha='center', fontsize=9)
ax.set_ylim(0, max(sizes) * 1.15)

plt.tight_layout()
plt.savefig("investment_set_size.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: investment_set_size.png")

# ============================================================
# CELL 17: Final data quality summary
# ============================================================

print("\n" + "="*60)
print("DATA QUALITY SUMMARY")
print("="*60)

print(f"""
Region: EUR
Total firms in static file: {len(eur_isins)}
Data period: Dec 1999 - Jan 2026 (monthly)
Out-of-sample: Jan 2014 - Dec 2025 (144 months)
Rebalancing: annual (Dec 2013 to Dec 2024 = 12 allocations)

Cleaning applied:
  - Dropped {64} Datastream error rows (null ISIN) per file
  - Converted string error codes to NaN
  - Treated {n_low} price observations < {LOW_PRICE_THRESHOLD} as missing
  - Forward-filled {n_filled} internal missing prices (per project instructions)
  - Forward-filled CO2 and revenue across years (per project instructions p.3)
  - Detected {len(delisted_firms)} delisted firms, applied -100% at delisting
  - Stale price threshold: {STALE_THRESHOLD:.0%} zero returns → 
    {len(stale_firms)} firms excluded (first window)

Investment set filters (same set for Parts 1 & 2, per Section 2.1):
  1. Valid price at end of year Y (monthly RI, Dec Y)
  2. At least {MIN_RETURNS} monthly returns in 10-year estimation window
  3. Stale price exclusion (>{STALE_THRESHOLD:.0%} zero returns)
  4. CO2 Scope 1 + Scope 2 both available at end of year Y (after forward-fill)
  (No revenue filter — revenue only needed for WACI reporting, not optimization)

  First allocation (Dec 2013): {len(investment_sets.get(2013, []))} firms
  Last allocation  (Dec 2024): {len(investment_sets.get(2024, []))} firms

Risk-free rate:
  - Fama-French RF treated as annual rate in percent
  - Converted to monthly: RF_monthly = (RF / 100) / 12

Issues to watch:
  - Covariance matrix: {len(investment_sets.get(2013, []))} firms × 120 obs → singular
  - Negative revenue (6 EUR firm-years): handled at WACI reporting stage
""")
