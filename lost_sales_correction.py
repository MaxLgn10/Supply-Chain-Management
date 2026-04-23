# lost_sales_correction.py
# Hierarchical Lost Sales Correction applied BEFORE forecasting.
# Corrects rows where stockout=True by estimating true demand.
#
# Level 1 — Cross-size anchor (same product × channel × season)
# Level 2 — Cross-channel anchor (same cluster, same product × season)
# Level 3 — Historical sell-through fallback (same product × channel × size)
#
# Usage:
#   from lost_sales_correction import apply_lost_sales_correction
#   df = apply_lost_sales_correction(df)
#   # then use df['units_corrected'] instead of df['units']

import numpy as np
import pandas as pd

# ── Correction parameters ─────────────────────────────────────────────────────
CORR_THRESHOLD       = 0.85  # only use channels with r >= 0.85 as anchors
MIN_ANCHOR_UNITS     = 5     # ignore anchors with fewer than 5 units sold
MAX_CORRECTION_FACTOR = 3.0  # corrected value never exceeds 3× original


def build_channel_clusters(df: pd.DataFrame,
                            threshold: float = CORR_THRESHOLD) -> dict:
    """
    Computes pairwise Pearson correlations between channel demand series
    (aggregated over all products/sizes in df) and returns, for each channel,
    the list of similar channels sorted by correlation descending.
    Only channels with r >= threshold are included.
    """
    agg = (df.groupby(['channel_id', 'season'])['units']
             .sum()
             .reset_index()
             .pivot(index='season', columns='channel_id', values='units'))

    corr = agg.corr()
    clusters = {}
    for ch in corr.columns:
        similar = (corr[ch]
                   .drop(ch)
                   .where(lambda x: x >= threshold)
                   .dropna()
                   .sort_values(ascending=False)
                   .index.tolist())
        clusters[ch] = similar
    return clusters


def apply_lost_sales_correction(df: pd.DataFrame,
                                 label: str = '') -> pd.DataFrame:
    """
    Input:  DataFrame with columns:
            product_id, channel_id, season, size, units, stock_units, stockout
    Output: Same DataFrame + column 'units_corrected'.
            Non-stockout rows: units_corrected == units (unchanged).
            Stockout rows:     units_corrected >= units (corrected upward).
    """
    df = df.copy()
    df['units_corrected'] = df['units'].astype(float)

    # ── Build data-driven channel clusters ────────────────────────────────────
    channel_clusters = build_channel_clusters(df, threshold=CORR_THRESHOLD)

    # ── Pre-compute historical size shares per (product_id, channel_id, size) ──
    # Only from non-stockout observations to avoid biased shares.
    ns = df[~df['stockout']].copy()

    ch_prod_total = ns.groupby(['product_id', 'channel_id'])['units'].sum()
    ch_prod_size  = ns.groupby(['product_id', 'channel_id', 'size'])['units'].sum()

    hist_shares = {}   # (product_id, channel_id, size) → fraction
    for (pid, ch, sz), su in ch_prod_size.items():
        total = ch_prod_total.get((pid, ch), 0)
        hist_shares[(pid, ch, sz)] = su / total if total > 0 else 0.0

    # ── Pre-compute historical sell-through per (product_id, channel_id, size) ──
    # Mean sell-through across all non-stockout years.
    hist_st = {}   # (product_id, channel_id, size) → avg sell-through
    for (pid, ch, sz), grp in ns.groupby(['product_id', 'channel_id', 'size']):
        valid = grp[grp['stock_units'] > 0]
        if len(valid) > 0:
            hist_st[(pid, ch, sz)] = (valid['units'] / valid['stock_units']).mean()

    # ── Apply correction row by row ────────────────────────────────────────────
    level_counts   = {1: 0, 2: 0, 3: 0, 'unchanged': 0, 'skipped_no_stock': 0}
    level1_factors = []
    level2_factors = []
    level3_factors = []
    cap_hits       = 0

    # Only correct rows where there was genuine stock on hand.
    # Rows with stock=0 are "never stocked", not lost-sales scenarios.
    skipped = df[df['stockout'] & (df['stock_units'] == 0)].index
    level_counts['skipped_no_stock'] = len(skipped)
    stockout_rows = df[df['stockout'] & (df['stock_units'] > 0)].index

    for idx in stockout_rows:
        row    = df.loc[idx]
        pid    = row['product_id']
        ch     = row['channel_id']
        season = row['season']
        sz     = row['size']
        stock  = row['stock_units']

        corrected = None

        # ── Level 1: other sizes, same product × channel × season ────────────
        anchors = df[
            (df['product_id'] == pid) &
            (df['channel_id'] == ch) &
            (df['season']     == season) &
            (~df['stockout']) &
            (df['size']       != sz)
        ]

        own_share = hist_shares.get((pid, ch, sz), 0.0)

        if len(anchors) > 0 and own_share > 0:
            estimates = []
            for _, anc in anchors.iterrows():
                if anc['units'] < MIN_ANCHOR_UNITS:   # Penny Trap fix
                    continue
                a_share = hist_shares.get((pid, ch, anc['size']), 0.0)
                if a_share > 0:
                    implied_total = anc['units'] / a_share
                    estimates.append(implied_total * own_share)
            if estimates:
                raw = max(row['units'], float(np.mean(estimates)))
                cap = (row['units'] if row['units'] > 0 else stock) * MAX_CORRECTION_FACTOR
                corrected = min(raw, cap)
                if raw >= cap:
                    cap_hits += 1
                level_counts[1] += 1
                if row['units'] > 0:
                    level1_factors.append(corrected / row['units'])

        # ── Level 2: similar channel, same product × season ───────────────────
        if corrected is None and stock > 0:
            similar = channel_clusters.get(ch, [])
            l2_estimates = []

            for sim_ch in similar:
                sim_rows = df[
                    (df['product_id'] == pid) &
                    (df['channel_id'] == sim_ch) &
                    (df['season']     == season) &
                    (~df['stockout']) &
                    (df['stock_units'] > 0)
                ]
                if len(sim_rows) > 0:
                    sim_st = (sim_rows['units'] / sim_rows['stock_units']).mean()
                    if sim_st > 0:
                        l2_estimates.append(stock / sim_st)

            if l2_estimates:
                raw = max(row['units'], float(np.mean(l2_estimates)))
                cap = (row['units'] if row['units'] > 0 else stock) * MAX_CORRECTION_FACTOR
                corrected = min(raw, cap)
                if raw >= cap:
                    cap_hits += 1
                level_counts[2] += 1
                if row['units'] > 0:
                    level2_factors.append(corrected / row['units'])

        # ── Level 3: historical sell-through fallback ─────────────────────────
        if corrected is None and stock > 0:
            st = hist_st.get((pid, ch, sz))
            if st is not None and st > 0:
                raw = max(row['units'], stock / st)
                cap = (row['units'] if row['units'] > 0 else stock) * MAX_CORRECTION_FACTOR
                corrected = min(raw, cap)
                if raw >= cap:
                    cap_hits += 1
                level_counts[3] += 1
                if row['units'] > 0:
                    level3_factors.append(corrected / row['units'])

        if corrected is not None:
            df.at[idx, 'units_corrected'] = corrected
        else:
            level_counts['unchanged'] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_so = max(len(stockout_rows), 1)   # avoid div-by-zero
    before   = df['units'].sum()
    after    = df['units_corrected'].sum()
    tag      = f' – {label}' if label else ''

    def _lvl_line(name, count, factors):
        base = f"  {name:<26}: {count:4d} ({count/total_so*100:4.1f}%)"
        if factors:
            f = np.asarray(factors)
            return f"{base}  factor: mean={f.mean():.2f}x  median={np.median(f):.2f}x  p95={np.percentile(f,95):.2f}x"
        return base

    print(f"\n{'='*70}")
    print(f"  Lost Sales Correction{tag}")
    print(f"{'='*70}")
    print(f"  Stocked-out rows processed : {total_so}")
    print(f"  Skipped (stock=0, no sale) : {level_counts['skipped_no_stock']:4d}  "
          f"← not genuine lost-sales scenarios")
    print(_lvl_line('Level 1 (cross-size)',      level_counts[1], level1_factors))
    print(_lvl_line('Level 2 (cross-channel)',   level_counts[2], level2_factors))
    print(_lvl_line('Level 3 (hist. sell-thru)', level_counts[3], level3_factors))
    print(f"  Unchanged (no data)        : {level_counts['unchanged']:4d} "
          f"({level_counts['unchanged']/total_so*100:.1f}%)  ← contributes downward bias")

    # ── Bias diagnostics ──────────────────────────────────────────────────────
    corrected_rows = df.loc[stockout_rows]
    if len(corrected_rows) > 0:
        st_corr = corrected_rows['units_corrected'] / corrected_rows['stock_units']
        implausible = (st_corr > 3).sum()
        print(f"\n  Diagnostics:")
        print(f"    Cap hits (reached {MAX_CORRECTION_FACTOR}x limit) : {cap_hits:4d} "
              f"({cap_hits/total_so*100:.1f}%)  ← high count = cap too tight")
        print(f"    Corrected sell-through (units_corrected/stock):")
        print(f"      median = {st_corr.median():.2f}x   "
              f"p95 = {st_corr.quantile(0.95):.2f}x   "
              f"max = {st_corr.max():.2f}x")
        print(f"    Implausible (>3x stock)                : {implausible:4d}")

    print(f"\n  Units before : {before:>8,.0f}")
    print(f"  Units after  : {after:>8,.0f}  (Δ = {after-before:+,.0f}, "
          f"+{(after-before)/before*100:.1f}%)")
    print(f"{'='*70}")

    return df
