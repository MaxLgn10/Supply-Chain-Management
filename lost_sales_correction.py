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
    level_counts   = {1: 0, 2: 0, 3: 0, 'unchanged': 0}
    level1_factors = []
    level2_factors = []
    level3_factors = []

    stockout_rows = df[df['stockout']].index

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
                level_counts[3] += 1
                if row['units'] > 0:
                    level3_factors.append(corrected / row['units'])

        if corrected is not None:
            df.at[idx, 'units_corrected'] = corrected
        else:
            level_counts['unchanged'] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_so = len(stockout_rows)
    before   = df['units'].sum()
    after    = df['units_corrected'].sum()
    tag      = f' – {label}' if label else ''

    print(f"\n{'='*55}")
    print(f"  Lost Sales Correction{tag}")
    print(f"{'='*55}")
    print(f"  Stocked-out rows processed : {total_so}")
    print(f"  Level 1 (cross-size)       : {level_counts[1]:4d} "
          f"({level_counts[1]/total_so*100:.1f}%)  "
          f"avg factor = {np.mean(level1_factors):.2f}x" if level1_factors else
          f"  Level 1 (cross-size)       : {level_counts[1]:4d} "
          f"({level_counts[1]/total_so*100:.1f}%)")
    print(f"  Level 2 (cross-channel)    : {level_counts[2]:4d} "
          f"({level_counts[2]/total_so*100:.1f}%)  "
          f"avg factor = {np.mean(level2_factors):.2f}x" if level2_factors else
          f"  Level 2 (cross-channel)    : {level_counts[2]:4d} "
          f"({level_counts[2]/total_so*100:.1f}%)")
    print(f"  Level 3 (hist. sell-thru)  : {level_counts[3]:4d} "
          f"({level_counts[3]/total_so*100:.1f}%)  "
          f"avg factor = {np.mean(level3_factors):.2f}x" if level3_factors else
          f"  Level 3 (hist. sell-thru)  : {level_counts[3]:4d} "
          f"({level_counts[3]/total_so*100:.1f}%)")
    print(f"  Unchanged (no data)        : {level_counts['unchanged']:4d} "
          f"({level_counts['unchanged']/total_so*100:.1f}%)")
    print(f"  Units before : {before:>8,.0f}")
    print(f"  Units after  : {after:>8,.0f}  (Δ = {after-before:+,.0f}, "
          f"+{(after-before)/before*100:.1f}%)")
    print(f"{'='*55}")

    return df
