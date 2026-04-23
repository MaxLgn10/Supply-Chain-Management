# validate_lost_sales_correction.py
# Holdout validation of the Lost Sales Correction.
#
# Idea: take real non-stockout rows (where we know the true demand),
# artificially force a stockout by reducing stock to a fraction of actual
# demand, run the LSC, and compare the corrected estimate to the known truth.
#
# Reports:
#   • Mean signed error (bias direction)
#   • MAPE (accuracy)
#   • % under- / over-estimated
#   • Share of lost sales recovered
#
# Input:  outputs/merged_data.csv
# Output: outputs/lsc_validation_report.csv + console summary

import os
import numpy as np
import pandas as pd
from lost_sales_correction import apply_lost_sales_correction

os.makedirs('outputs', exist_ok=True)

RANDOM_SEED        = 42
SAMPLES_PER_FRAC   = 200   # per stockout fraction
STOCKOUT_FRACTIONS = [0.5, 0.7, 0.9]   # force stock = X% of actual demand
MIN_DEMAND         = 3     # only use rows where actual demand is meaningful

rng = np.random.default_rng(RANDOM_SEED)


def run_scenario(df_all: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Force stockouts on non-stockout rows at given fraction, run LSC, return diagnostics."""
    df = df_all.copy()

    pool = df[(~df['stockout']) & (df['stock_units'] >= MIN_DEMAND)]
    if len(pool) == 0:
        return pd.DataFrame()

    n = min(SAMPLES_PER_FRAC, len(pool))
    sample_idx = pool.sample(n=n, random_state=rng.integers(1_000_000)).index

    actual_demand = df.loc[sample_idx, 'units'].copy()
    forced_stock  = np.maximum((actual_demand * frac).round().astype(int), 1)

    df.loc[sample_idx, 'stock_units'] = forced_stock
    df.loc[sample_idx, 'units']       = forced_stock
    df.loc[sample_idx, 'stockout']    = True

    corrected = apply_lost_sales_correction(df, label=f'holdout_{int(frac*100)}pct')

    res = pd.DataFrame({
        'actual_demand':     actual_demand.values,
        'forced_stock':      forced_stock.values,
        'corrected_demand':  corrected.loc[sample_idx, 'units_corrected'].values,
    })
    res['lost_sales_true']      = res['actual_demand'] - res['forced_stock']
    res['lost_sales_recovered'] = res['corrected_demand'] - res['forced_stock']
    res['error']                = res['corrected_demand'] - res['actual_demand']
    res['rel_error']            = res['error'] / res['actual_demand']
    res['scenario_fraction']    = frac
    return res


def summarize(res: pd.DataFrame) -> dict:
    recovered_pct = (res['lost_sales_recovered'].sum() /
                     res['lost_sales_true'].sum() * 100) if res['lost_sales_true'].sum() > 0 else 0
    return {
        'samples':              len(res),
        'mean_actual':          res['actual_demand'].mean(),
        'mean_forced_stock':    res['forced_stock'].mean(),
        'mean_corrected':       res['corrected_demand'].mean(),
        'mean_signed_error':    res['error'].mean(),
        'mape_pct':             (res['error'].abs() / res['actual_demand']).mean() * 100,
        'pct_underestimated':   (res['corrected_demand'] < res['actual_demand']).mean() * 100,
        'pct_overestimated':    (res['corrected_demand'] > res['actual_demand']).mean() * 100,
        'lost_sales_recovered_pct': recovered_pct,
    }


if __name__ == '__main__':
    df_all = pd.read_csv('outputs/merged_data.csv')
    print(f"Loaded {len(df_all):,} rows, seasons {df_all['season'].min()}–{df_all['season'].max()}")

    all_results = []
    summaries   = []

    for frac in STOCKOUT_FRACTIONS:
        print(f"\n{'#'*70}\n  HOLDOUT SCENARIO: forced_stock = {int(frac*100)}% of actual demand\n{'#'*70}")
        res = run_scenario(df_all, frac)
        if res.empty:
            continue
        all_results.append(res)
        summaries.append({'fraction': frac, **summarize(res)})

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'scenario':<10} {'n':>5} {'actual':>7} {'stock':>7} {'corrected':>10} "
          f"{'bias':>7} {'MAPE':>7} {'under%':>7} {'over%':>7} {'recov%':>7}")
    print('-' * 82)
    for s in summaries:
        print(f"stock={int(s['fraction']*100):>2}%    "
              f"{s['samples']:>5} "
              f"{s['mean_actual']:>7.1f} "
              f"{s['mean_forced_stock']:>7.1f} "
              f"{s['mean_corrected']:>10.1f} "
              f"{s['mean_signed_error']:>+7.2f} "
              f"{s['mape_pct']:>6.1f}% "
              f"{s['pct_underestimated']:>6.1f}% "
              f"{s['pct_overestimated']:>6.1f}% "
              f"{s['lost_sales_recovered_pct']:>6.1f}%")

    print(f"\nInterpretation:")
    print(f"  bias     ≈ 0    → unbiased   (negative = under-estimate, positive = over-estimate)")
    print(f"  MAPE     lower is better")
    print(f"  recov%   share of truly lost sales the LSC recovers (higher = better)")
    print(f"{'='*70}")

    all_res_df = pd.concat(all_results, ignore_index=True)
    all_res_df.to_csv('outputs/lsc_validation_report.csv', index=False)
    print(f"\n  Detail saved: outputs/lsc_validation_report.csv  ({len(all_res_df)} rows)")
