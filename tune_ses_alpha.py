# tune_ses_alpha.py
# Finds the globally best SES alpha across ALL product categories and channels.
# Train: 2018–2024  |  Validate: 2025  |  MAPE averaged over all category×channel series
# Input:  outputs/merged_data.csv

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ── Category definitions — read directly from official product file ────────────
def load_categories(path='data/PPP_stu_product_categories.xlsx'):
    prod = pd.read_excel(path, header=0)
    prod.columns = ['id', 'name', 'category_group', 'category', 'price', 'cost']
    prod['category_group'] = prod['category_group'].ffill()
    cats = {}
    for grp, sub in prod.groupby('category_group'):
        cats[grp] = sub['id'].tolist()
    return cats

CATEGORIES = load_categories()

VALIDATION_YEAR = 2025

df = pd.read_csv('outputs/merged_data.csv')

# ── Build all category×channel series ────────────────────────────────────────
series_list = []
for cat_name, product_ids in CATEGORIES.items():
    cat_df = df[df['product_id'].isin(product_ids)].copy()
    agg = (cat_df.groupby(['channel_id', 'season'])['units']
                 .sum().reset_index())
    for channel, grp in agg.groupby('channel_id'):
        grp = grp.sort_values('season')
        y   = grp['units'].values.astype(float)
        yr  = grp['season'].values
        series_list.append({'cat': cat_name, 'channel': channel, 'y': y, 'yr': yr})

print(f"Total series used for tuning: {len(series_list)}  "
      f"({len(CATEGORIES)} categories × ~11 channels)\n")

# ── Grid search ───────────────────────────────────────────────────────────────
alphas  = [round(a, 2) for a in np.arange(0.05, 1.00, 0.01)]   # 0.01 steps for precise optimum
results = []

print(f"{'Alpha':<8} | {'Avg MAPE':>10} | {'Median MAPE':>12} | {'# series':>9}")
print("-" * 48)

for alpha in alphas:
    mapes = []
    for s in series_list:
        y, yr  = s['y'], s['yr']
        train  = y[yr < VALIDATION_YEAR]
        actual = y[yr == VALIDATION_YEAR]
        if len(actual) == 0 or actual[0] == 0:
            continue
        model = SimpleExpSmoothing(train, initialization_method="estimated").fit(
                    smoothing_level=alpha, optimized=False)
        pred  = float(model.forecast(1)[0])
        mapes.append(abs(pred - actual[0]) / actual[0] * 100)

    avg    = np.mean(mapes)
    median = np.median(mapes)
    results.append({'alpha': alpha, 'avg_mape': avg, 'median_mape': median})
    print(f"α={alpha:.2f}   | {avg:9.2f}% | {median:11.2f}% | {len(mapes):>9}")

best = min(results, key=lambda x: x['avg_mape'])
print("\n" + "=" * 48)
print(f"Best alpha: α={best['alpha']:.2f}")
print(f"  Avg MAPE    = {best['avg_mape']:.2f}%")
print(f"  Median MAPE = {best['median_mape']:.2f}%")
print("=" * 48)
print(f"\n→ Use SES(α={best['alpha']:.2f}) in all forecasting scripts")
