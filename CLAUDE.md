# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Academic group assignment (Ghent University, Supply Chain Management, prof. LP Kerkhove) solving the **Pre-Pack Problem (PPP)** for a fictional apparel retailer "Sustainable Essentials". Goal: forecast winter 2026 demand per SKU × channel and design optimal pre-packs (size assortments) that minimize the sum of pack creation, handling, and capital cost.

- **Submission deadline:** 5 May 2026, 23:59
- **Presentation:** 7 May 2026 (5 min + 10 min Q&A)
- **Full brief:** `data/SCM - Group assignment - 2026.pdf`
- **Solution template:** `data/PPP_solutionFile_2026.xlsx` (fill `PacksDefined` + `PackAllocation` tabs only — do not add sheets/macros)

### Cost parameters (hardcoded in `01_data_exploration.py`)
- Pack creation: **€134** per unique pack type
- Handling: **€11.03** per pack (also per unit without packs — the baseline)
- Cost of capital: **24.3%** on unsold-inventory cost

## Pipeline architecture

Three scripts run in order. Each stage writes into `outputs/` and feeds the next.

```
01_data_exploration.py  →  outputs/merged_data.csv   (single source of truth)
        ↓
run_forecasting.py       (uses lost_sales_correction.py)
        ↓ per category:   outputs/forecast_<slug>_channel_size.csv   ← main optimization input
                          outputs/forecast_<slug>_channel_total.csv
                          outputs/forecast_<slug>_results.xlsx       (4-tab report)
                          outputs/plots_<slug>/*.png
        ↓
[not yet implemented]    pre-pack optimization → fills PPP_solutionFile_2026.xlsx
```

### Stage 1 — Data exploration
Merges `PPP_stu_demand.xlsx`, `PPP_stu_stock.xlsx`, `PPP_stu_products.xlsx` into `outputs/merged_data.csv`. Defines `stockout = units >= stock_units` — this flag drives everything downstream. Seasons run 2018–2025; 2026 is to be forecast.

### Stage 2 — Lost Sales Correction (LSC) → forecasting
**Categories are defined in `data/PPP_stu_product_categories.xlsx`** (12 groups like `sweatshirt men`, `hoodie women`, `gloves`, etc.). This file was created by the group — there is no official grouping from the assignment. `run_forecasting.py` and both tuning scripts read it via `load_categories()`.

`lost_sales_correction.py` runs **before** forecasting. Hierarchical, 3 levels (fallback order):
1. **Cross-size** — same product × channel × season, uses historical size-share weights
2. **Cross-channel** — channels clustered via Pearson correlation (threshold `r ≥ 0.85`) on season-level demand; uses stock × sell-through of correlated peers
3. **Historical sell-through** — product × channel × size fallback

Guardrails: `MIN_ANCHOR_UNITS = 5` (Penny Trap fix), `MAX_CORRECTION_FACTOR = 3.0` (cap). Correction writes `units_corrected` column; downstream code uses this, never `units`.

Forecasting uses a **MAPE-weighted ensemble** of 4 methods, validated on 2025:
- MA(3) — window `MA_WINDOW=3` tuned globally via `tune_ma_window.py`
- SES with α=0.47 — tuned globally via `tune_ses_alpha.py`
- Holt's linear trend (auto-optimized)
- Linear regression on all seasons

Weights are inverse-MAPE. The ensemble forecast is then disaggregated to `channel × size` via historical size-shares per channel.

**Known gap:** The current output gives `channel × size` totals per category but **not** per product (SKU). The PPP optimization needs per-SKU demand, so the forecast still needs product-level disaggregation (e.g., via historical product shares within each category × channel) before optimization can start.

### Stage 3 — Optimization (not yet implemented)
Must produce two tabs in `PPP_solutionFile_2026.xlsx`:
- `PacksDefined` — rows = packs, columns = SKUs, values = units of SKU per pack
- `PackAllocation` — rows = packs, columns = sales channels, values = pack count

## Commands

Scripts run from the project root, in order:
```bash
python 01_data_exploration.py        # regenerate outputs/merged_data.csv after any data change
python run_forecasting.py            # full pipeline across all 12 categories (~1–2 min)
```

Hyperparameter re-tuning (only needed when LSC logic or categories change):
```bash
python tune_ma_window.py             # prints best MA window; update MA_WINDOW in run_forecasting.py
python tune_ses_alpha.py             # prints best α;        update SES_ALPHA   in run_forecasting.py
```

No tests. No linter. No build. Project runs in the `.venv/` virtualenv (already set up).

## Conventions worth knowing

- **Slugs for filenames** come from `slugify()` in `run_forecasting.py` (e.g., `"T-shirt women"` → `t_shirt_women`). If a category is renamed, stale files under the old slug must be deleted manually.
- **Size order is dynamic per category** — `get_size_order()` picks XS–XXL for menswear, XXS–XL for womenswear, `['onesize']` for accessories/gloves/socks. Don't hardcode size lists.
- **Excel formatting** in `run_forecasting.py`: only tabs in `has_total_row` (`Per Channel`, `Channel x Size`, `Size Verification`) get the styled TOTAL row. `Method Weights` must NOT — its last row is "Linear Regression", not a total.
- **`outputs/` is gitignored.** Generated artefacts are not committed.
- **LSC calibration history** (April 2026): `MIN_ANCHOR_UNITS=5` fixed a Penny Trap (single-unit anchors inflating corrections); `MAX_CORRECTION_FACTOR=3.0` replaced a 29× commercially implausible outlier.
