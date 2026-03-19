# Script 1 – Data exploration
# Run this first before forecasting or optimization.
# Loads the three source files, merges them, and saves plots + summary tables.

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- cost parameters (2026 edition) --------------------------
COST_OF_CAPITAL = 0.243   # 24.3%
HANDLING_COST   = 11.03   # € per unit through DC (no packs)
PACK_CREATION   = 134.00  # € per unique pack type

# --- file paths ----------------------------------------------
PATH_PRODUCTS = "data/PPP_stu_products.xlsx"
PATH_DEMAND   = "data/PPP_stu_demand.xlsx"
PATH_STOCK    = "data/PPP_stu_stock.xlsx"
OUTPUT_DIR    = "outputs/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & merge data ──────────────────────────────────────────────────────

products = pd.read_excel(PATH_PRODUCTS)
demand   = pd.read_excel(PATH_DEMAND)
stock    = pd.read_excel(PATH_STOCK).rename(columns={"units": "stock_units"})

df = demand.merge(stock, on=["product_id", "channel_id", "season", "size"], how="left")
df = df.merge(products[["id", "name", "category", "price", "cost"]],
              left_on="product_id", right_on="id", how="left").drop(columns="id")

df["sku"]          = df["product_id"].astype(str) + "_" + df["size"].astype(str)
df["leftover"]     = (df["stock_units"] - df["units"]).clip(lower=0)
df["capital_cost"] = df["leftover"] * df["cost"] * COST_OF_CAPITAL
df["sell_through"] = df["units"] / df["stock_units"]
df["stockout"]     = df["units"] >= df["stock_units"]   # sold everything → possible unmet demand

print(f"Loaded: {len(products)} products | {demand.shape[0]:,} demand rows | {stock.shape[0]:,} stock rows")
print(f"SKUs: {df['sku'].nunique()} | Channels: {df['channel_id'].nunique()} | Seasons: {sorted(df['season'].unique())}\n")


# ── 2. Overall summary ────────────────────────────────────────────────────────

print("=== Overall summary ===")
print(f"Units sold (all years):    {df['units'].sum():>10,.0f}")
print(f"Units stocked (all years): {df['stock_units'].sum():>10,.0f}")
print(f"Unsold units:              {df['leftover'].sum():>10,.0f}")
print(f"Avg sell-through:          {df['units'].sum()/df['stock_units'].sum()*100:>9.1f}%")
print(f"Total capital cost:        €{df['capital_cost'].sum():>9,.0f}\n")


# ── 3. Baseline cost – no pre-packs (benchmark for Script 03) ─────────────────
# Current process: every single unit passes through the DC individually.
# No pack setup costs, but handling is charged per unit.

latest_season = df["season"].max()
df_ref = df[df["season"] == latest_season]

baseline_handling = df_ref["stock_units"].sum() * HANDLING_COST
baseline_capital  = df_ref["capital_cost"].sum()
baseline_total    = baseline_handling + baseline_capital

print(f"=== Baseline cost – season {latest_season} (no packs) ===")
print(f"Handling (per unit): €{baseline_handling:>9,.2f}")
print(f"Capital cost:        €{baseline_capital:>9,.2f}")
print(f"Total:               €{baseline_total:>9,.2f}  ← target to beat\n")


# ── 4. Stockout check (feeds into lost-sales correction in Script 02) ─────────

stockout_rate = df["stockout"].mean() * 100
print(f"=== Stockout analysis ===")
print(f"Stockout rate (all SKU×channel×season combos): {stockout_rate:.1f}%\n")

print("Stockout rate by season:")
for s, r in df.groupby("season")["stockout"].mean().mul(100).items():
    print(f"  {s}: {r:.1f}%")

print("\nTop 10 most stocked-out SKUs:")
top_stockouts = (df[df["stockout"]]
                 .groupby(["product_id", "name", "size"])
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False))
print(top_stockouts.head(10).to_string(index=False))
print()

df[["product_id", "name", "channel_id", "season", "size",
    "sku", "units", "stock_units", "stockout", "cost"]].to_csv(
    OUTPUT_DIR + "stockout_flags.csv", index=False)


# ── 5. Plots ──────────────────────────────────────────────────────────────────

# plot 1 – demand per season
season_demand = df.groupby("season")["units"].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(season_demand["season"], season_demand["units"], color="#2a6db5", width=0.6)
ax.plot(season_demand["season"], season_demand["units"], color="#1a3f6f", marker="o", lw=2)
ax.set_title("Total Demand Per Season (All Channels)", fontsize=13, fontweight="bold")
ax.set_xlabel("Season (Winter Year)")
ax.set_ylabel("Units Sold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot1_demand_per_season.png", dpi=150)
plt.close()

# plot 2 – demand per channel over time
channel_season = df.groupby(["season", "channel_id"])["units"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
palette = sns.color_palette("tab10", df["channel_id"].nunique())
for i, ch in enumerate(sorted(df["channel_id"].unique())):
    sub = channel_season[channel_season["channel_id"] == ch]
    ax.plot(sub["season"], sub["units"], marker="o", label=ch, color=palette[i], lw=1.8)
ax.set_title("Demand Per Channel Over Time", fontsize=13, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("Units Sold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot2_demand_per_channel.png", dpi=150)
plt.close()

# plot 3 – sell-through per channel
st = (df.groupby("channel_id")
        .apply(lambda g: g["units"].sum() / g["stock_units"].sum() * 100)
        .reset_index(name="sell_through")
        .sort_values("sell_through"))
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(st["channel_id"], st["sell_through"], color="#2a6db5")
ax.axvline(100, color="red", linestyle="--", lw=1)
ax.set_title("Average Sell-Through Rate Per Channel", fontsize=13, fontweight="bold")
ax.set_xlabel("Sell-Through Rate (%)")
for bar, val in zip(bars, st["sell_through"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot3_sellthrough_per_channel.png", dpi=150)
plt.close()

# plot 4 – size distribution
size_order  = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "onesize"]
size_demand = df.groupby("size")["units"].sum().reindex(size_order).dropna()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(size_demand.index, size_demand.values, color="#2a6db5", width=0.6)
ax.set_title("Total Demand by Size (All Products & Channels)", fontsize=13, fontweight="bold")
ax.set_xlabel("Size")
ax.set_ylabel("Units Sold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot4_size_distribution.png", dpi=150)
plt.close()

# plot 5 – capital cost per season
cap_season = df.groupby("season")["capital_cost"].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(cap_season["season"], cap_season["capital_cost"], color="#c0392b", width=0.6, alpha=0.85)
ax.set_title("Capital Cost of Unsold Inventory Per Season", fontsize=13, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("Capital Cost (EUR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot5_capital_cost_per_season.png", dpi=150)
plt.close()

# plot 6 – menswear vs womenswear
cat_season = df.groupby(["season", "category"])["units"].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
for cat, color in zip(["Menswear", "Womenswear"], ["#2a6db5", "#e84393"]):
    sub = cat_season[cat_season["category"] == cat]
    ax.plot(sub["season"], sub["units"], marker="o", label=cat, color=color, lw=2)
ax.set_title("Demand by Category Over Time", fontsize=13, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("Units Sold")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "plot6_category_demand.png", dpi=150)
plt.close()

print("Plots saved (plot1–plot6)\n")


# ── 6. Summary tables ─────────────────────────────────────────────────────────

print("=== Channel summary ===")
ch_summary = df.groupby("channel_id").agg(
    sold          = ("units", "sum"),
    stocked       = ("stock_units", "sum"),
    leftover      = ("leftover", "sum"),
    capital_cost  = ("capital_cost", "sum"),
    stockout_pct  = ("stockout", "mean")
).reset_index()
ch_summary["sell_through_%"] = (ch_summary["sold"] / ch_summary["stocked"] * 100).round(1)
ch_summary["stockout_%"]     = (ch_summary["stockout_pct"] * 100).round(1)
print(ch_summary[["channel_id", "sold", "stocked", "leftover",
                   "sell_through_%", "stockout_%", "capital_cost"]]
      .sort_values("sold", ascending=False).to_string(index=False))

print("\n=== Top 10 overstocked SKUs ===")
sku_summary = df.groupby(["product_id", "name", "size"]).agg(
    sold         = ("units", "sum"),
    stocked      = ("stock_units", "sum"),
    leftover     = ("leftover", "sum"),
    capital_cost = ("capital_cost", "sum")
).reset_index().sort_values("capital_cost", ascending=False)
print(sku_summary.head(10)[["name", "size", "sold", "stocked",
                              "leftover", "capital_cost"]].to_string(index=False))


# ── 7. Export merged dataset for scripts 02 and 03 ────────────────────────────

df.to_csv(OUTPUT_DIR + "merged_data.csv", index=False)
print(f"\nExported merged_data.csv and stockout_flags.csv to {OUTPUT_DIR}")