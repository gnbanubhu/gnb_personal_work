import pandas as pd
import glob
import os

# ── Load all CSV files from test_data ────────────────────────────────────────
csv_files = sorted(glob.glob("test_data/*.csv"))

dfs = []
for file in csv_files:
    year = os.path.basename(file).split("-")[0]
    df = pd.read_csv(file)
    df["year"] = int(year)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

print("=" * 60)
print("COMBINED DATASET OVERVIEW")
print("=" * 60)
print(f"Total records : {len(combined_df)}")
print(f"Years covered : {sorted(combined_df['year'].unique())}")
print(f"\nColumns: {list(combined_df.columns)}")
print(f"\nData Types:\n{combined_df.dtypes}")

# ── Basic stats ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(combined_df.describe())

# ── Records per year ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOTAL FLIGHT COUNT PER YEAR")
print("=" * 60)
yearly = combined_df.groupby("year")["count"].sum().reset_index()
yearly.columns = ["Year", "Total Flights"]
print(yearly.to_string(index=False))

# ── Top 10 destination countries (all years) ──────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 DESTINATION COUNTRIES (All Years)")
print("=" * 60)
top_dest = (
    combined_df.groupby("DEST_COUNTRY_NAME")["count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_dest.columns = ["Destination", "Total Flights"]
print(top_dest.to_string(index=False))

# ── Top 10 origin countries (all years) ───────────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 ORIGIN COUNTRIES (All Years)")
print("=" * 60)
top_origin = (
    combined_df.groupby("ORIGIN_COUNTRY_NAME")["count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_origin.columns = ["Origin", "Total Flights"]
print(top_origin.to_string(index=False))

# ── Top 10 routes (all years) ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 ROUTES (All Years)")
print("=" * 60)
top_routes = (
    combined_df.groupby(["ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME"])["count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_routes.columns = ["Origin", "Destination", "Total Flights"]
print(top_routes.to_string(index=False))

# ── Year-over-year growth ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("YEAR-OVER-YEAR FLIGHT GROWTH")
print("=" * 60)
yearly["Growth %"] = yearly["Total Flights"].pct_change().mul(100).round(2)
print(yearly.to_string(index=False))

# ── Missing values check ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(combined_df.isnull().sum())
