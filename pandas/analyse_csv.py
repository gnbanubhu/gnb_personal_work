import pandas as pd
import glob
import os


def load_data(path="test_data/*.csv"):
    csv_files = sorted(glob.glob(path))
    dfs = []
    for file in csv_files:
        year = os.path.basename(file).split("-")[0]
        df = pd.read_csv(file)
        df["year"] = int(year)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def print_overview(df):
    print("=" * 60)
    print("COMBINED DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records : {len(df)}")
    print(f"Years covered : {sorted(df['year'].unique())}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")


def print_basic_stats(df):
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df.describe())


def print_flights_per_year(df):
    print("\n" + "=" * 60)
    print("TOTAL FLIGHT COUNT PER YEAR")
    print("=" * 60)
    yearly = df.groupby("year")["count"].sum().reset_index()
    yearly.columns = ["Year", "Total Flights"]
    print(yearly.to_string(index=False))
    return yearly


def print_top_destinations(df, n=10):
    print("\n" + "=" * 60)
    print(f"TOP {n} DESTINATION COUNTRIES (All Years)")
    print("=" * 60)
    top_dest = (
        df.groupby("DEST_COUNTRY_NAME")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top_dest.columns = ["Destination", "Total Flights"]
    print(top_dest.to_string(index=False))


def print_top_origins(df, n=10):
    print("\n" + "=" * 60)
    print(f"TOP {n} ORIGIN COUNTRIES (All Years)")
    print("=" * 60)
    top_origin = (
        df.groupby("ORIGIN_COUNTRY_NAME")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top_origin.columns = ["Origin", "Total Flights"]
    print(top_origin.to_string(index=False))


def print_top_routes(df, n=10):
    print("\n" + "=" * 60)
    print(f"TOP {n} ROUTES (All Years)")
    print("=" * 60)
    top_routes = (
        df.groupby(["ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME"])["count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top_routes.columns = ["Origin", "Destination", "Total Flights"]
    print(top_routes.to_string(index=False))


def print_yoy_growth(yearly):
    print("\n" + "=" * 60)
    print("YEAR-OVER-YEAR FLIGHT GROWTH")
    print("=" * 60)
    yearly["Growth %"] = yearly["Total Flights"].pct_change().mul(100).round(2)
    print(yearly.to_string(index=False))


def print_missing_values(df):
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())


def main():
    df = load_data()
    print_overview(df)
    print_basic_stats(df)
    yearly = print_flights_per_year(df)
    print_top_destinations(df)
    print_top_origins(df)
    print_top_routes(df)
    print_yoy_growth(yearly)
    print_missing_values(df)


if __name__ == "__main__":
    main()
