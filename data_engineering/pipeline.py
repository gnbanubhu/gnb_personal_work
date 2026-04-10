"""
data_engineering/pipeline.py
------------------------------
A complete Data Engineering Pipeline using PySpark.

Simulates a real-world Retail ETL pipeline with these stages:

  Stage 1 — INGEST      : Load raw CSV-like data into Spark DataFrames
  Stage 2 — VALIDATE    : Check nulls, duplicates, schema, data quality
  Stage 3 — CLEAN       : Handle nulls, fix types, standardize values
  Stage 4 — TRANSFORM   : Enrich, join, aggregate, derive new columns
  Stage 5 — LOAD        : Write results to partitioned Parquet output
  Stage 6 — AUDIT       : Log pipeline metrics and data lineage summary

Usage:
    python pipeline.py
"""

import os
import tempfile
from datetime import datetime, date

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType, BooleanType
)

# ── Spark Session ──────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("RetailDataEngineeringPipeline") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

OUTPUT_DIR   = tempfile.mkdtemp()
PIPELINE_RUN = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
audit_log    = []


def section(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def log_audit(stage: str, table: str, records_in: int, records_out: int, notes: str = ""):
    audit_log.append({
        "stage":       stage,
        "table":       table,
        "records_in":  records_in,
        "records_out": records_out,
        "dropped":     records_in - records_out,
        "notes":       notes,
        "timestamp":   datetime.utcnow().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 1 — INGEST : Load Raw Data")

# Raw customers (with intentional dirty data)
raw_customers_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("name",        StringType(), True),
    StructField("email",       StringType(), True),
    StructField("segment",     StringType(), True),
    StructField("city",        StringType(), True),
    StructField("join_date",   StringType(), True),
    StructField("age",         StringType(), True),   # stored as string intentionally
])

raw_customers_data = [
    ("C001", "Alice Johnson",  "alice@email.com",   "Premium", "New York",    "2021-03-15", "34"),
    ("C002", "bob smith",      "bob@email.com",     "regular", "los angeles", "2020-07-01", "38"),
    ("C003", "CHARLIE BROWN",  "charlie@email.com", "BUDGET",  "CHICAGO",     "2022-01-10", "31"),
    ("C004", "Diana Williams", "diana@email.com",   "Premium", "Houston",     "2019-11-20", "36"),
    ("C005", "Eve Davis",      None,                "Regular", "Phoenix",     "2023-04-05", "28"),
    ("C006", "Frank Miller",   "frank@email.com",   "Premium", "Seattle",     "2021-09-30", "-5"),  # invalid age
    ("C001", "Alice Johnson",  "alice@email.com",   "Premium", "New York",    "2021-03-15", "34"),  # duplicate
    (None,   "Unknown Person", "unknown@email.com", "Regular", "Dallas",      "2022-06-15", "25"),  # null ID
]

# Raw products (with dirty data)
raw_products_schema = StructType([
    StructField("product_id",   StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category",     StringType(), True),
    StructField("price",        StringType(), True),  # stored as string
    StructField("cost",         StringType(), True),
    StructField("stock",        StringType(), True),
])

raw_products_data = [
    ("P001", "Laptop Pro 15",       "Electronics", "1299.99", "750.00", "50"),
    ("P002", "Wireless Headphones", "Electronics", "149.99",  "45.00",  "200"),
    ("P003", "Running Shoes",       "Footwear",    "89.99",   "35.00",  "150"),
    ("P004", "Organic Milk 1L",     "Grocery",     "3.49",    "1.20",   "500"),
    ("P005", "Office Chair",        "Furniture",   "299.99",  "120.00", "30"),
    ("P006", "Python Cookbook",     "Books",       "49.99",   "15.00",  "100"),
    ("P007", "Yoga Mat",            "Sports",      "-10.00",  "12.00",  "80"),  # invalid price
    ("P008", "Coffee Beans 500g",   "Grocery",     "14.99",   "5.00",   None),  # null stock
]

# Raw sales transactions (with dirty data)
raw_sales_schema = StructType([
    StructField("sale_id",     StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_id",  StringType(), True),
    StructField("quantity",    StringType(), True),  # stored as string
    StructField("amount",      StringType(), True),
    StructField("sale_date",   StringType(), True),
    StructField("channel",     StringType(), True),
])

raw_sales_data = [
    ("S001", "C001", "P001", "1", "1007.39", "2026-04-01", "In-store"),
    ("S002", "C002", "P003", "2", "165.58",  "2026-04-01", "In-store"),
    ("S003", "C003", "P008", "4", "55.16",   "2026-04-02", "In-store"),
    ("S004", "C004", "P002", "1", "117.29",  "2026-04-03", "Online"),
    ("S005", "C001", "P006", "3", "137.97",  "2026-04-04", "In-store"),
    ("S006", "C005", "P007", "2", "38.63",   "2026-04-05", "In-store"),
    ("S007", "C002", "P004", "6", "19.26",   "2026-04-06", "In-store"),
    ("S008", "C004", "P005", "1", "275.99",  "2026-04-07", "Online"),
    ("S009", "C003", "P001", "1", "1016.59", "2026-04-08", "In-store"),
    ("S010", "C001", "P008", "2", "27.58",   "2026-04-09", "In-store"),
    ("S011", "C999", "P001", "1", "1007.39", "2026-04-09", "Online"),  # unknown customer
    ("S012", "C001", "P999", "1", "99.99",   "2026-04-09", "Online"),  # unknown product
    (None,   "C002", "P003", "1", "82.79",   "2026-04-09", "Online"),  # null sale_id
]

raw_customers = spark.createDataFrame(raw_customers_data, schema=raw_customers_schema)
raw_products  = spark.createDataFrame(raw_products_data,  schema=raw_products_schema)
raw_sales     = spark.createDataFrame(raw_sales_data,     schema=raw_sales_schema)

print(f"\n  raw_customers : {raw_customers.count()} rows | {len(raw_customers.columns)} cols")
print(f"  raw_products  : {raw_products.count()} rows  | {len(raw_products.columns)} cols")
print(f"  raw_sales     : {raw_sales.count()} rows  | {len(raw_sales.columns)} cols")
raw_customers.show(truncate=False)
raw_products.show(truncate=False)
raw_sales.show(truncate=False)

log_audit("INGEST", "raw_customers", 0, raw_customers.count(), "Loaded raw customer data")
log_audit("INGEST", "raw_products",  0, raw_products.count(),  "Loaded raw product data")
log_audit("INGEST", "raw_sales",     0, raw_sales.count(),     "Loaded raw sales data")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 2 — VALIDATE : Data Quality Checks")

def validate(df: DataFrame, name: str):
    """Run DQ checks: null counts, duplicate IDs, row count."""
    total = df.count()
    print(f"\n  [{name}]  total_rows={total}")

    # Null check per column
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])
    print("  Null counts per column:")
    null_counts.show()

    return total

customers_total = validate(raw_customers, "raw_customers")
products_total  = validate(raw_products,  "raw_products")
sales_total     = validate(raw_sales,     "raw_sales")

# Duplicate check
dup_customers = raw_customers.groupBy("customer_id").count().filter(F.col("count") > 1)
dup_sales     = raw_sales.groupBy("sale_id").count().filter(F.col("count") > 1)
print(f"\n  Duplicate customer_ids : {dup_customers.count()}")
print(f"  Duplicate sale_ids     : {dup_sales.count()}")
dup_customers.show()

log_audit("VALIDATE", "raw_customers", customers_total, customers_total,
          f"Nulls detected in id/email, {dup_customers.count()} duplicates found")
log_audit("VALIDATE", "raw_products",  products_total, products_total,
          "Negative price in P007, null stock in P008")
log_audit("VALIDATE", "raw_sales",     sales_total, sales_total,
          "Null sale_id, orphan customer/product references")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLEAN
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 3 — CLEAN : Fix Data Quality Issues")

# ── Clean Customers ───────────────────────────────────────────────────────────
print("\n  Cleaning customers...")

clean_customers = (
    raw_customers
    .dropDuplicates(["customer_id"])                          # remove duplicates
    .filter(F.col("customer_id").isNotNull())                 # drop null IDs
    .withColumn("name",    F.initcap(F.col("name")))          # normalize case
    .withColumn("segment", F.initcap(F.col("segment")))       # normalize case
    .withColumn("city",    F.initcap(F.col("city")))          # normalize case
    .withColumn("email",   F.coalesce(F.col("email"),
                           F.lit("unknown@email.com")))       # fill null email
    .withColumn("age",     F.col("age").cast(IntegerType()))  # cast to int
    .withColumn("age",     F.when(F.col("age") < 0, None)
                            .otherwise(F.col("age")))         # fix invalid age
    .withColumn("join_date", F.to_date(F.col("join_date"),
                             "yyyy-MM-dd"))                   # cast to date
)

print(f"  Before: {raw_customers.count()} rows  →  After: {clean_customers.count()} rows")
clean_customers.show(truncate=False)
log_audit("CLEAN", "customers", raw_customers.count(), clean_customers.count(),
          "Removed duplicates/null IDs, normalized case, fixed types")

# ── Clean Products ────────────────────────────────────────────────────────────
print("\n  Cleaning products...")

clean_products = (
    raw_products
    .filter(F.col("product_id").isNotNull())
    .withColumn("price", F.col("price").cast(DoubleType()))
    .withColumn("cost",  F.col("cost").cast(DoubleType()))
    .withColumn("stock", F.col("stock").cast(IntegerType()))
    .withColumn("price", F.when(F.col("price") <= 0, None)   # fix invalid price
                          .otherwise(F.col("price")))
    .withColumn("stock", F.coalesce(F.col("stock"), F.lit(0)))# fill null stock
)

print(f"  Before: {raw_products.count()} rows  →  After: {clean_products.count()} rows")
clean_products.show(truncate=False)
log_audit("CLEAN", "products", raw_products.count(), clean_products.count(),
          "Cast types, fixed negative price, filled null stock")

# ── Clean Sales ───────────────────────────────────────────────────────────────
print("\n  Cleaning sales...")

valid_customer_ids = clean_customers.select("customer_id")
valid_product_ids  = clean_products.select("product_id")

clean_sales = (
    raw_sales
    .filter(F.col("sale_id").isNotNull())                     # drop null sale_ids
    .withColumn("quantity",  F.col("quantity").cast(IntegerType()))
    .withColumn("amount",    F.col("amount").cast(DoubleType()))
    .withColumn("sale_date", F.to_date(F.col("sale_date"), "yyyy-MM-dd"))
    .join(valid_customer_ids, on="customer_id", how="inner")  # remove orphan customers
    .join(valid_product_ids,  on="product_id",  how="inner")  # remove orphan products
)

print(f"  Before: {raw_sales.count()} rows  →  After: {clean_sales.count()} rows")
clean_sales.show(truncate=False)
log_audit("CLEAN", "sales", raw_sales.count(), clean_sales.count(),
          "Removed null sale_id, orphan customer/product refs")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 4 — TRANSFORM : Enrich & Aggregate")

# ── 4a. Enrich sales with customer and product info ───────────────────────────
print("\n  4a. Enriched Sales (join customers + products)")

enriched_sales = (
    clean_sales
    .join(
        clean_customers.select("customer_id", "name", "segment", "city"),
        on="customer_id", how="left"
    )
    .join(
        clean_products.select("product_id", "product_name", "category", "price", "cost"),
        on="product_id", how="left"
    )
    .withColumn("profit",       F.round(F.col("amount") - (F.col("cost") * F.col("quantity")), 2))
    .withColumn("margin_pct",   F.round((F.col("profit") / F.col("amount")) * 100, 2))
    .withColumn("is_bulk",      F.col("quantity") > 3)
    .withColumn("sale_year",    F.year(F.col("sale_date")))
    .withColumn("sale_month",   F.month(F.col("sale_date")))
    .withColumn("day_of_week",  F.date_format(F.col("sale_date"), "EEEE"))
    .withColumn("pipeline_run", F.lit(PIPELINE_RUN))
)

enriched_sales.select(
    "sale_id", "name", "product_name", "category",
    "quantity", "amount", "profit", "margin_pct", "channel", "sale_date"
).show(truncate=False)

# ── 4b. Sales summary by customer ────────────────────────────────────────────
print("\n  4b. Customer Sales Summary")

customer_summary = (
    enriched_sales
    .groupBy("customer_id", "name", "segment", "city")
    .agg(
        F.count("sale_id")            .alias("total_orders"),
        F.sum("quantity")             .alias("total_units"),
        F.round(F.sum("amount"), 2)   .alias("total_revenue"),
        F.round(F.sum("profit"), 2)   .alias("total_profit"),
        F.round(F.avg("amount"), 2)   .alias("avg_order_value"),
        F.round(F.avg("margin_pct"),2).alias("avg_margin_pct"),
        F.max("sale_date")            .alias("last_purchase_date"),
    )
    .withColumn("customer_tier",
        F.when(F.col("total_revenue") > 1000, "Gold")
         .when(F.col("total_revenue") > 300,  "Silver")
         .otherwise("Bronze")
    )
    .orderBy(F.col("total_revenue").desc())
)

customer_summary.show(truncate=False)

# ── 4c. Sales summary by product category ────────────────────────────────────
print("\n  4c. Category Sales Summary")

category_summary = (
    enriched_sales
    .groupBy("category")
    .agg(
        F.count("sale_id")             .alias("total_transactions"),
        F.sum("quantity")              .alias("total_units_sold"),
        F.round(F.sum("amount"),  2)   .alias("total_revenue"),
        F.round(F.sum("profit"),  2)   .alias("total_profit"),
        F.round(F.avg("margin_pct"),2) .alias("avg_margin_pct"),
    )
    .orderBy(F.col("total_revenue").desc())
)

category_summary.show(truncate=False)

# ── 4d. Daily sales trend ─────────────────────────────────────────────────────
print("\n  4d. Daily Sales Trend")

daily_trend = (
    enriched_sales
    .groupBy("sale_date", "day_of_week")
    .agg(
        F.count("sale_id")           .alias("transactions"),
        F.sum("quantity")            .alias("units_sold"),
        F.round(F.sum("amount"), 2)  .alias("daily_revenue"),
        F.round(F.sum("profit"), 2)  .alias("daily_profit"),
    )
    .orderBy("sale_date")
)

daily_trend.show(truncate=False)

# ── 4e. Channel performance ───────────────────────────────────────────────────
print("\n  4e. Channel Performance")

channel_perf = (
    enriched_sales
    .groupBy("channel")
    .agg(
        F.count("sale_id")                                       .alias("transactions"),
        F.round(F.sum("amount"), 2)                              .alias("total_revenue"),
        F.round(F.avg("amount"), 2)                              .alias("avg_order_value"),
        F.round(F.sum("amount") / F.sum(F.sum("amount")).over(
            __import__("pyspark.sql.window", fromlist=["Window"])
            .Window.rowsBetween(
                __import__("pyspark.sql.window", fromlist=["Window"])
                .Window.unboundedPreceding,
                __import__("pyspark.sql.window", fromlist=["Window"])
                .Window.unboundedFollowing
            )
        ) * 100, 2)                                              .alias("revenue_share_pct"),
    )
    .orderBy(F.col("total_revenue").desc())
)

channel_perf.show(truncate=False)

log_audit("TRANSFORM", "enriched_sales",    clean_sales.count(), enriched_sales.count(),    "Joined customers + products, derived profit/margin/tier")
log_audit("TRANSFORM", "customer_summary",  enriched_sales.count(), customer_summary.count(), "Aggregated per customer with tier classification")
log_audit("TRANSFORM", "category_summary",  enriched_sales.count(), category_summary.count(), "Aggregated per product category")
log_audit("TRANSFORM", "daily_trend",       enriched_sales.count(), daily_trend.count(),       "Daily revenue trend aggregation")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 5 — LOAD : Write to Parquet")

# Write enriched sales partitioned by category
enriched_path = os.path.join(OUTPUT_DIR, "enriched_sales")
enriched_sales.write.mode("overwrite") \
    .partitionBy("category") \
    .parquet(enriched_path)

# Write customer summary
customer_path = os.path.join(OUTPUT_DIR, "customer_summary")
customer_summary.write.mode("overwrite").parquet(customer_path)

# Write category summary
category_path = os.path.join(OUTPUT_DIR, "category_summary")
category_summary.write.mode("overwrite").parquet(category_path)

# Write daily trend
trend_path = os.path.join(OUTPUT_DIR, "daily_trend")
daily_trend.write.mode("overwrite").parquet(trend_path)

# Verify written files
print(f"\n  Output directory : {OUTPUT_DIR}")
for folder in ["enriched_sales", "customer_summary", "category_summary", "daily_trend"]:
    path  = os.path.join(OUTPUT_DIR, folder)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(f"  {folder:<25} → {len(files)} partition(s)/file(s) written")

# Read back to verify
print("\n  Read-back verification — enriched_sales:")
spark.read.parquet(enriched_path).select(
    "sale_id", "name", "product_name", "amount", "profit", "category"
).show(5, truncate=False)

log_audit("LOAD", "enriched_sales",   enriched_sales.count(),   enriched_sales.count(),   f"Written to {enriched_path} partitioned by category")
log_audit("LOAD", "customer_summary", customer_summary.count(), customer_summary.count(), f"Written to {customer_path}")
log_audit("LOAD", "category_summary", category_summary.count(), category_summary.count(), f"Written to {category_path}")
log_audit("LOAD", "daily_trend",      daily_trend.count(),      daily_trend.count(),      f"Written to {trend_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 6 — AUDIT : Pipeline Metrics & Lineage")

total_revenue   = enriched_sales.agg(F.round(F.sum("amount"), 2)).collect()[0][0]
total_profit    = enriched_sales.agg(F.round(F.sum("profit"), 2)).collect()[0][0]
total_orders    = enriched_sales.count()
top_customer    = customer_summary.orderBy(F.col("total_revenue").desc()).first()
top_category    = category_summary.orderBy(F.col("total_revenue").desc()).first()

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │           PIPELINE EXECUTION SUMMARY                │
  ├─────────────────────────────────────────────────────┤
  │  Pipeline Run   : {PIPELINE_RUN:<33}│
  │  Total Orders   : {total_orders:<33}│
  │  Total Revenue  : ${total_revenue:<32}│
  │  Total Profit   : ${total_profit:<32}│
  │  Top Customer   : {top_customer['name']:<33}│
  │  Top Category   : {top_category['category']:<33}│
  └─────────────────────────────────────────────────────┘
""")

print("  Data Lineage & Audit Log:")
print(f"  {'Stage':<12} {'Table':<22} {'In':>6} {'Out':>6} {'Dropped':>8}  Notes")
print(f"  {'-'*12} {'-'*22} {'-'*6} {'-'*6} {'-'*8}  {'-'*30}")
for entry in audit_log:
    print(f"  {entry['stage']:<12} {entry['table']:<22} "
          f"{entry['records_in']:>6} {entry['records_out']:>6} "
          f"{entry['dropped']:>8}  {entry['notes'][:40]}")

spark.stop()
