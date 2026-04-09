"""
dataframes/sample.py
---------------------
Demonstrates core PySpark DataFrame operations including creation,
schema inspection, column operations, filtering, sorting, and
common DataFrame manipulations.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import (
    col, lit, upper, lower, length, trim, concat, concat_ws,
    when, coalesce, isnull, isnan, regexp_replace, to_date,
    year, month, dayofweek, current_date, date_diff
)

spark = SparkSession.builder \
    .appName("DataFramesSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── DataFrame Creation ────────────────────────────────────────────────────────
print("=" * 55)
print("DATAFRAME CREATION — Explicit Schema")
print("=" * 55)

schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("name",       StringType(),  False),
    StructField("department", StringType(),  True),
    StructField("salary",     DoubleType(),  True),
    StructField("join_date",  StringType(),  True),
])

data = [
    (1, "Alice",   "Engineering", 95000.0,  "2021-03-15"),
    (2, "Bob",     "Marketing",   72000.0,  "2020-07-01"),
    (3, "Charlie", "Engineering", 88000.0,  "2022-01-10"),
    (4, "Diana",   "HR",          65000.0,  "2019-11-20"),
    (5, "Eve",     "Marketing",   78000.0,  "2023-04-05"),
    (6, "Frank",   None,          102000.0, "2021-09-30"),   # null department
    (7, "Grace",   "HR",          None,     "2022-06-15"),   # null salary
]

df = spark.createDataFrame(data, schema)

print("Schema:")
df.printSchema()
print("Data:")
df.show()

# ── Schema Inspection ─────────────────────────────────────────────────────────
print("=" * 55)
print("SCHEMA INSPECTION")
print("=" * 55)
print(f"  Columns    : {df.columns}")
print(f"  Row count  : {df.count()}")
print(f"  Partitions : {df.rdd.getNumPartitions()}")
df.describe("salary").show()

# ── Column Operations ─────────────────────────────────────────────────────────
print("=" * 55)
print("COLUMN OPERATIONS")
print("=" * 55)

df.select(
    col("name"),
    upper(col("name")).alias("name_upper"),
    lower(col("name")).alias("name_lower"),
    length(col("name")).alias("name_length"),
    concat_ws(" - ", col("name"), col("department")).alias("name_dept"),
    col("salary"),
    (col("salary") * 1.10).alias("salary_10pct_raise"),
).show(truncate=False)

# ── Null Handling ─────────────────────────────────────────────────────────────
print("=" * 55)
print("NULL HANDLING")
print("=" * 55)

print("  Rows with nulls:")
df.filter(isnull(col("department")) | isnull(col("salary"))).show()

print("  After fillna:")
df.fillna({"department": "Unknown", "salary": 0.0}).show()

print("  After dropna:")
df.dropna(subset=["department", "salary"]).show()

# ── Conditional Columns ───────────────────────────────────────────────────────
print("=" * 55)
print("CONDITIONAL COLUMNS — when/otherwise")
print("=" * 55)

df.withColumn("salary_band",
    when(col("salary") >= 90000, "High")
    .when(col("salary") >= 70000, "Mid")
    .when(col("salary").isNotNull(), "Low")
    .otherwise("Unknown")
).select("name", "salary", "salary_band").show()

# ── Date Operations ───────────────────────────────────────────────────────────
print("=" * 55)
print("DATE OPERATIONS")
print("=" * 55)

df.withColumn("join_date", to_date(col("join_date"), "yyyy-MM-dd")) \
  .withColumn("join_year",    year(col("join_date"))) \
  .withColumn("join_month",   month(col("join_date"))) \
  .withColumn("days_employed", date_diff(current_date(), col("join_date"))) \
  .select("name", "join_date", "join_year", "join_month", "days_employed") \
  .show()

# ── Filtering & Sorting ───────────────────────────────────────────────────────
print("=" * 55)
print("FILTERING & SORTING")
print("=" * 55)

print("  High earners in Engineering:")
df.filter(
    (col("department") == "Engineering") & (col("salary") > 90000)
).orderBy(col("salary").desc()).show()

# ── Adding & Dropping Columns ─────────────────────────────────────────────────
print("=" * 55)
print("ADD / DROP / RENAME COLUMNS")
print("=" * 55)

result = df \
    .withColumn("bonus", col("salary") * 0.10) \
    .withColumnRenamed("name", "employee_name") \
    .drop("join_date")

result.show()

spark.stop()
