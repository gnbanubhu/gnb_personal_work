"""
spark_r/sample.py
------------------
Demonstrates SparkR-equivalent operations implemented in PySpark —
statistical summaries, correlation, sampling, and data distribution
analysis commonly used in R-style data exploration workflows.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, mean, stddev, skewness, kurtosis,
    corr, round, percentile_approx, count,
    when, lit
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder \
    .appName("SparkRSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample Dataset ────────────────────────────────────────────────────────────
schema = StructType([
    StructField("id",          IntegerType(), False),
    StructField("name",        StringType(),  False),
    StructField("department",  StringType(),  True),
    StructField("salary",      DoubleType(),  True),
    StructField("experience",  IntegerType(), True),
    StructField("performance", DoubleType(),  True),
])

data = [
    (1,  "Alice",   "Engineering", 95000.0,  5, 4.5),
    (2,  "Bob",     "Marketing",   72000.0,  3, 3.8),
    (3,  "Charlie", "Engineering", 88000.0,  7, 4.2),
    (4,  "Diana",   "HR",          65000.0,  2, 3.5),
    (5,  "Eve",     "Marketing",   78000.0,  4, 4.0),
    (6,  "Frank",   "Engineering", 102000.0, 9, 4.8),
    (7,  "Grace",   "HR",          68000.0,  3, 3.6),
    (8,  "Hank",    "Engineering", 91000.0,  6, 4.3),
    (9,  "Iris",    "Marketing",   83000.0,  5, 4.1),
    (10, "Jack",    "HR",          71000.0,  4, 3.9),
    (11, "Karen",   "Engineering", 97000.0,  7, 4.6),
    (12, "Leo",     "Marketing",   69000.0,  2, 3.7),
]

df = spark.createDataFrame(data, schema)

# ── Summary Statistics (R: summary()) ─────────────────────────────────────────
print("=" * 55)
print("SUMMARY STATISTICS  (R equivalent: summary())")
print("=" * 55)
df.describe("salary", "experience", "performance").show()

# ── Distribution Analysis ─────────────────────────────────────────────────────
print("=" * 55)
print("DISTRIBUTION ANALYSIS  (R equivalent: hist / quantile)")
print("=" * 55)

df.select(
    round(mean("salary"),    2).alias("mean"),
    round(stddev("salary"),  2).alias("std_dev"),
    round(skewness("salary"),2).alias("skewness"),
    round(kurtosis("salary"),2).alias("kurtosis"),
    percentile_approx("salary", 0.25).alias("Q1"),
    percentile_approx("salary", 0.50).alias("median"),
    percentile_approx("salary", 0.75).alias("Q3"),
    percentile_approx("salary", [0.05, 0.95]).alias("5th_95th_pct"),
).show(truncate=False)

# ── Correlation Matrix (R: cor()) ─────────────────────────────────────────────
print("=" * 55)
print("CORRELATION  (R equivalent: cor())")
print("=" * 55)

salary_exp_corr  = df.select(corr("salary", "experience")).collect()[0][0]
salary_perf_corr = df.select(corr("salary", "performance")).collect()[0][0]
exp_perf_corr    = df.select(corr("experience", "performance")).collect()[0][0]

print(f"  salary    vs experience  : {salary_exp_corr:.4f}")
print(f"  salary    vs performance : {salary_perf_corr:.4f}")
print(f"  experience vs performance: {exp_perf_corr:.4f}")

# ── Group-level Stats (R: tapply / aggregate) ─────────────────────────────────
print("\n" + "=" * 55)
print("GROUP STATISTICS  (R equivalent: tapply / aggregate)")
print("=" * 55)

df.groupBy("department").agg(
    count("*").alias("n"),
    round(mean("salary"),      2).alias("mean_salary"),
    round(stddev("salary"),    2).alias("sd_salary"),
    round(mean("experience"),  2).alias("mean_exp"),
    round(mean("performance"), 2).alias("mean_perf"),
).orderBy("department").show()

# ── Sampling (R: sample()) ────────────────────────────────────────────────────
print("=" * 55)
print("SAMPLING  (R equivalent: sample())")
print("=" * 55)

sampled = df.sample(withReplacement=False, fraction=0.5, seed=42)
print(f"  Original rows : {df.count()}")
print(f"  Sampled rows  : {sampled.count()}  (50% sample, seed=42)")
sampled.select("name", "department", "salary").show()

# ── Frequency Table (R: table()) ──────────────────────────────────────────────
print("=" * 55)
print("FREQUENCY TABLE  (R equivalent: table())")
print("=" * 55)

df.withColumn("salary_band",
    when(col("salary") >= 90000, "High  (≥90K)")
    .when(col("salary") >= 75000, "Mid   (75K-90K)")
    .otherwise("Low   (<75K)")
).groupBy("salary_band").agg(
    count("*").alias("frequency")
).orderBy("salary_band").show(truncate=False)

# ── Cross Tabulation (R: xtabs / table()) ─────────────────────────────────────
print("=" * 55)
print("CROSS TABULATION  (R equivalent: xtabs())")
print("=" * 55)
df.stat.crosstab("department", "experience").show()

spark.stop()
