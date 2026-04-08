"""
optimization/sample.py
----------------------
Demonstrates PySpark performance optimization techniques including
caching, partitioning, predicate pushdown, and avoiding shuffles.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum

spark = SparkSession.builder \
    .appName("OptimizationSample") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample Data ───────────────────────────────────────────────────────────────
data = [
    (1, "Alice",   "Engineering", 95000,  "NYC"),
    (2, "Bob",     "Marketing",   72000,  "LA"),
    (3, "Charlie", "Engineering", 88000,  "NYC"),
    (4, "Diana",   "HR",          65000,  "NYC"),
    (5, "Eve",     "Marketing",   78000,  "SF"),
    (6, "Frank",   "Engineering", 102000, "SF"),
    (7, "Grace",   "HR",          68000,  "LA"),
    (8, "Hank",    "Engineering", 91000,  "NYC"),
    (9, "Iris",    "Marketing",   83000,  "SF"),
    (10,"Jack",    "HR",          71000,  "NYC"),
]

df = spark.createDataFrame(data, ["id", "name", "department", "salary", "city"])

# ── OPTIMIZATION 1: Caching ───────────────────────────────────────────────────
print("=" * 55)
print("OPTIMIZATION 1 — CACHING")
print("=" * 55)

df.cache()   # cache after first action — reused across multiple queries

dept_stats = df.groupBy("department") \
    .agg(
        count("*").alias("headcount"),
        avg("salary").alias("avg_salary"),
        sum("salary").alias("total_salary")
    )

city_stats = df.groupBy("city") \
    .agg(count("*").alias("headcount"), avg("salary").alias("avg_salary"))

print("Department Stats (from cache):")
dept_stats.orderBy(col("avg_salary").desc()).show()

print("City Stats (from cache — no recomputation):")
city_stats.orderBy(col("avg_salary").desc()).show()

# ── OPTIMIZATION 2: Predicate Pushdown ───────────────────────────────────────
print("=" * 55)
print("OPTIMIZATION 2 — PREDICATE PUSHDOWN")
print("=" * 55)

# Filter early — reduces rows before expensive operations
high_earners = df.filter(col("salary") > 80000) \
    .groupBy("department") \
    .agg(count("*").alias("high_earner_count"))

print("High Earners (> $80K) per Department:")
high_earners.orderBy(col("high_earner_count").desc()).show()

# ── OPTIMIZATION 3: Avoid Wide Transformations Where Possible ─────────────────
print("=" * 55)
print("OPTIMIZATION 3 — COLUMN SELECTION (Column Pruning)")
print("=" * 55)

# Select only required columns early — reduces data size through the pipeline
slim_df = df.select("name", "department", "salary")

top_earners = slim_df.orderBy(col("salary").desc()).limit(3)

print("Top 3 Earners (only required columns passed through pipeline):")
top_earners.show()

# ── OPTIMIZATION 4: Repartition for Balanced Processing ──────────────────────
print("=" * 55)
print("OPTIMIZATION 4 — REPARTITIONING")
print("=" * 55)

print(f"  Default partitions : {df.rdd.getNumPartitions()}")

repartitioned = df.repartition(4, "department")  # colocate same dept on same partition
print(f"  After repartition  : {repartitioned.rdd.getNumPartitions()}")
print(f"  Partitioned by     : department (reduces shuffle in groupBy later)")

# ── OPTIMIZATION 5: Cache Cleanup ────────────────────────────────────────────
df.unpersist()
print("\n  Cache unpersisted — memory freed.")

spark.stop()
