"""
aggregations/sample.py
-----------------------
Demonstrates PySpark aggregation operations including groupBy,
window functions, pivot, rollup, and cube aggregations.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, sum, min, max, round,
    stddev, collect_list, first,
    row_number, rank, dense_rank, lag, lead
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("AggregationsSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample Data ───────────────────────────────────────────────────────────────
data = [
    ("Alice",   "Engineering", "NYC", 95000, 2021),
    ("Bob",     "Marketing",   "LA",  72000, 2021),
    ("Charlie", "Engineering", "NYC", 88000, 2022),
    ("Diana",   "HR",          "NYC", 65000, 2021),
    ("Eve",     "Marketing",   "SF",  78000, 2022),
    ("Frank",   "Engineering", "SF",  102000,2022),
    ("Grace",   "HR",          "LA",  68000, 2022),
    ("Hank",    "Engineering", "NYC", 91000, 2023),
    ("Iris",    "Marketing",   "SF",  83000, 2023),
    ("Jack",    "HR",          "NYC", 71000, 2023),
]
df = spark.createDataFrame(data, ["name", "department", "city", "salary", "year"])

# ── Basic GroupBy Aggregations ────────────────────────────────────────────────
print("=" * 55)
print("BASIC GROUP BY AGGREGATIONS")
print("=" * 55)

df.groupBy("department").agg(
    count("*").alias("headcount"),
    round(avg("salary"), 0).alias("avg_salary"),
    sum("salary").alias("total_salary"),
    min("salary").alias("min_salary"),
    max("salary").alias("max_salary"),
    round(stddev("salary"), 2).alias("salary_stddev"),
).orderBy(col("avg_salary").desc()).show()

# ── Multi-column GroupBy ──────────────────────────────────────────────────────
print("=" * 55)
print("MULTI-COLUMN GROUP BY (department + year)")
print("=" * 55)

df.groupBy("department", "year").agg(
    count("*").alias("headcount"),
    round(avg("salary"), 0).alias("avg_salary"),
).orderBy("department", "year").show()

# ── Collect List ──────────────────────────────────────────────────────────────
print("=" * 55)
print("COLLECT LIST — Group names by department")
print("=" * 55)

df.groupBy("department").agg(
    collect_list("name").alias("members")
).show(truncate=False)

# ── Window Functions ──────────────────────────────────────────────────────────
print("=" * 55)
print("WINDOW FUNCTIONS")
print("=" * 55)

win_dept = Window.partitionBy("department").orderBy(col("salary").desc())
win_all  = Window.orderBy(col("salary").desc())

windowed = df.withColumn("rank_in_dept",  rank().over(win_dept)) \
             .withColumn("dense_rank",    dense_rank().over(win_dept)) \
             .withColumn("row_num",       row_number().over(win_dept)) \
             .withColumn("overall_rank",  rank().over(win_all))

print("  Rank within department:")
windowed.select("name", "department", "salary", "rank_in_dept", "dense_rank", "overall_rank") \
    .orderBy("department", "rank_in_dept").show()

# ── Lag & Lead ────────────────────────────────────────────────────────────────
print("=" * 55)
print("LAG & LEAD — Salary comparison within dept")
print("=" * 55)

win_year = Window.partitionBy("department").orderBy("year")
df.withColumn("prev_year_salary", lag("salary", 1).over(win_year)) \
  .withColumn("next_year_salary", lead("salary", 1).over(win_year)) \
  .select("name", "department", "year", "salary", "prev_year_salary", "next_year_salary") \
  .orderBy("department", "year").show()

# ── Pivot ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("PIVOT — Avg salary by department per year")
print("=" * 55)

df.groupBy("department") \
  .pivot("year", [2021, 2022, 2023]) \
  .agg(round(avg("salary"), 0)) \
  .show()

# ── Rollup ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("ROLLUP — Subtotals by department and city")
print("=" * 55)

df.rollup("department", "city").agg(
    count("*").alias("headcount"),
    round(avg("salary"), 0).alias("avg_salary")
).orderBy("department", "city").show()

spark.stop()
