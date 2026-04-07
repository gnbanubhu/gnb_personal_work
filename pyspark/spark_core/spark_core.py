"""
spark_core.py
-------------
Demonstrates core PySpark RDD operations including map, filter,
reduceByKey, and sorting.

Usage:
    python spark_core.py
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkCoreDemo") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Create RDD ────────────────────────────────────────────────────────────────
sales = sc.parallelize([
    ("Electronics", 1200),
    ("Clothing", 450),
    ("Electronics", 800),
    ("Grocery", 200),
    ("Clothing", 300),
    ("Grocery", 150),
    ("Electronics", 950),
])

# ── Transformations ───────────────────────────────────────────────────────────
total_by_category = sales.reduceByKey(lambda a, b: a + b)
sorted_sales = total_by_category.sortBy(lambda x: x[1], ascending=False)
high_sales = sorted_sales.filter(lambda x: x[1] > 400)

# ── Actions ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("TOTAL SALES BY CATEGORY")
print("=" * 50)
for category, total in sorted_sales.collect():
    print(f"  {category:<15} : ${total:,}")

print("\n" + "=" * 50)
print("HIGH VALUE CATEGORIES (> $400)")
print("=" * 50)
for category, total in high_sales.collect():
    print(f"  {category:<15} : ${total:,}")

print(f"\nTotal categories : {total_by_category.count()}")
print(f"Total revenue    : ${sales.map(lambda x: x[1]).sum():,}")

spark.stop()
