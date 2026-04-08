"""
rdds/sample.py
--------------
Demonstrates core RDD operations in PySpark including creation,
transformations, and actions on Resilient Distributed Datasets.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RDDSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Create RDDs ───────────────────────────────────────────────────────────────
numbers = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
words   = sc.parallelize(["spark", "rdd", "python", "bigdata", "spark", "rdd"])

# ── Transformations ───────────────────────────────────────────────────────────
squared     = numbers.map(lambda x: x ** 2)
evens       = numbers.filter(lambda x: x % 2 == 0)
word_pairs  = words.map(lambda w: (w, 1))
word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

# ── Actions ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("ORIGINAL NUMBERS")
print("=" * 50)
print(numbers.collect())

print("\n" + "=" * 50)
print("SQUARED NUMBERS")
print("=" * 50)
print(squared.collect())

print("\n" + "=" * 50)
print("EVEN NUMBERS")
print("=" * 50)
print(evens.collect())

print("\n" + "=" * 50)
print("WORD COUNTS")
print("=" * 50)
for word, count in word_counts.sortBy(lambda x: x[1], ascending=False).collect():
    print(f"  {word:<10} : {count}")

print("\n" + "=" * 50)
print("RDD STATS")
print("=" * 50)
print(f"  Count  : {numbers.count()}")
print(f"  Sum    : {numbers.sum()}")
print(f"  Mean   : {numbers.mean()}")
print(f"  Max    : {numbers.max()}")
print(f"  Min    : {numbers.min()}")

spark.stop()
