"""
actions/sample.py
-----------------
Demonstrates all major PySpark RDD and DataFrame actions —
operations that trigger execution of the DAG and return
results to the driver.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("ActionsSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Sample RDD ────────────────────────────────────────────────────────────────
numbers = sc.parallelize([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
sales   = sc.parallelize([
    ("Electronics", 1200),
    ("Clothing",     450),
    ("Electronics",  800),
    ("Grocery",      200),
    ("Clothing",     300),
])

print("=" * 55)
print("RDD ACTIONS")
print("=" * 55)

# collect() — returns all elements to driver
print(f"\n  collect()      : {numbers.collect()}")

# count() — returns total number of elements
print(f"  count()        : {numbers.count()}")

# first() — returns the first element
print(f"  first()        : {numbers.first()}")

# take(n) — returns first n elements
print(f"  take(3)        : {numbers.take(3)}")

# top(n) — returns top n elements (descending)
print(f"  top(3)         : {numbers.top(3)}")

# sum() — returns the sum of all elements
print(f"  sum()          : {numbers.sum()}")

# mean() — returns the mean of all elements
print(f"  mean()         : {numbers.mean():.2f}")

# max() / min() — returns max and min
print(f"  max()          : {numbers.max()}")
print(f"  min()          : {numbers.min()}")

# countByValue() — returns frequency of each element
print(f"  countByValue() : {dict(numbers.countByValue())}")

# reduce() — aggregates elements using a function
total = numbers.reduce(lambda a, b: a + b)
print(f"  reduce(+)      : {total}")

# countByKey() — counts occurrences of each key
print(f"  countByKey()   : {dict(sales.countByKey())}")

# collectAsMap() — returns key-value RDD as a dict
kv_rdd = sc.parallelize([("a", 1), ("b", 2), ("c", 3)])
print(f"  collectAsMap() : {kv_rdd.collectAsMap()}")

# foreach() — applies a function to each element (side effect)
print(f"\n  foreach() printing elements:")
numbers.distinct().sortBy(lambda x: x).foreach(lambda x: print(f"    → {x}"))

# ── Sample DataFrame ──────────────────────────────────────────────────────────
df = spark.createDataFrame([
    (1, "Alice",   "Engineering", 95000),
    (2, "Bob",     "Marketing",   72000),
    (3, "Charlie", "Engineering", 88000),
    (4, "Diana",   "HR",          65000),
    (5, "Eve",     "Marketing",   78000),
], ["id", "name", "department", "salary"])

print("\n" + "=" * 55)
print("DATAFRAME ACTIONS")
print("=" * 55)

# show() — prints first n rows
print("\n  show():")
df.show()

# count()
print(f"  count()        : {df.count()}")

# first() — returns first Row object
print(f"  first()        : {df.first()}")

# take(n) — returns list of Row objects
print(f"  take(2)        : {df.take(2)}")

# collect() — returns all rows as list of Row objects
rows = df.collect()
print(f"  collect() count: {len(rows)} rows returned to driver")

# describe() — summary statistics
print("\n  describe():")
df.describe("salary").show()

spark.stop()
