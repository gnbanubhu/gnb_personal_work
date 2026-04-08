"""
transformations/sample.py
--------------------------
Demonstrates all major PySpark RDD and DataFrame transformations —
lazy operations that build the DAG without triggering execution.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, length, when

spark = SparkSession.builder \
    .appName("TransformationsSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Sample RDD ────────────────────────────────────────────────────────────────
numbers = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
words   = sc.parallelize(["spark", "is", "fast", "spark", "is", "great"])
sales   = sc.parallelize([
    ("Electronics", 1200),
    ("Clothing",     450),
    ("Electronics",  800),
    ("Grocery",      200),
    ("Clothing",     300),
])

print("=" * 55)
print("NARROW TRANSFORMATIONS (No Shuffle)")
print("=" * 55)

# map() — applies function to each element
squared = numbers.map(lambda x: x ** 2)
print(f"\n  map(x²)          : {squared.collect()}")

# filter() — keeps elements matching condition
evens = numbers.filter(lambda x: x % 2 == 0)
print(f"  filter(even)     : {evens.collect()}")

# flatMap() — maps then flattens one level
sentences = sc.parallelize(["hello world", "spark is great"])
all_words  = sentences.flatMap(lambda s: s.split(" "))
print(f"  flatMap(split)   : {all_words.collect()}")

# mapValues() — applies function to values only (key unchanged)
upper_words = words.map(lambda w: (w, 1)).mapValues(lambda v: v * 10)
print(f"  mapValues(x10)   : {upper_words.collect()}")

# distinct() — removes duplicates
unique_words = words.distinct()
print(f"  distinct()       : {sorted(unique_words.collect())}")

# union() — combines two RDDs
rdd1 = sc.parallelize([1, 2, 3])
rdd2 = sc.parallelize([4, 5, 6])
print(f"  union()          : {rdd1.union(rdd2).collect()}")

# sample() — random sample
sampled = numbers.sample(withReplacement=False, fraction=0.5, seed=42)
print(f"  sample(0.5)      : {sampled.collect()}")

print("\n" + "=" * 55)
print("WIDE TRANSFORMATIONS (Shuffle Required)")
print("=" * 55)

# groupByKey() — groups values by key (avoid when possible — use reduceByKey)
grouped = sales.groupByKey().mapValues(list)
print(f"\n  groupByKey()     :")
for k, v in grouped.collect():
    print(f"    {k:<15} : {v}")

# reduceByKey() — more efficient than groupByKey for aggregation
totals = sales.reduceByKey(lambda a, b: a + b)
print(f"  reduceByKey(+)   :")
for k, v in totals.collect():
    print(f"    {k:<15} : ${v:,}")

# sortByKey() — sorts by key
sorted_totals = totals.sortByKey()
print(f"  sortByKey()      : {sorted_totals.collect()}")

# join() — inner join two key-value RDDs
locations = sc.parallelize([
    ("Electronics", "Floor 1"),
    ("Clothing",    "Floor 2"),
    ("Grocery",     "Floor 3"),
])
joined = totals.join(locations)
print(f"  join()           :")
for k, (total, loc) in joined.collect():
    print(f"    {k:<15} : ${total:,} — {loc}")

# coalesce() — reduce partitions without full shuffle
print(f"\n  Partitions before coalesce : {numbers.getNumPartitions()}")
coalesced = numbers.coalesce(2)
print(f"  Partitions after coalesce  : {coalesced.getNumPartitions()}")

print("\n" + "=" * 55)
print("DATAFRAME TRANSFORMATIONS")
print("=" * 55)

df = spark.createDataFrame([
    (1, "Alice",   "Engineering", 95000),
    (2, "Bob",     "Marketing",   72000),
    (3, "Charlie", "Engineering", 88000),
    (4, "Diana",   "HR",          65000),
    (5, "Eve",     "Marketing",   78000),
], ["id", "name", "department", "salary"])

# select() — choose columns
print("\n  select(name, salary):")
df.select("name", "salary").show()

# filter() / where() — row filtering
print("  filter(salary > 75000):")
df.filter(col("salary") > 75000).show()

# withColumn() — add or replace column
print("  withColumn(salary_band):")
df.withColumn("salary_band",
    when(col("salary") >= 90000, "High")
    .when(col("salary") >= 70000, "Mid")
    .otherwise("Low")
).show()

# groupBy() + agg()
print("  groupBy(department).count():")
df.groupBy("department").count().orderBy("count", ascending=False).show()

# orderBy()
print("  orderBy(salary DESC):")
df.orderBy(col("salary").desc()).show()

spark.stop()
