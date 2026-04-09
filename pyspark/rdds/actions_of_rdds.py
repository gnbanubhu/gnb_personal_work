"""
rdds/actions_of_rdds.py
------------------------
Demonstrates ALL available RDD actions in PySpark.

Actions trigger the execution of the DAG and return a result
to the driver or write data to external storage.

Categories:
  1.  Collection Actions
  2.  Counting Actions
  3.  Aggregation Actions
  4.  Statistical Actions
  5.  Search & Lookup Actions
  6.  Side-Effect Actions
  7.  Save Actions

Usage:
    python actions_of_rdds.py
"""

import tempfile, os
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RDDActions") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def sub(title):
    print(f"\n  ── {title}")


# ── Base RDDs ──────────────────────────────────────────────────────────────────
numbers  = sc.parallelize([5, 3, 8, 1, 9, 2, 7, 4, 6, 10])
words    = sc.parallelize(["spark", "rdd", "python", "bigdata", "spark", "rdd", "spark"])
pairs    = sc.parallelize([("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5), ("c", 6)])
floats   = sc.parallelize([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — COLLECTION ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 1 — COLLECTION ACTIONS")

# 1. collect()
# Returns all elements of the RDD as a Python list to the driver.
# WARNING: Use only on small RDDs — loads all data into driver memory.
sub("1. collect()")
result = numbers.collect()
print(f"     numbers.collect()  : {result}")

# 2. first()
# Returns the first element of the RDD.
sub("2. first()")
result = numbers.first()
print(f"     numbers.first()    : {result}")

# 3. take(n)
# Returns the first n elements of the RDD as a list.
sub("3. take(n)")
result = numbers.take(4)
print(f"     numbers.take(4)    : {result}")

# 4. takeSample(withReplacement, num, seed)
# Returns a random sample of num elements.
# withReplacement=True allows the same element to be sampled more than once.
sub("4. takeSample(withReplacement, num, seed)")
result_no_replace  = numbers.takeSample(withReplacement=False, num=5, seed=42)
result_with_replace = numbers.takeSample(withReplacement=True,  num=5, seed=42)
print(f"     takeSample(False, 5) : {result_no_replace}")
print(f"     takeSample(True,  5) : {result_with_replace}")

# 5. takeOrdered(n, key)
# Returns the first n elements in ascending order (or by custom key function).
sub("5. takeOrdered(n, key)")
asc_result  = numbers.takeOrdered(5)
desc_result = numbers.takeOrdered(5, key=lambda x: -x)
print(f"     takeOrdered(5)           : {asc_result}")
print(f"     takeOrdered(5, desc)     : {desc_result}")

# 6. top(n, key)
# Returns the top n elements in descending order (or by custom key function).
sub("6. top(n, key)")
top_result  = numbers.top(5)
top_custom  = numbers.top(5, key=lambda x: -x)
print(f"     top(5)                   : {top_result}")
print(f"     top(5, key=-x) (smallest): {top_custom}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — COUNTING ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 2 — COUNTING ACTIONS")

# 7. count()
# Returns the total number of elements in the RDD.
sub("7. count()")
result = numbers.count()
print(f"     numbers.count()    : {result}")

# 8. countByValue()
# Returns a dict of {element: count} for each unique element.
sub("8. countByValue()")
result = words.countByValue()
print(f"     Input              : {words.collect()}")
print(f"     countByValue()     :")
for word, count in sorted(result.items(), key=lambda x: -x[1]):
    print(f"       {word:<12} : {count}")

# 9. countByKey()
# Returns a dict of {key: count} for each unique key in a Pair RDD.
sub("9. countByKey()")
result = pairs.countByKey()
print(f"     Input              : {pairs.collect()}")
print(f"     countByKey()       :")
for key, count in sorted(result.items()):
    print(f"       {key:<5} : {count}")

# 10. countApprox(timeout, confidence)
# Returns an approximate count within a timeout (useful for very large RDDs).
sub("10. countApprox(timeout, confidence)")
result = numbers.countApprox(timeout=1000, confidence=0.95)
print(f"     countApprox(1000ms, 0.95) : ~{result}")

# 11. countApproxDistinct(relativeSD)
# Returns an approximate count of distinct elements.
sub("11. countApproxDistinct(relativeSD)")
result = words.countApproxDistinct(relativeSD=0.05)
print(f"     Input                       : {words.collect()}")
print(f"     countApproxDistinct(0.05)   : ~{result}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — AGGREGATION ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 3 — AGGREGATION ACTIONS")

# 12. reduce(func)
# Aggregates all elements using an associative and commutative binary function.
sub("12. reduce(func)")
total   = numbers.reduce(lambda a, b: a + b)
product = numbers.reduce(lambda a, b: a * b)
print(f"     Input                : {numbers.collect()}")
print(f"     reduce(sum)          : {total}")
print(f"     reduce(product)      : {product}")

# 13. fold(zeroValue, func)
# Like reduce() but with an initial zero value applied per partition and overall.
sub("13. fold(zeroValue, func)")
result = numbers.fold(0, lambda a, b: a + b)
print(f"     fold(0, sum)         : {result}")

# 14. aggregate(zeroValue, seqOp, combOp)
# Most general aggregation — uses different functions for within-partition
# and across-partition aggregation. Can return a different type from input.
sub("14. aggregate(zeroValue, seqOp, combOp)")
zero     = (0, 0)                                              # (sum, count)
seq_op   = lambda acc, val: (acc[0] + val,  acc[1] + 1)       # within partition
comb_op  = lambda a,   b:   (a[0]   + b[0], a[1]   + b[1])   # across partitions
sum_cnt  = numbers.aggregate(zero, seq_op, comb_op)
avg      = sum_cnt[0] / sum_cnt[1]
print(f"     aggregate(sum, count): sum={sum_cnt[0]}, count={sum_cnt[1]}")
print(f"     computed average     : {avg}")

# 15. sum()
# Returns the sum of all numeric elements.
sub("15. sum()")
result = numbers.sum()
print(f"     numbers.sum()        : {result}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — STATISTICAL ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 4 — STATISTICAL ACTIONS")

# 16. mean()
# Returns the arithmetic mean of all numeric elements.
sub("16. mean()")
result = numbers.mean()
print(f"     numbers.mean()       : {result}")

# 17. max()
# Returns the maximum element.
sub("17. max()")
result = numbers.max()
print(f"     numbers.max()        : {result}")

# 18. min()
# Returns the minimum element.
sub("18. min()")
result = numbers.min()
print(f"     numbers.min()        : {result}")

# 19. variance()
# Returns the population variance of the numeric elements.
sub("19. variance()")
result = numbers.variance()
print(f"     numbers.variance()   : {result:.4f}")

# 20. stdev()
# Returns the population standard deviation of the numeric elements.
sub("20. stdev()")
result = numbers.stdev()
print(f"     numbers.stdev()      : {result:.4f}")

# 21. sampleVariance()
# Returns the sample variance (uses N-1 denominator).
sub("21. sampleVariance()")
result = numbers.sampleVariance()
print(f"     sampleVariance()     : {result:.4f}")

# 22. sampleStdev()
# Returns the sample standard deviation (uses N-1 denominator).
sub("22. sampleStdev()")
result = numbers.sampleStdev()
print(f"     sampleStdev()        : {result:.4f}")

# 23. stats()
# Returns a StatCounter object with count, mean, stdev, max, min in one call.
sub("23. stats()")
stats = numbers.stats()
print(f"     stats().count  : {stats.count()}")
print(f"     stats().mean   : {stats.mean()}")
print(f"     stats().stdev  : {stats.stdev():.4f}")
print(f"     stats().max    : {stats.max()}")
print(f"     stats().min    : {stats.min()}")
print(f"     stats().sum    : {stats.sum()}")
print(f"     stats()        : {stats}")

# 24. histogram(buckets)
# Computes a histogram of the RDD elements.
# buckets can be an int (number of equal-width buckets) or a list of boundaries.
sub("24. histogram(buckets)")
buckets, counts = numbers.histogram(5)
print(f"     histogram(5 buckets):")
for i in range(len(counts)):
    print(f"       [{buckets[i]:.1f} - {buckets[i+1]:.1f}] : {'█' * counts[i]} ({counts[i]})")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — SEARCH & LOOKUP ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 5 — SEARCH & LOOKUP ACTIONS")

# 25. lookup(key)
# Returns a list of values associated with a given key in a Pair RDD.
sub("25. lookup(key)")
result = pairs.lookup("a")
print(f"     Input              : {pairs.collect()}")
print(f"     lookup('a')        : {result}")
print(f"     lookup('b')        : {pairs.lookup('b')}")
print(f"     lookup('c')        : {pairs.lookup('c')}")

# 26. collectAsMap()
# Returns the Pair RDD as a Python dict — last value wins for duplicate keys.
sub("26. collectAsMap()")
result = pairs.collectAsMap()
print(f"     Input              : {pairs.collect()}")
print(f"     collectAsMap()     : {result}")
print(f"     Note: duplicate keys ('a','b','c') → last value wins")

# 27. isEmpty()
# Returns True if the RDD has no elements.
sub("27. isEmpty()")
empty_rdd = sc.emptyRDD()
print(f"     numbers.isEmpty()  : {numbers.isEmpty()}")
print(f"     emptyRDD.isEmpty() : {empty_rdd.isEmpty()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6 — SIDE-EFFECT ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 6 — SIDE-EFFECT ACTIONS")

# 28. foreach(func)
# Applies a function to each element — used for side effects (e.g., write to DB).
# Runs on executors, NOT the driver — print output may not appear in order.
sub("28. foreach(func)")
print(f"     Input              : {numbers.collect()}")
print(f"     foreach(print) — output from executors (order may vary):")
sc.parallelize([1, 2, 3]).foreach(lambda x: print(f"       executor processed: {x}"))

# 29. foreachPartition(func)
# Like foreach but applies a function to each partition as an iterator.
# More efficient for batch operations (e.g., open DB connection once per partition).
sub("29. foreachPartition(func)")
def process_partition(iterator):
    elements = list(iterator)
    print(f"       partition elements: {elements}")

sc.parallelize([1, 2, 3, 4, 5, 6], 3).foreachPartition(process_partition)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7 — SAVE ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 7 — SAVE ACTIONS")

output_dir = tempfile.mkdtemp()

# 30. saveAsTextFile(path)
# Saves each element of the RDD as a line in a text file.
# Writes one file per partition into the given directory.
sub("30. saveAsTextFile(path)")
text_path = os.path.join(output_dir, "text_output")
numbers.saveAsTextFile(text_path)
saved_files = [f for f in os.listdir(text_path) if f.startswith("part-")]
print(f"     saveAsTextFile() → {text_path}")
print(f"     Files written    : {sorted(saved_files)}")
# Read back one part to verify
sample_file = os.path.join(text_path, sorted(saved_files)[0])
with open(sample_file) as f:
    lines = f.readlines()
print(f"     Sample content   : {[l.strip() for l in lines]}")

# 31. saveAsPickleFile(path)
# Saves the RDD as a serialized Python pickle file (preserves Python types).
sub("31. saveAsPickleFile(path)")
pickle_path = os.path.join(output_dir, "pickle_output")
numbers.saveAsPickleFile(pickle_path)
pickle_files = [f for f in os.listdir(pickle_path) if f.startswith("part-")]
print(f"     saveAsPickleFile() → {pickle_path}")
print(f"     Files written      : {sorted(pickle_files)}")
# Read back to verify
loaded_rdd = sc.pickleFile(pickle_path)
print(f"     Loaded back        : {sorted(loaded_rdd.collect())}")

# 32. saveAsSequenceFile(path)
# Saves a Pair RDD as a Hadoop SequenceFile (binary key-value format).
sub("32. saveAsSequenceFile(path)")
seq_path = os.path.join(output_dir, "sequence_output")
pairs.saveAsSequenceFile(seq_path)
seq_files = [f for f in os.listdir(seq_path) if f.startswith("part-")]
print(f"     saveAsSequenceFile() → {seq_path}")
print(f"     Files written        : {sorted(seq_files)}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("ACTIONS SUMMARY")
print("""
  #   Action                    Category
  ────────────────────────────────────────────────────────────
  1   collect()                 Collection
  2   first()                   Collection
  3   take(n)                   Collection
  4   takeSample()              Collection
  5   takeOrdered(n)            Collection
  6   top(n)                    Collection
  7   count()                   Counting
  8   countByValue()            Counting
  9   countByKey()              Counting
  10  countApprox()             Counting
  11  countApproxDistinct()     Counting
  12  reduce(func)              Aggregation
  13  fold(zeroValue, func)     Aggregation
  14  aggregate()               Aggregation
  15  sum()                     Aggregation
  16  mean()                    Statistical
  17  max()                     Statistical
  18  min()                     Statistical
  19  variance()                Statistical
  20  stdev()                   Statistical
  21  sampleVariance()          Statistical
  22  sampleStdev()             Statistical
  23  stats()                   Statistical
  24  histogram(buckets)        Statistical
  25  lookup(key)               Search & Lookup
  26  collectAsMap()            Search & Lookup
  27  isEmpty()                 Search & Lookup
  28  foreach(func)             Side-Effect
  29  foreachPartition(func)    Side-Effect
  30  saveAsTextFile()          Save
  31  saveAsPickleFile()        Save
  32  saveAsSequenceFile()      Save
  ────────────────────────────────────────────────────────────
  Total : 32 actions
""")

spark.stop()
