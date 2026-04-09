"""
rdds/transformations_of_rdds.py
--------------------------------
Demonstrates ALL available RDD transformations in PySpark.

Transformations are lazy — they build the DAG but do NOT execute
until an action (collect, count, show, etc.) is called.

Categories:
  1.  Element-wise Transformations
  2.  Filtering & Set Transformations
  3.  Key-Value (Pair RDD) Transformations
  4.  Join Transformations
  5.  Aggregation Transformations
  6.  Sorting Transformations
  7.  Partitioning Transformations
  8.  Zipping Transformations
  9.  Structural Transformations

Usage:
    python transformations_of_rdds.py
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RDDTransformations") \
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
numbers  = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
words    = sc.parallelize(["spark", "rdd", "python", "bigdata", "spark", "rdd"])
pairs    = sc.parallelize([("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5)])
pairs2   = sc.parallelize([("a", 10), ("b", 20), ("d", 30)])
sentences = sc.parallelize(["hello world", "spark rdd", "python bigdata"])
nested   = sc.parallelize([[1, 2], [3, 4], [5, 6]])


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — ELEMENT-WISE TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 1 — ELEMENT-WISE TRANSFORMATIONS")

# 1. map()
# Applies a function to each element → returns one output per input
sub("1. map()")
squared = numbers.map(lambda x: x ** 2)
print(f"     Input   : {numbers.collect()}")
print(f"     map(x²) : {squared.collect()}")

# 2. flatMap()
# Applies a function to each element → flattens the result (one-to-many)
sub("2. flatMap()")
flat_words = sentences.flatMap(lambda s: s.split(" "))
print(f"     Input       : {sentences.collect()}")
print(f"     flatMap(split): {flat_words.collect()}")

# 3. mapPartitions()
# Applies a function to each partition as an iterator (more efficient than map)
sub("3. mapPartitions()")
def double_partition(iterator):
    yield from (x * 2 for x in iterator)

doubled = numbers.mapPartitions(double_partition)
print(f"     Input              : {numbers.collect()}")
print(f"     mapPartitions(x*2) : {doubled.collect()}")

# 4. mapPartitionsWithIndex()
# Like mapPartitions but also provides the partition index
sub("4. mapPartitionsWithIndex()")
def add_partition_index(index, iterator):
    yield from ((index, x) for x in iterator)

indexed = numbers.mapPartitionsWithIndex(add_partition_index)
print(f"     mapPartitionsWithIndex (partition_id, value):")
for item in indexed.collect()[:5]:
    print(f"       {item}")

# 5. keyBy()
# Creates a Pair RDD by applying a function to each element as the key
sub("5. keyBy()")
keyed = numbers.keyBy(lambda x: "even" if x % 2 == 0 else "odd")
print(f"     Input         : {numbers.collect()}")
print(f"     keyBy(even/odd): {keyed.collect()}")

# 6. glom()
# Coalesces each partition into a list — returns RDD of lists
sub("6. glom()")
small_rdd  = sc.parallelize([1, 2, 3, 4, 5, 6], 3)
glommed    = small_rdd.glom()
print(f"     Input (3 partitions) : {small_rdd.collect()}")
print(f"     glom()               : {glommed.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — FILTERING & SET TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 2 — FILTERING & SET TRANSFORMATIONS")

# 7. filter()
# Returns only elements matching a condition
sub("7. filter()")
evens = numbers.filter(lambda x: x % 2 == 0)
print(f"     Input         : {numbers.collect()}")
print(f"     filter(even)  : {evens.collect()}")

# 8. distinct()
# Removes duplicate elements
sub("8. distinct()")
print(f"     Input         : {words.collect()}")
print(f"     distinct()    : {words.distinct().collect()}")

# 9. sample()
# Returns a random sample of the RDD (withReplacement, fraction, seed)
sub("9. sample()")
sampled = numbers.sample(withReplacement=False, fraction=0.5, seed=42)
print(f"     Input               : {numbers.collect()}")
print(f"     sample(frac=0.5)    : {sampled.collect()}")

# 10. union()
# Combines two RDDs without removing duplicates
sub("10. union()")
rdd1      = sc.parallelize([1, 2, 3])
rdd2      = sc.parallelize([3, 4, 5])
union_rdd = rdd1.union(rdd2)
print(f"     rdd1          : {rdd1.collect()}")
print(f"     rdd2          : {rdd2.collect()}")
print(f"     union()       : {union_rdd.collect()}")

# 11. intersection()
# Returns elements present in both RDDs (removes duplicates)
sub("11. intersection()")
intersection_rdd = rdd1.intersection(rdd2)
print(f"     rdd1          : {rdd1.collect()}")
print(f"     rdd2          : {rdd2.collect()}")
print(f"     intersection(): {intersection_rdd.collect()}")

# 12. subtract()
# Returns elements in first RDD that are NOT in the second
sub("12. subtract()")
subtract_rdd = rdd1.subtract(rdd2)
print(f"     rdd1          : {rdd1.collect()}")
print(f"     rdd2          : {rdd2.collect()}")
print(f"     subtract()    : {subtract_rdd.collect()}")

# 13. cartesian()
# Returns every combination (cross product) of elements from two RDDs
sub("13. cartesian()")
small1      = sc.parallelize([1, 2])
small2      = sc.parallelize(["a", "b"])
cartesian   = small1.cartesian(small2)
print(f"     rdd1          : {small1.collect()}")
print(f"     rdd2          : {small2.collect()}")
print(f"     cartesian()   : {cartesian.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — KEY-VALUE (PAIR RDD) TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 3 — KEY-VALUE (PAIR RDD) TRANSFORMATIONS")

# 14. keys()
# Returns only the keys from a Pair RDD
sub("14. keys()")
print(f"     Input    : {pairs.collect()}")
print(f"     keys()   : {pairs.keys().collect()}")

# 15. values()
# Returns only the values from a Pair RDD
sub("15. values()")
print(f"     values() : {pairs.values().collect()}")

# 16. mapValues()
# Applies a function to values only — keeps keys unchanged
sub("16. mapValues()")
mapped_values = pairs.mapValues(lambda v: v * 10)
print(f"     Input          : {pairs.collect()}")
print(f"     mapValues(x*10): {mapped_values.collect()}")

# 17. flatMapValues()
# Like flatMap but applies only to values — keeps keys, flattens value results
sub("17. flatMapValues()")
pair_lists  = sc.parallelize([("a", [1, 2, 3]), ("b", [4, 5])])
flat_vals   = pair_lists.flatMapValues(lambda v: v)
print(f"     Input             : {pair_lists.collect()}")
print(f"     flatMapValues()   : {flat_vals.collect()}")

# 18. groupByKey()
# Groups all values for each key into an iterable
sub("18. groupByKey()")
grouped = pairs.groupByKey().mapValues(list)
print(f"     Input         : {pairs.collect()}")
print(f"     groupByKey()  : {grouped.collect()}")

# 19. reduceByKey()
# Merges values for each key using an associative reduce function (more efficient than groupByKey)
sub("19. reduceByKey()")
reduced = pairs.reduceByKey(lambda a, b: a + b)
print(f"     Input             : {pairs.collect()}")
print(f"     reduceByKey(sum)  : {reduced.collect()}")

# 20. subtractByKey()
# Returns key-value pairs from first RDD whose keys are NOT in the second
sub("20. subtractByKey()")
subtracted = pairs.subtractByKey(pairs2)
print(f"     pairs             : {pairs.collect()}")
print(f"     pairs2            : {pairs2.collect()}")
print(f"     subtractByKey()   : {subtracted.collect()}")

# 21. partitionBy()
# Repartitions a Pair RDD by key using a custom partitioner
sub("21. partitionBy()")
partitioned = pairs.partitionBy(2)
print(f"     Input              : {pairs.collect()}")
print(f"     partitionBy(2)     : {partitioned.glom().collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — JOIN TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 4 — JOIN TRANSFORMATIONS")

# 22. join()
# Inner join — returns only keys present in both RDDs
sub("22. join() — inner join")
joined = pairs.join(pairs2)
print(f"     pairs             : {pairs.collect()}")
print(f"     pairs2            : {pairs2.collect()}")
print(f"     join()            : {joined.collect()}")

# 23. leftOuterJoin()
# All keys from left RDD; right side is None if key not found
sub("23. leftOuterJoin()")
left_join = pairs.leftOuterJoin(pairs2)
print(f"     leftOuterJoin()   : {left_join.collect()}")

# 24. rightOuterJoin()
# All keys from right RDD; left side is None if key not found
sub("24. rightOuterJoin()")
right_join = pairs.rightOuterJoin(pairs2)
print(f"     rightOuterJoin()  : {right_join.collect()}")

# 25. fullOuterJoin()
# All keys from both RDDs; None on missing side
sub("25. fullOuterJoin()")
full_join = pairs.fullOuterJoin(pairs2)
print(f"     fullOuterJoin()   : {full_join.collect()}")

# 26. cogroup()
# Groups values from multiple Pair RDDs by key into a single RDD
sub("26. cogroup()")
cogrouped = pairs.cogroup(pairs2).mapValues(lambda v: (list(v[0]), list(v[1])))
print(f"     cogroup()         : {cogrouped.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — AGGREGATION TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 5 — AGGREGATION TRANSFORMATIONS")

# 27. aggregateByKey()
# Aggregates values by key using separate functions for within-partition
# and across-partition merging
sub("27. aggregateByKey()")
# Goal: compute (sum, count) per key
zero_value   = (0, 0)
seq_func     = lambda acc, v: (acc[0] + v, acc[1] + 1)    # within partition
merge_func   = lambda a, b:   (a[0] + b[0], a[1] + b[1])  # across partitions
aggregated   = pairs.aggregateByKey(zero_value, seq_func, merge_func)
result       = aggregated.mapValues(lambda v: v[0] / v[1])  # average
print(f"     Input               : {pairs.collect()}")
print(f"     aggregateByKey(avg) : {result.collect()}")

# 28. combineByKey()
# Most general per-key aggregation — create combiner, merge value, merge combiners
sub("28. combineByKey()")
create_combiner = lambda v:    (v, 1)
merge_value     = lambda c, v: (c[0] + v, c[1] + 1)
merge_combiners = lambda c1, c2: (c1[0] + c2[0], c1[1] + c2[1])
combined        = pairs.combineByKey(create_combiner, merge_value, merge_combiners)
avg_by_key      = combined.mapValues(lambda v: round(v[0] / v[1], 2))
print(f"     Input               : {pairs.collect()}")
print(f"     combineByKey(avg)   : {avg_by_key.collect()}")

# 29. foldByKey()
# Like reduceByKey but with a zero/initial value per key
sub("29. foldByKey()")
folded = pairs.foldByKey(0, lambda a, b: a + b)
print(f"     Input               : {pairs.collect()}")
print(f"     foldByKey(0, sum)   : {folded.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6 — SORTING TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 6 — SORTING TRANSFORMATIONS")

# 30. sortBy()
# Sorts an RDD by a key function — works on any RDD
sub("30. sortBy()")
unsorted    = sc.parallelize([5, 1, 9, 3, 7, 2, 8, 4, 6])
asc_sorted  = unsorted.sortBy(lambda x: x)
desc_sorted = unsorted.sortBy(lambda x: x, ascending=False)
print(f"     Input               : {unsorted.collect()}")
print(f"     sortBy(asc)         : {asc_sorted.collect()}")
print(f"     sortBy(desc)        : {desc_sorted.collect()}")

# 31. sortByKey()
# Sorts a Pair RDD by key — only available on Pair RDDs
sub("31. sortByKey()")
unsorted_pairs = sc.parallelize([("c", 3), ("a", 1), ("b", 2), ("e", 5), ("d", 4)])
sorted_pairs   = unsorted_pairs.sortByKey(ascending=True)
print(f"     Input               : {unsorted_pairs.collect()}")
print(f"     sortByKey(asc)      : {sorted_pairs.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7 — PARTITIONING TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 7 — PARTITIONING TRANSFORMATIONS")

# 32. repartition()
# Reshuffles data into a new number of partitions (with full shuffle)
sub("32. repartition()")
rdd_4p = numbers.repartition(4)
print(f"     numbers.getNumPartitions()         : {numbers.getNumPartitions()}")
print(f"     repartition(4).getNumPartitions()  : {rdd_4p.getNumPartitions()}")

# 33. coalesce()
# Reduces partitions with minimal shuffling (no full shuffle when reducing)
sub("33. coalesce()")
rdd_2p = numbers.coalesce(2)
print(f"     numbers.getNumPartitions()         : {numbers.getNumPartitions()}")
print(f"     coalesce(2).getNumPartitions()     : {rdd_2p.getNumPartitions()}")

# 34. pipe()
# Passes each partition through a shell command
sub("34. pipe()")
text_rdd  = sc.parallelize(["hello", "world", "spark"])
piped_rdd = text_rdd.pipe("tr '[:lower:]' '[:upper:]'")
print(f"     Input       : {text_rdd.collect()}")
print(f"     pipe(tr)    : {piped_rdd.collect()}")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 8 — ZIPPING TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 8 — ZIPPING TRANSFORMATIONS")

# 35. zip()
# Combines two RDDs element-by-element into Pair RDD
# Both RDDs must have same number of partitions and elements
sub("35. zip()")
keys_rdd   = sc.parallelize(["a", "b", "c", "d", "e"])
values_rdd = sc.parallelize([10, 20, 30, 40, 50])
zipped     = keys_rdd.zip(values_rdd)
print(f"     keys      : {keys_rdd.collect()}")
print(f"     values    : {values_rdd.collect()}")
print(f"     zip()     : {zipped.collect()}")

# 36. zipWithIndex()
# Zips each element with its index (0-based)
sub("36. zipWithIndex()")
indexed_words = words.zipWithIndex()
print(f"     Input            : {words.collect()}")
print(f"     zipWithIndex()   : {indexed_words.collect()}")

# 37. zipWithUniqueId()
# Zips each element with a unique (not necessarily consecutive) long ID
sub("37. zipWithUniqueId()")
unique_id_words = words.zipWithUniqueId()
print(f"     Input              : {words.collect()}")
print(f"     zipWithUniqueId()  : {unique_id_words.collect()}")

# 38. zip() + mapPartitions() — partition-level zipping (Python alternative)
# zipPartitions() is only available in Scala/Java PySpark API.
# In Python, achieve the same by zipping RDDs and using mapPartitions.
sub("38. zip() + mapPartitions() — partition-level pairing (Python alternative to zipPartitions)")
rdd_a       = sc.parallelize([1, 2, 3, 4], 2)
rdd_b       = sc.parallelize([10, 20, 30, 40], 2)
zipped_ab   = rdd_a.zip(rdd_b).mapPartitions(lambda it: (a + b for a, b in it))
print(f"     rdd_a              : {rdd_a.collect()}")
print(f"     rdd_b              : {rdd_b.collect()}")
print(f"     zip+mapPartitions(+): {zipped_ab.collect()}")
print(f"     Note: zipPartitions() is Scala/Java only — not available in PySpark Python API")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 9 — STRUCTURAL TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("CATEGORY 9 — STRUCTURAL TRANSFORMATIONS")

# 39. cache() / persist()
# Persists the RDD in memory for reuse across multiple actions
sub("39. cache() / persist()")
from pyspark import StorageLevel
cached_rdd    = sc.parallelize(range(1, 11)).cache()
persisted_rdd = sc.parallelize(range(1, 11)).persist(StorageLevel.MEMORY_AND_DISK)
print(f"     cache()                   : RDD persisted in MEMORY_ONLY")
print(f"     persist(MEMORY_AND_DISK)  : RDD persisted in MEMORY + DISK fallback")
print(f"     First action on cached    : {cached_rdd.count()}")
print(f"     Second action (from cache): {cached_rdd.sum()}")
cached_rdd.unpersist()
persisted_rdd.unpersist()

# 40. unpersist()
# Removes the RDD from cache/persistence
sub("40. unpersist()")
temp_rdd = sc.parallelize(range(1, 6)).cache()
temp_rdd.count()            # trigger caching
temp_rdd.unpersist()
print(f"     unpersist() : RDD removed from cache")

# 41. checkpoint()
# Saves RDD to a reliable storage (HDFS/local dir) to truncate long lineage
sub("41. checkpoint()")
import tempfile, os
checkpoint_dir = tempfile.mkdtemp()
sc.setCheckpointDir(checkpoint_dir)
checkpoint_rdd = numbers.map(lambda x: x + 1)
checkpoint_rdd.checkpoint()
checkpoint_rdd.count()      # triggers checkpoint write
print(f"     checkpoint() : RDD lineage truncated, saved to {checkpoint_dir}")
print(f"     isCheckpointed : {checkpoint_rdd.isCheckpointed()}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("TRANSFORMATION SUMMARY")
print("""
  #   Transformation          Category
  ────────────────────────────────────────────────────────────
  1   map()                   Element-wise
  2   flatMap()               Element-wise
  3   mapPartitions()         Element-wise
  4   mapPartitionsWithIndex()Element-wise
  5   keyBy()                 Element-wise
  6   glom()                  Element-wise
  7   filter()                Filtering & Set
  8   distinct()              Filtering & Set
  9   sample()                Filtering & Set
  10  union()                 Filtering & Set
  11  intersection()          Filtering & Set
  12  subtract()              Filtering & Set
  13  cartesian()             Filtering & Set
  14  keys()                  Key-Value (Pair RDD)
  15  values()                Key-Value (Pair RDD)
  16  mapValues()             Key-Value (Pair RDD)
  17  flatMapValues()         Key-Value (Pair RDD)
  18  groupByKey()            Key-Value (Pair RDD)
  19  reduceByKey()           Key-Value (Pair RDD)
  20  subtractByKey()         Key-Value (Pair RDD)
  21  partitionBy()           Key-Value (Pair RDD)
  22  join()                  Join
  23  leftOuterJoin()         Join
  24  rightOuterJoin()        Join
  25  fullOuterJoin()         Join
  26  cogroup()               Join
  27  aggregateByKey()        Aggregation
  28  combineByKey()          Aggregation
  29  foldByKey()             Aggregation
  30  sortBy()                Sorting
  31  sortByKey()             Sorting
  32  repartition()           Partitioning
  33  coalesce()              Partitioning
  34  pipe()                  Partitioning
  35  zip()                   Zipping
  36  zipWithIndex()          Zipping
  37  zipWithUniqueId()       Zipping
  38  zipPartitions()         Zipping
  39  cache() / persist()     Structural
  40  unpersist()             Structural
  41  checkpoint()            Structural
  ────────────────────────────────────────────────────────────
  Total : 41 transformations
""")

spark.stop()
