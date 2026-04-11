"""
pyspark/performance_tuning/spark_tune.py
-----------------------------------------
Comprehensive PySpark Performance Tuning guide covering:

  1.  SparkSession Configuration     — memory, cores, shuffle settings
  2.  Partitioning                    — repartition vs coalesce, optimal partition count
  3.  Caching & Persistence          — StorageLevel strategies
  4.  Broadcast Joins                — small table broadcast to avoid shuffle
  5.  Adaptive Query Execution (AQE) — dynamic optimization at runtime
  6.  Predicate Pushdown             — filter early, read less
  7.  Column Pruning                 — select only needed columns
  8.  Bucketing                      — pre-shuffle data for repeated joins
  9.  Skew Handling                  — salting technique for skewed keys
  10. Shuffle Optimization           — reduce shuffle partitions, sort-merge vs hash
  11. Execution Plan Analysis        — explain(), logical vs physical plan
  12. Speculative Execution          — handle straggler tasks

Usage:
    python spark_tune.py
"""

import os
import tempfile
import time
from pyspark.sql              import SparkSession
from pyspark.sql.functions    import (
    col, rand, broadcast, expr, lit, concat,
    floor, count, sum as _sum, avg, when
)
from pyspark.storagelevel     import StorageLevel


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def elapsed(start: float) -> str:
    return f"{(time.time() - start) * 1000:.1f} ms"


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPARKSESSION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

section("1. SPARKSESSION CONFIGURATION")

spark = (
    SparkSession.builder
    .appName("SparkPerformanceTuning")

    # ── Memory ───────────────────────────────────────────────────────────────
    .config("spark.driver.memory",            "2g")    # driver heap
    .config("spark.executor.memory",          "2g")    # executor heap
    .config("spark.memory.fraction",          "0.8")   # fraction for execution+storage
    .config("spark.memory.storageFraction",   "0.3")   # of above, reserved for storage

    # ── Parallelism & Shuffle ─────────────────────────────────────────────────
    .config("spark.sql.shuffle.partitions",   "8")     # default 200 → tune for local
    .config("spark.default.parallelism",      "8")

    # ── Adaptive Query Execution ──────────────────────────────────────────────
    .config("spark.sql.adaptive.enabled",                        "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled",     "true")
    .config("spark.sql.adaptive.skewJoin.enabled",               "true")

    # ── Join strategy ─────────────────────────────────────────────────────────
    .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))  # 10 MB

    # ── Speculative execution ─────────────────────────────────────────────────
    .config("spark.speculation",              "true")
    .config("spark.speculation.multiplier",   "1.5")
    .config("spark.speculation.quantile",     "0.9")

    # ── Warehouse dir (static — must be set at creation) ──────────────────────
    .config("spark.sql.warehouse.dir",
            os.path.join(tempfile.gettempdir(), "spark_warehouse"))

    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n  Key configuration settings:")
configs = [
    "spark.driver.memory",
    "spark.executor.memory",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.adaptive.skewJoin.enabled",
    "spark.sql.autoBroadcastJoinThreshold",
    "spark.speculation",
]
for key in configs:
    val = spark.conf.get(key, "not set")
    print(f"    {key:<50} = {val}")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════════════

# Large fact table
orders = (
    spark.range(0, 100_000)
    .withColumn("customer_id", (col("id") % 500).cast("int"))
    .withColumn("product_id",  (col("id") % 200).cast("int"))
    .withColumn("amount",      (rand() * 1000).cast("double"))
    .withColumn("region",      when(col("id") % 4 == 0, "NORTH")
                               .when(col("id") % 4 == 1, "SOUTH")
                               .when(col("id") % 4 == 2, "EAST")
                               .otherwise("WEST"))
    .withColumnRenamed("id", "order_id")
)

# Small dimension table (good candidate for broadcast)
customers = (
    spark.range(0, 500)
    .withColumn("name",    concat(lit("Customer_"), col("id")))
    .withColumn("segment", when(col("id") % 3 == 0, "Gold")
                           .when(col("id") % 3 == 1, "Silver")
                           .otherwise("Bronze"))
    .withColumnRenamed("id", "customer_id")
)

products = (
    spark.range(0, 200)
    .withColumn("category", when(col("id") % 4 == 0, "Electronics")
                            .when(col("id") % 4 == 1, "Clothing")
                            .when(col("id") % 4 == 2, "Food")
                            .otherwise("Books"))
    .withColumn("price",    (rand() * 500 + 10).cast("double"))
    .withColumnRenamed("id", "product_id")
)

print(f"\n  Sample data created:")
print(f"    orders    : {orders.count():>8,} rows")
print(f"    customers : {customers.count():>8,} rows")
print(f"    products  : {products.count():>8,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARTITIONING
# ══════════════════════════════════════════════════════════════════════════════

section("2. PARTITIONING")

subsection("Default partition count")
print(f"    orders partitions (default)  : {orders.rdd.getNumPartitions()}")

# repartition — full shuffle, increases OR decreases, round-robin or by column
subsection("repartition() — full shuffle, use to increase or balance partitions")
repartitioned = orders.repartition(16)
print(f"    After repartition(16)        : {repartitioned.rdd.getNumPartitions()} partitions")

repartitioned_by_col = orders.repartition(8, "region")
print(f"    After repartition(8,'region'): {repartitioned_by_col.rdd.getNumPartitions()} partitions")
print(f"    (rows per region partition — same region rows go to same partition)")

# coalesce — no shuffle, only reduces partitions
subsection("coalesce() — no shuffle, use only to reduce partitions")
coalesced = orders.coalesce(4)
print(f"    After coalesce(4)            : {coalesced.rdd.getNumPartitions()} partitions")

subsection("Partition size guidance")
print("""
    Rule of thumb:
      • Target 128 MB – 200 MB per partition
      • Too few  → large tasks, OOM risk, low parallelism
      • Too many → task scheduling overhead, tiny tasks
      • Formula  : partitions = max(2 × cores, totalDataMB / 128)
      • After a shuffle, tune with spark.sql.shuffle.partitions
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CACHING & PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

section("3. CACHING & PERSISTENCE")

subsection("cache() — MEMORY_AND_DISK (default)")
orders_cached = orders.cache()
t = time.time()
orders_cached.count()     # triggers materialization
print(f"    First count (cold cache)  : {elapsed(t)}")

t = time.time()
orders_cached.count()     # served from cache
print(f"    Second count (warm cache) : {elapsed(t)}")

subsection("persist() — explicit StorageLevel")
# StorageLevel options:
#   MEMORY_ONLY          — fastest, fails if not enough RAM (spills nothing)
#   MEMORY_AND_DISK      — spills to disk when RAM full (default cache())
#   DISK_ONLY            — always on disk (slowest, lowest memory cost)
#   MEMORY_ONLY_SER      — serialized in RAM (smaller, CPU overhead)
#   MEMORY_AND_DISK_SER  — serialized, spills to disk
#   OFF_HEAP             — Tungsten off-heap memory

orders_mem_ser = orders.persist(StorageLevel.MEMORY_AND_DISK)
orders_mem_ser.count()    # materialize

print("""
    StorageLevel        RAM   Disk  Serialized  Best for
    ─────────────────   ───   ────  ──────────  ─────────────────────────────
    MEMORY_ONLY          ✓     ✗       ✗        Fastest; data fits in RAM
    MEMORY_AND_DISK      ✓     ✓       ✗        Default; safe fallback
    DISK_ONLY            ✗     ✓       ✓        Large data, low-memory env
    MEMORY_ONLY_SER      ✓     ✗       ✓        Smaller RAM, more CPU
    MEMORY_AND_DISK_SER  ✓     ✓       ✓        Balanced for large datasets
    OFF_HEAP             ✓     ✗       ✓        Avoids GC pressure
""")

subsection("When to cache — decision guide")
print("""
    Cache when a DataFrame is:
      ✔  Reused in multiple actions / iterations
      ✔  Result of an expensive transformation chain
      ✔  Used in iterative ML algorithms

    Do NOT cache when:
      ✘  Used only once (wastes memory)
      ✘  Very large (larger than available cluster RAM)
      ✘  Source is already fast (in-memory table, small file)
""")

subsection("unpersist() — release cache")
orders_cached.unpersist()
orders_mem_ser.unpersist()
print("    orders_cached and orders_mem_ser unpersisted.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BROADCAST JOINS
# ══════════════════════════════════════════════════════════════════════════════

section("4. BROADCAST JOINS")

subsection("Regular shuffle join (no broadcast hint)")
t = time.time()
regular_join = orders.join(customers, "customer_id").count()
print(f"    Regular join result : {regular_join:,} rows  |  {elapsed(t)}")

subsection("Broadcast join — small table replicated to each executor")
t = time.time()
broadcast_join = orders.join(broadcast(customers), "customer_id").count()
print(f"    Broadcast join result: {broadcast_join:,} rows  |  {elapsed(t)}")

print("""
    How it works:
      • Small table is serialized and sent to EVERY executor
      • Each executor joins locally — NO shuffle of the large table
      • Threshold: spark.sql.autoBroadcastJoinThreshold (default 10 MB)
      • Force with: broadcast(df) hint even above threshold

    When to use:
      ✔  One side is < 10–50 MB (dimension/lookup tables)
      ✔  Star-schema fact ↔ dimension joins
      ✘  Both sides are large → sort-merge join is better
""")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ADAPTIVE QUERY EXECUTION (AQE)
# ══════════════════════════════════════════════════════════════════════════════

section("5. ADAPTIVE QUERY EXECUTION (AQE)")

print("""
    AQE (Spark 3.0+) re-optimizes the query plan at runtime using
    actual statistics collected after each shuffle stage.

    Three main features:

    ┌─────────────────────────────────┬──────────────────────────────────────┐
    │ Feature                         │ What it does                         │
    ├─────────────────────────────────┼──────────────────────────────────────┤
    │ Coalesce shuffle partitions     │ Merges small post-shuffle partitions │
    │                                 │ to avoid thousands of tiny tasks     │
    ├─────────────────────────────────┼──────────────────────────────────────┤
    │ Convert sort-merge → broadcast  │ If a join side turns out small after │
    │                                 │ filtering, switches to broadcast join│
    ├─────────────────────────────────┼──────────────────────────────────────┤
    │ Skew join optimization          │ Splits skewed partitions into smaller│
    │                                 │ tasks and replicates the other side  │
    └─────────────────────────────────┴──────────────────────────────────────┘

    Configuration:
      spark.sql.adaptive.enabled                    = true   (master switch)
      spark.sql.adaptive.coalescePartitions.enabled = true
      spark.sql.adaptive.skewJoin.enabled           = true
      spark.sql.adaptive.skewJoin.skewedPartitionFactor       = 5
      spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes = 256MB
""")

print(f"    AQE enabled : {spark.conf.get('spark.sql.adaptive.enabled')}")
print(f"    Coalesce    : {spark.conf.get('spark.sql.adaptive.coalescePartitions.enabled')}")
print(f"    Skew join   : {spark.conf.get('spark.sql.adaptive.skewJoin.enabled')}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICATE PUSHDOWN
# ══════════════════════════════════════════════════════════════════════════════

section("6. PREDICATE PUSHDOWN")

subsection("Without pushdown — filter AFTER full scan")
t = time.time()
no_pushdown = orders.filter(col("region") == "NORTH").count()
print(f"    Rows in NORTH: {no_pushdown:,}  |  {elapsed(t)}")

subsection("Predicate pushdown with Parquet — filter pushed into file reader")
parquet_path = os.path.join(tempfile.gettempdir(), "orders_parquet")
orders.write.mode("overwrite").parquet(parquet_path)

t = time.time()
orders_parquet = spark.read.parquet(parquet_path)
north_orders   = orders_parquet.filter(col("region") == "NORTH").count()
print(f"    Rows in NORTH (from Parquet): {north_orders:,}  |  {elapsed(t)}")

print("""
    How predicate pushdown works:
      • Spark pushes WHERE/filter conditions down to the data source
      • Parquet / ORC: only relevant row groups are read (skip entire blocks)
      • JDBC sources: WHERE clause sent to the database engine
      • Reduces I/O significantly for selective queries

    Verify it works:
      df.filter(...).explain(True)
      Look for "PushedFilters" in the Physical Plan output
""")


# ══════════════════════════════════════════════════════════════════════════════
# 7. COLUMN PRUNING
# ══════════════════════════════════════════════════════════════════════════════

section("7. COLUMN PRUNING")

subsection("Bad — select * reads all columns")
all_cols = orders_parquet.select("*")
print(f"    Columns read (select *)        : {len(all_cols.columns)}")

subsection("Good — select only needed columns (column pruning)")
pruned = orders_parquet.select("order_id", "customer_id", "amount")
print(f"    Columns read (pruned)          : {len(pruned.columns)}")

print("""
    Benefits:
      • Parquet/ORC are columnar — unneeded columns are never read from disk
      • Reduces deserialization cost and memory pressure
      • Catalyst optimizer applies this automatically when possible

    Best practices:
      ✔  Explicitly select only required columns
      ✔  Avoid df.select("*") in production pipelines
      ✔  Drop intermediate columns early in the transformation chain
""")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BUCKETING
# ══════════════════════════════════════════════════════════════════════════════

section("8. BUCKETING")

print("""
    Bucketing pre-organizes data into a fixed number of files ("buckets")
    based on a hash of a column. When two bucketed tables are joined on the
    same bucket column, Spark skips the shuffle entirely.

    Write bucketed table:
      df.write
        .bucketBy(8, "customer_id")   ← number of buckets, join key
        .sortBy("customer_id")        ← optional, enables sort-merge join
        .saveAsTable("orders_bucketed")

    Read and join (no shuffle):
      spark.table("orders_bucketed")
           .join(spark.table("customers_bucketed"), "customer_id")

    Guidelines:
      • Use for large tables joined repeatedly on the same key
      • Number of buckets should be a multiple of the number of executors
      • Bucket column should have high cardinality (customer_id, order_id)
      • Works with Hive/Spark metastore (saveAsTable, not write.parquet)
      • bucketBy + sortBy enables sort-merge join without shuffle
""")

import shutil
spark.sql("DROP TABLE IF EXISTS orders_bucketed")
bucketed_loc = os.path.join(tempfile.gettempdir(), "spark_warehouse", "orders_bucketed")
if os.path.exists(bucketed_loc):
    shutil.rmtree(bucketed_loc)
(
    orders.write
    .mode("overwrite")
    .bucketBy(8, "customer_id")
    .sortBy("customer_id")
    .saveAsTable("orders_bucketed")
)
print("    orders_bucketed table written (8 buckets on customer_id).")

bucketed_df = spark.table("orders_bucketed")
print(f"    Read back: {bucketed_df.count():,} rows from bucketed table")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SKEW HANDLING — SALTING TECHNIQUE
# ══════════════════════════════════════════════════════════════════════════════

section("9. SKEW HANDLING — SALTING TECHNIQUE")

subsection("Simulate skewed data")
# 80% of orders go to customer_id = 0 (hot key)
skewed_orders = (
    orders
    .withColumn(
        "customer_id",
        when(rand() < 0.8, lit(0)).otherwise(col("customer_id"))
    )
)

key_counts = (
    skewed_orders
    .groupBy("customer_id")
    .count()
    .orderBy(col("count").desc())
)
print("    Top 5 customer_id frequencies (skewed distribution):")
key_counts.show(5, truncate=False)

subsection("Salting — distribute skewed key across multiple partitions")

SALT_FACTOR = 8

# Salt the large (skewed) table: append random salt 0..7 to the join key
salted_orders = (
    skewed_orders
    .withColumn("salt",        (rand() * SALT_FACTOR).cast("int"))
    .withColumn("customer_id_salted",
                concat(col("customer_id").cast("string"), lit("_"), col("salt")))
)

# Explode the small (dimension) table: replicate each row SALT_FACTOR times
from pyspark.sql.functions import explode, array

salted_customers = (
    customers
    .withColumn("salt_array", array([lit(i) for i in range(SALT_FACTOR)]))
    .withColumn("salt",       explode(col("salt_array")))
    .withColumn("customer_id_salted",
                concat(col("customer_id").cast("string"), lit("_"), col("salt")))
    .drop("salt_array", "salt")
)

t = time.time()
salted_result = salted_orders.join(salted_customers, "customer_id_salted").count()
print(f"    Salted join rows  : {salted_result:,}  |  {elapsed(t)}")

print("""
    How salting works:
      1. Append random integer (0 to N-1) to each large-table join key
      2. Replicate small table N times, appending each integer
      3. Join on the salted key — hot partition splits across N partitions
      4. No single executor gets the entire hot key anymore

    Spark 3 AQE skewJoin does this automatically when enabled.
    Manual salting is useful for Spark 2.x or when AQE isn't sufficient.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SHUFFLE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

section("10. SHUFFLE OPTIMIZATION")

print("""
    Shuffles are the most expensive operations in Spark — they move data
    across the network. Minimize and optimize them.

    ┌──────────────────────────────┬────────────────────────────────────────┐
    │ Technique                    │ Description                            │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Reduce shuffle.partitions    │ Default 200 is too high for small data │
    │                              │ Tune to: 2–3× number of executor cores │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Use reduceByKey over         │ reduceByKey pre-aggregates on the map  │
    │ groupByKey (RDD API)         │ side, sending less data over the wire  │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Avoid wide transformations   │ groupBy, join, distinct, repartition   │
    │ when narrow ones suffice     │ all trigger shuffles                   │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Sort-merge join              │ Default for large-large joins; both    │
    │                              │ sides sorted & merged without full OOM │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Hash join                    │ Faster than sort-merge for medium      │
    │                              │ tables; disabled by default            │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Enable shuffle compression   │ spark.shuffle.compress = true          │
    │                              │ spark.shuffle.spill.compress = true    │
    └──────────────────────────────┴────────────────────────────────────────┘

    Shuffle tuning configs:
      spark.sql.shuffle.partitions               = 200  (tune down for small data)
      spark.shuffle.compress                     = true
      spark.shuffle.spill.compress               = true
      spark.reducer.maxSizeInFlight              = 48m
      spark.shuffle.file.buffer                  = 32k
      spark.shuffle.sort.bypassMergeThreshold    = 200
""")

subsection("Aggregation without shuffle (mapSide combine)")
t = time.time()
region_totals = orders.groupBy("region").agg(_sum("amount").alias("total"))
region_totals.show(truncate=False)
print(f"    groupBy+agg time: {elapsed(t)}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. EXECUTION PLAN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

section("11. EXECUTION PLAN ANALYSIS")

query = (
    orders
    .filter(col("region") == "NORTH")
    .join(broadcast(customers), "customer_id")
    .groupBy("segment")
    .agg(_sum("amount").alias("total_revenue"), count("*").alias("order_count"))
    .orderBy(col("total_revenue").desc())
)

subsection("Logical Plan (parsed → analyzed → optimized)")
query.explain(mode="simple")

subsection("Extended Plan — all 4 stages")
query.explain(mode="extended")

subsection("Cost-based Plan")
query.explain(mode="cost")

subsection("Formatted Plan (most readable)")
query.explain(mode="formatted")

print("""
    Plan reading guide:
      ✦ Read bottom-up — execution starts at the bottom
      ✦ FileScan        — data source; check PushedFilters for predicate pushdown
      ✦ Filter          — applied after scan (if not pushed down)
      ✦ BroadcastHashJoin — confirms broadcast join was chosen
      ✦ HashAggregate   — local (partial) + shuffle + global aggregation
      ✦ Exchange        — shuffle boundary; minimize these
      ✦ Sort            — sort within partitions
      ✦ TakeOrderedAndProject — ORDER BY + LIMIT combined
""")

subsection("Query result")
query.show(truncate=False)


# ══════════════════════════════════════════════════════════════════════════════
# 12. SPECULATIVE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

section("12. SPECULATIVE EXECUTION")

print("""
    Speculative execution launches duplicate copies of slow (straggler) tasks
    on other executors. Whichever finishes first is used; the other is killed.

    Configuration:
      spark.speculation               = true   (enable)
      spark.speculation.interval      = 100ms  (how often to check for stragglers)
      spark.speculation.multiplier    = 1.5    (task is straggler if 1.5× median)
      spark.speculation.quantile      = 0.75   (75%% of tasks must finish first)
      spark.speculation.minTaskRuntime= 100ms  (ignore very short tasks)

    When to enable:
      ✔  Heterogeneous cluster (some nodes slower than others)
      ✔  Occasional slow nodes due to I/O or GC pauses
      ✔  Long-running batch jobs where tail latency matters

    When to disable:
      ✘  Non-idempotent writes (duplicate speculative task = duplicate write)
      ✘  Streaming jobs with exactly-once semantics
      ✘  Cost-sensitive environments (duplicate tasks = extra resources)

    Current setting: spark.speculation = %s
""" % spark.conf.get("spark.speculation"))


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TUNING QUICK REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

section("PERFORMANCE TUNING QUICK REFERENCE")

print("""
  ┌─────────────────────────────────────────────────────────────┐
  │          SPARK PERFORMANCE TUNING CHEAT SHEET               │
  ├────────────────────────┬────────────────────────────────────┤
  │ Problem                │ Solution                           │
  ├────────────────────────┼────────────────────────────────────┤
  │ Too many small tasks   │ coalesce() or increase partition   │
  │                        │ size; tune shuffle.partitions      │
  ├────────────────────────┼────────────────────────────────────┤
  │ OOM on executors       │ Increase executor.memory;          │
  │                        │ persist(MEMORY_AND_DISK_SER)       │
  ├────────────────────────┼────────────────────────────────────┤
  │ Slow joins             │ Broadcast small tables;            │
  │                        │ bucket repeated join keys          │
  ├────────────────────────┼────────────────────────────────────┤
  │ Skewed partitions      │ Enable AQE skewJoin or use salting │
  ├────────────────────────┼────────────────────────────────────┤
  │ Slow repeated queries  │ cache() / persist() the DataFrame  │
  ├────────────────────────┼────────────────────────────────────┤
  │ Reading too much data  │ Filter early (predicate pushdown); │
  │                        │ select only needed columns         │
  ├────────────────────────┼────────────────────────────────────┤
  │ Excessive shuffles     │ Reduce wide transformations;       │
  │                        │ use reduceByKey not groupByKey     │
  ├────────────────────────┼────────────────────────────────────┤
  │ Straggler tasks        │ Enable speculative execution       │
  ├────────────────────────┼────────────────────────────────────┤
  │ Unknown bottleneck     │ Read the execution plan:           │
  │                        │ df.explain(mode="formatted")       │
  └────────────────────────┴────────────────────────────────────┘

  Key configs summary:
    spark.sql.shuffle.partitions                     = 8–200
    spark.sql.adaptive.enabled                       = true
    spark.sql.autoBroadcastJoinThreshold             = 10485760 (10 MB)
    spark.memory.fraction                            = 0.8
    spark.speculation                                = true
    spark.shuffle.compress                           = true
""")

# ── Cleanup ───────────────────────────────────────────────────────────────────
spark.stop()
print("  SparkSession stopped.")
print("═" * 62)
