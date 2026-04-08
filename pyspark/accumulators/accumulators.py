"""
accumulators/sample.py
----------------------
Demonstrates Spark Accumulators — shared variables that are only
added to through an associative and commutative operation, used
for counters and sums across distributed tasks.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AccumulatorSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Create Accumulators ───────────────────────────────────────────────────────
total_records   = sc.accumulator(0)
error_count     = sc.accumulator(0)
valid_count     = sc.accumulator(0)

# ── Sample Data ───────────────────────────────────────────────────────────────
records = sc.parallelize([
    ("Alice",   95000, "valid"),
    ("Bob",     -1000, "error"),   # invalid salary
    ("Charlie", 88000, "valid"),
    ("Diana",   None,  "error"),   # missing salary
    ("Eve",     78000, "valid"),
    ("Frank",   102000,"valid"),
    ("Grace",   -500,  "error"),   # invalid salary
])

# ── Process with Accumulators ─────────────────────────────────────────────────
def process_record(record):
    name, salary, status = record
    total_records.add(1)
    if status == "error":
        error_count.add(1)
    else:
        valid_count.add(1)
    return record

processed = records.map(process_record)

# ── Trigger action to execute the DAG ────────────────────────────────────────
valid_records = processed.filter(lambda r: r[2] == "valid")

print("=" * 50)
print("VALID RECORDS")
print("=" * 50)
for name, salary, status in valid_records.collect():
    print(f"  {name:<10} : ${salary:,}")

# ── Read Accumulator values (only accurate after action) ──────────────────────
print("\n" + "=" * 50)
print("ACCUMULATOR RESULTS")
print("=" * 50)
print(f"  Total records processed : {total_records.value}")
print(f"  Valid records           : {valid_count.value}")
print(f"  Error records           : {error_count.value}")
print(f"  Error rate              : {error_count.value / total_records.value * 100:.1f}%")

spark.stop()
