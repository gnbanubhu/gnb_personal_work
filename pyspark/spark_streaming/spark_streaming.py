"""
spark_streaming.py
------------------
Demonstrates Spark Structured Streaming using a rate source
to simulate a live data stream with windowed aggregations.

Usage:
    python spark_streaming.py
    (Runs for 15 seconds then stops automatically)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, current_timestamp

spark = SparkSession.builder \
    .appName("SparkStreamingDemo") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Simulated Stream Source (rate = 1 row/sec) ────────────────────────────────
stream_df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 5) \
    .load()

# ── Transformations ───────────────────────────────────────────────────────────
processed = stream_df.select(
    col("timestamp"),
    col("value"),
    (col("value") % 3).alias("category")   # simulate 3 categories: 0, 1, 2
)

# ── Windowed Aggregation (10-sec window, sliding every 5 sec) ─────────────────
windowed = processed \
    .withWatermark("timestamp", "10 seconds") \
    .groupBy(window("timestamp", "10 seconds", "5 seconds"), "category") \
    .count()

# ── Output to Console ─────────────────────────────────────────────────────────
print("=" * 55)
print("SPARK STRUCTURED STREAMING — RATE SOURCE")
print("Streaming for 15 seconds...")
print("=" * 55)

query = windowed.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .option("checkpointLocation", "/tmp/spark_streaming_checkpoint") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination(timeout=15)
query.stop()

print("\nStreaming query stopped.")
spark.stop()
