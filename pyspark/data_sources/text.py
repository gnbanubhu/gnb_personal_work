"""
data_sources/text.py
---------------------
Demonstrates reading and writing plain text files in PySpark
including single-line text, multi-line text, parsing structured
data from raw text, and writing text output.

Usage:
    python text.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, trim, regexp_extract, regexp_replace,
    length, upper, lower, when, lit
)

spark = SparkSession.builder \
    .appName("TextDataSource") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

output_dir = "/tmp/pyspark_text"
os.makedirs(output_dir, exist_ok=True)

# ── Write Sample Text Files ───────────────────────────────────────────────────
raw_csv_text = "\n".join([
    "Alice,Engineering,95000",
    "Bob,Marketing,72000",
    "Charlie,Engineering,88000",
    "Diana,HR,65000",
    "Eve,Marketing,78000",
    "Frank,Engineering,102000",
])

log_text = "\n".join([
    "2024-01-15 10:23:45 INFO  User Alice logged in",
    "2024-01-15 10:24:10 ERROR Database connection failed",
    "2024-01-15 10:25:00 INFO  Query executed successfully",
    "2024-01-15 10:26:30 WARN  High memory usage detected",
    "2024-01-15 10:27:15 ERROR Timeout on request /api/data",
    "2024-01-15 10:28:00 INFO  User Bob logged in",
    "2024-01-15 10:29:45 ERROR Null pointer exception in module X",
])

with open(f"{output_dir}/employees.txt", "w") as f:
    f.write(raw_csv_text)

with open(f"{output_dir}/app.log", "w") as f:
    f.write(log_text)

print(f"Sample text files created at: {output_dir}")

# ── Read Text File ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("READ TEXT — One row per line (default)")
print("=" * 55)

text_df = spark.read.text(f"{output_dir}/employees.txt")
print("  Schema:")
text_df.printSchema()
text_df.show(truncate=False)

# ── Parse Structured Data from Raw Text ──────────────────────────────────────
print("=" * 55)
print("PARSE CSV-LIKE TEXT — Split by delimiter")
print("=" * 55)

parsed_df = text_df \
    .select(split(col("value"), ",").alias("fields")) \
    .select(
        trim(col("fields")[0]).alias("name"),
        trim(col("fields")[1]).alias("department"),
        trim(col("fields")[2]).cast("integer").alias("salary"),
    )

parsed_df.show()
parsed_df.printSchema()

# ── Filter and Transform Parsed Text ─────────────────────────────────────────
print("=" * 55)
print("FILTER — High earners from parsed text")
print("=" * 55)

parsed_df.filter(col("salary") > 80000) \
    .orderBy(col("salary").desc()) \
    .show()

# ── Read Log File ─────────────────────────────────────────────────────────────
print("=" * 55)
print("READ LOG FILE — Parse with regex")
print("=" * 55)

log_df = spark.read.text(f"{output_dir}/app.log")
print("  Raw log lines:")
log_df.show(truncate=False)

# ── Extract Log Fields using Regex ────────────────────────────────────────────
print("=" * 55)
print("REGEX EXTRACTION — Parse log fields")
print("=" * 55)

log_pattern = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+)\s+(.*)"

parsed_log = log_df.select(
    regexp_extract(col("value"), log_pattern, 1).alias("date"),
    regexp_extract(col("value"), log_pattern, 2).alias("time"),
    regexp_extract(col("value"), log_pattern, 3).alias("level"),
    regexp_extract(col("value"), log_pattern, 4).alias("message"),
)

parsed_log.show(truncate=False)

# ── Filter by Log Level ───────────────────────────────────────────────────────
print("=" * 55)
print("FILTER — ERROR log lines only")
print("=" * 55)

parsed_log.filter(col("level") == "ERROR").show(truncate=False)

# ── Log Level Summary ─────────────────────────────────────────────────────────
print("=" * 55)
print("LOG LEVEL SUMMARY — Count by level")
print("=" * 55)

parsed_log.groupBy("level").count() \
    .orderBy(col("count").desc()) \
    .show()

# ── Text Transformations ──────────────────────────────────────────────────────
print("=" * 55)
print("TEXT TRANSFORMATIONS")
print("=" * 55)

text_df.select(
    col("value"),
    length(col("value")).alias("line_length"),
    upper(col("value")).alias("uppercased"),
).show(truncate=False)

# ── Filter Lines containing a keyword ────────────────────────────────────────
print("=" * 55)
print("KEYWORD FILTER — Lines containing 'Engineering'")
print("=" * 55)

text_df.filter(col("value").contains("Engineering")).show(truncate=False)

# ── Write Text Output ─────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE TEXT — Save DataFrame as text")
print("=" * 55)

text_output_path = f"{output_dir}/output"
parsed_df.select(
    col("name").cast("string")
).coalesce(1).write.mode("overwrite").text(text_output_path)

print(f"  Written to : {text_output_path}")
print(f"  Format     : one value per line (only single string column allowed)")

output_check = spark.read.text(text_output_path)
print("  Written content:")
output_check.show(truncate=False)

spark.stop()
