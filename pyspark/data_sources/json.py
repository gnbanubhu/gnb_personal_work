"""
data_sources/json.py
---------------------
Demonstrates reading and writing JSON data in PySpark including
single-line JSON, multi-line JSON, nested JSON, and schema
inference vs explicit schema.

Usage:
    python json.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, ArrayType
)
from pyspark.sql.functions import col, explode

spark = SparkSession.builder \
    .appName("JSONDataSource") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

output_dir = "/tmp/pyspark_json"
os.makedirs(output_dir, exist_ok=True)

# ── Sample Data ───────────────────────────────────────────────────────────────
data = [
    (1, "Alice",   "Engineering", 95000.0),
    (2, "Bob",     "Marketing",   72000.0),
    (3, "Charlie", "Engineering", 88000.0),
    (4, "Diana",   "HR",          65000.0),
    (5, "Eve",     "Marketing",   78000.0),
]
schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("name",       StringType(),  False),
    StructField("department", StringType(),  True),
    StructField("salary",     DoubleType(),  True),
])
df = spark.createDataFrame(data, schema)

# ── Write JSON ────────────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE JSON")
print("=" * 55)

json_path = f"{output_dir}/employees"
df.coalesce(1).write.mode("overwrite").json(json_path)
print(f"  Written to : {json_path}")
print(f"  Format     : newline-delimited JSON (one JSON object per line)")

# ── Read JSON — Schema Inference ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("READ JSON — Schema Inference")
print("=" * 55)

inferred_df = spark.read.json(json_path)
print("  Inferred Schema:")
inferred_df.printSchema()
inferred_df.show()

# ── Read JSON — Explicit Schema ───────────────────────────────────────────────
print("=" * 55)
print("READ JSON — Explicit Schema (faster, no scan needed)")
print("=" * 55)

explicit_df = spark.read.schema(schema).json(json_path)
print("  Explicit Schema:")
explicit_df.printSchema()
explicit_df.show()

# ── Write Multi-line JSON ─────────────────────────────────────────────────────
print("=" * 55)
print("WRITE & READ — Multi-line JSON")
print("=" * 55)

multiline_path = f"{output_dir}/employees_multiline"
df.coalesce(1).write.mode("overwrite") \
    .option("multiline", True) \
    .json(multiline_path)
print(f"  Written to : {multiline_path}")

multiline_df = spark.read \
    .option("multiline", True) \
    .json(multiline_path)
multiline_df.show()

# ── Read Nested JSON ──────────────────────────────────────────────────────────
print("=" * 55)
print("NESTED JSON — Struct and Array fields")
print("=" * 55)

nested_schema = StructType([
    StructField("name",       StringType(), False),
    StructField("department", StringType(), True),
    StructField("skills",     ArrayType(StringType()), True),
    StructField("address", StructType([
        StructField("city",  StringType(), True),
        StructField("state", StringType(), True),
    ]), True),
])

nested_data = [
    ("Alice",   "Engineering", ["Python", "Spark", "SQL"], ("San Francisco", "CA")),
    ("Bob",     "Marketing",   ["Excel",  "PowerBI"],      ("New York",       "NY")),
    ("Charlie", "Engineering", ["Java",   "Scala",  "ML"], ("Chicago",        "IL")),
]

nested_df = spark.createDataFrame(nested_data, nested_schema)

nested_json_path = f"{output_dir}/employees_nested"
nested_df.coalesce(1).write.mode("overwrite").json(nested_json_path)

read_nested = spark.read.schema(nested_schema).json(nested_json_path)
print("  Nested JSON Schema:")
read_nested.printSchema()
read_nested.show(truncate=False)

print("  Accessing nested fields:")
read_nested.select(
    col("name"),
    col("address.city").alias("city"),
    col("address.state").alias("state"),
).show()

print("  Exploding array field (skills):")
read_nested.select(col("name"), explode(col("skills")).alias("skill")).show()

# ── Read JSON with Options ────────────────────────────────────────────────────
print("=" * 55)
print("READ JSON — Common Options")
print("=" * 55)

print("  Options used:")
print("    allowComments       = true  → allows // comments in JSON")
print("    allowUnquotedFieldNames = true → lenient field names")
print("    allowSingleQuotes   = true  → single-quoted strings")
print("    allowNumericLeadingZeros = true → e.g. 007")
print("    mode = DROPMALFORMED → drops corrupt records silently")
print("    mode = PERMISSIVE    → nulls for corrupt fields (default)")
print("    mode = FAILFAST      → throws error on corrupt record")

spark.stop()
