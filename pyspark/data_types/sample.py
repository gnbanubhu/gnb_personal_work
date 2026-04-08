"""
data_types/sample.py
---------------------
Demonstrates PySpark data types including primitive types,
complex types (ArrayType, MapType, StructType), and schema
definitions using StructField.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, ArrayType, MapType, LongType
)
from pyspark.sql.functions import col, explode, map_keys, map_values

spark = SparkSession.builder \
    .appName("DataTypesSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Primitive Types ────────────────────────────────────────────────────────────
print("=" * 55)
print("PRIMITIVE TYPES — Explicit Schema")
print("=" * 55)

schema = StructType([
    StructField("id",        IntegerType(), nullable=False),
    StructField("name",      StringType(),  nullable=False),
    StructField("salary",    DoubleType(),  nullable=True),
    StructField("age",       IntegerType(), nullable=True),
    StructField("is_active", BooleanType(), nullable=True),
])

data = [
    (1, "Alice",   95000.0, 30, True),
    (2, "Bob",     72000.5, 25, True),
    (3, "Charlie", 88000.0, 35, False),
]

df = spark.createDataFrame(data, schema)
df.printSchema()
df.show()

# ── ArrayType ────────────────────────────────────────────────────────────────
print("=" * 55)
print("ARRAY TYPE")
print("=" * 55)

array_schema = StructType([
    StructField("name",   StringType(),               nullable=False),
    StructField("skills", ArrayType(StringType()),     nullable=True),
])

array_data = [
    ("Alice",   ["Python", "Spark", "SQL"]),
    ("Bob",     ["Java",   "Scala"]),
    ("Charlie", ["Python", "R",     "ML"]),
]

array_df = spark.createDataFrame(array_data, array_schema)
array_df.printSchema()
array_df.show(truncate=False)

# explode array into separate rows
print("  After explode():")
array_df.select("name", explode(col("skills")).alias("skill")).show()

# ── MapType ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("MAP TYPE")
print("=" * 55)

map_schema = StructType([
    StructField("name",    StringType(),                        nullable=False),
    StructField("scores",  MapType(StringType(), IntegerType()), nullable=True),
])

map_data = [
    ("Alice",   {"math": 95, "science": 88, "english": 92}),
    ("Bob",     {"math": 78, "science": 85}),
    ("Charlie", {"math": 91, "science": 79, "english": 84}),
]

map_df = spark.createDataFrame(map_data, map_schema)
map_df.printSchema()
map_df.show(truncate=False)

print("  Map keys:")
map_df.select("name", map_keys("scores").alias("subjects")).show(truncate=False)

# ── StructType (Nested) ───────────────────────────────────────────────────────
print("=" * 55)
print("NESTED STRUCT TYPE")
print("=" * 55)

nested_schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("address", StructType([
        StructField("city",    StringType(),  nullable=True),
        StructField("state",   StringType(),  nullable=True),
        StructField("zipcode", IntegerType(), nullable=True),
    ]), nullable=True),
])

nested_data = [
    ("Alice",   ("San Francisco", "CA", 94102)),
    ("Bob",     ("New York",      "NY", 10001)),
    ("Charlie", ("Chicago",       "IL", 60601)),
]

nested_df = spark.createDataFrame(nested_data, nested_schema)
nested_df.printSchema()
nested_df.show(truncate=False)

# Access nested fields using dot notation
print("  Accessing nested fields:")
nested_df.select("name", "address.city", "address.state").show()

spark.stop()
