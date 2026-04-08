"""
data_sources/sample.py
-----------------------
Demonstrates reading and writing data using various PySpark
data sources including CSV, JSON, and Parquet formats.

Usage:
    python sample.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

spark = SparkSession.builder \
    .appName("DataSourcesSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Output directory ──────────────────────────────────────────────────────────
output_dir = "/tmp/pyspark_data_sources"
os.makedirs(output_dir, exist_ok=True)

# ── Sample DataFrame ──────────────────────────────────────────────────────────
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

# ── CSV ───────────────────────────────────────────────────────────────────────
print("=" * 55)
print("CSV — Write & Read")
print("=" * 55)

csv_path = f"{output_dir}/employees.csv"
df.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_path)
print(f"  Written to : {csv_path}")

csv_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(csv_path)
print("  Read back:")
csv_df.show()
print(f"  Schema inferred:")
csv_df.printSchema()

# ── JSON ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("JSON — Write & Read")
print("=" * 55)

json_path = f"{output_dir}/employees.json"
df.coalesce(1).write.mode("overwrite").json(json_path)
print(f"  Written to : {json_path}")

json_df = spark.read.json(json_path)
print("  Read back:")
json_df.show()
json_df.printSchema()

# ── Parquet ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("PARQUET — Write & Read (columnar, compressed)")
print("=" * 55)

parquet_path = f"{output_dir}/employees.parquet"
df.write.mode("overwrite").parquet(parquet_path)
print(f"  Written to : {parquet_path}")

parquet_df = spark.read.parquet(parquet_path)
print("  Read back:")
parquet_df.show()
parquet_df.printSchema()

# ── Parquet with partition ─────────────────────────────────────────────────────
print("=" * 55)
print("PARQUET — Partitioned Write by Department")
print("=" * 55)

partitioned_path = f"{output_dir}/employees_partitioned"
df.write.mode("overwrite").partitionBy("department").parquet(partitioned_path)
print(f"  Written to : {partitioned_path}")
print(f"  Partitions : department=Engineering / department=HR / department=Marketing")

partitioned_df = spark.read.parquet(partitioned_path)
print("  Read back (all partitions):")
partitioned_df.orderBy("id").show()

# ── In-memory Temp View ───────────────────────────────────────────────────────
print("=" * 55)
print("IN-MEMORY TEMP VIEW — SQL on DataFrame")
print("=" * 55)

df.createOrReplaceTempView("employees")
result = spark.sql("SELECT department, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC")
result.show()

spark.stop()
