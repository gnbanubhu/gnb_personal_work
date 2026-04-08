"""
data_sources/avro.py
---------------------
Demonstrates reading and writing Avro files in PySpark including
schema definition, compression, and partitioned writes.

Avro is a row-based format with schema embedded in the file,
commonly used in Kafka and data pipelines.

Usage:
    python avro.py

Requirement:
    spark-avro package (included with Spark 2.4+):
    SparkSession config: spark.jars.packages = org.apache.spark:spark-avro_2.12:3.5.6
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("AvroDataSource") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.6") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

output_dir = "/tmp/pyspark_avro"
os.makedirs(output_dir, exist_ok=True)

# ── Sample Data ───────────────────────────────────────────────────────────────
schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("name",       StringType(),  False),
    StructField("department", StringType(),  True),
    StructField("city",       StringType(),  True),
    StructField("salary",     DoubleType(),  True),
    StructField("year",       IntegerType(), True),
])

data = [
    (1, "Alice",   "Engineering", "NYC", 95000.0,  2022),
    (2, "Bob",     "Marketing",   "LA",  72000.0,  2022),
    (3, "Charlie", "Engineering", "NYC", 88000.0,  2023),
    (4, "Diana",   "HR",          "NYC", 65000.0,  2022),
    (5, "Eve",     "Marketing",   "SF",  78000.0,  2023),
    (6, "Frank",   "Engineering", "SF",  102000.0, 2023),
    (7, "Grace",   "HR",          "LA",  68000.0,  2023),
]
df = spark.createDataFrame(data, schema)

# ── Write Avro ────────────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE AVRO — Default (snappy compression)")
print("=" * 55)

avro_path = f"{output_dir}/employees"
df.write.mode("overwrite").format("avro").save(avro_path)
print(f"  Written to  : {avro_path}")
print(f"  Format      : row-based (schema embedded in file header)")
print(f"  Compression : snappy (default)")

# ── Read Avro ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("READ AVRO")
print("=" * 55)

avro_df = spark.read.format("avro").load(avro_path)
print("  Schema (read from embedded Avro schema):")
avro_df.printSchema()
avro_df.show()

# ── Compression Codecs ────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE AVRO — Compression Codecs")
print("=" * 55)

for codec in ["snappy", "deflate", "bzip2", "uncompressed"]:
    path = f"{output_dir}/employees_{codec}"
    df.write.mode("overwrite") \
        .option("compression", codec) \
        .format("avro") \
        .save(path)
    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(path)
        for f in fn if f.endswith(".avro")
    )
    print(f"  {codec:<15} → {size:>8,} bytes")

# ── Partitioned Avro ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PARTITIONED AVRO — by department")
print("=" * 55)

partitioned_path = f"{output_dir}/employees_partitioned"
df.write.mode("overwrite") \
    .partitionBy("department") \
    .format("avro") \
    .save(partitioned_path)
print(f"  Written to: {partitioned_path}")
print(f"  Layout:")
print(f"    department=Engineering/")
print(f"    department=HR/")
print(f"    department=Marketing/")

part_df = spark.read.format("avro").load(partitioned_path)
print("  Read back all partitions:")
part_df.orderBy("id").show()

# ── Filter on Partitioned Avro ────────────────────────────────────────────────
print("=" * 55)
print("FILTER — Partition Pruning on Avro")
print("=" * 55)

eng_df = spark.read.format("avro").load(partitioned_path) \
    .filter(col("department") == "Engineering")
print("  Filter: department=Engineering only")
eng_df.show()

# ── Column Projection ─────────────────────────────────────────────────────────
print("=" * 55)
print("COLUMN PROJECTION — Select specific columns")
print("=" * 55)

projected = spark.read.format("avro").load(avro_path) \
    .select("name", "department", "salary")
print("  Selected: name, department, salary")
projected.orderBy(col("salary").desc()).show()

# ── Avro vs Other Formats ─────────────────────────────────────────────────────
print("=" * 55)
print("AVRO vs PARQUET vs ORC — Key Differences")
print("=" * 55)
print(f"  {'Feature':<28} {'Avro':<18} {'Parquet':<18} {'ORC':<18}")
print(f"  {'-'*28} {'-'*18} {'-'*18} {'-'*18}")
print(f"  {'Storage layout':<28} {'Row-based':<18} {'Columnar':<18} {'Columnar':<18}")
print(f"  {'Schema location':<28} {'In file header':<18} {'In footer':<18} {'In file':<18}")
print(f"  {'Best for':<28} {'Write-heavy':<18} {'Read-heavy':<18} {'Hive/Hadoop':<18}")
print(f"  {'Kafka integration':<28} {'Native':<18} {'Limited':<18} {'Limited':<18}")
print(f"  {'Compression (default)':<28} {'snappy':<18} {'snappy':<18} {'zlib':<18}")
print(f"  {'Predicate pushdown':<28} {'Partition only':<18} {'Yes + stats':<18} {'Yes + index':<18}")
print(f"  {'Schema evolution':<28} {'Excellent':<18} {'Good':<18} {'Good':<18}")
print(f"  {'Splittable':<28} {'Yes':<18} {'Yes':<18} {'Yes':<18}")

spark.stop()
