"""
data_sources/parquet.py
------------------------
Demonstrates reading and writing Parquet files in PySpark including
columnar storage, compression, partitioning, predicate pushdown,
and schema evolution.

Usage:
    python parquet.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("ParquetDataSource") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

output_dir = "/tmp/pyspark_parquet"
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

# ── Write Parquet ─────────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE PARQUET — Default (snappy compression)")
print("=" * 55)

parquet_path = f"{output_dir}/employees"
df.write.mode("overwrite").parquet(parquet_path)
print(f"  Written to  : {parquet_path}")
print(f"  Format      : columnar (stores column-by-column)")
print(f"  Compression : snappy (default)")

# ── Read Parquet ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("READ PARQUET")
print("=" * 55)

parquet_df = spark.read.parquet(parquet_path)
print("  Schema (preserved from write — no inference needed):")
parquet_df.printSchema()
parquet_df.show()

# ── Compression Codecs ────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE PARQUET — Different Compression Codecs")
print("=" * 55)

for codec in ["snappy", "gzip", "lz4", "none"]:
    path = f"{output_dir}/employees_{codec}"
    df.write.mode("overwrite").option("compression", codec).parquet(path)
    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(path)
        for f in fn if f.endswith(".parquet")
    )
    print(f"  {codec:<8} → {size:>8,} bytes  saved to {path}")

# ── Partitioned Parquet ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PARTITIONED PARQUET — by department and year")
print("=" * 55)

partitioned_path = f"{output_dir}/employees_partitioned"
df.write.mode("overwrite") \
    .partitionBy("department", "year") \
    .parquet(partitioned_path)
print(f"  Written to  : {partitioned_path}")
print(f"  Partition layout:")
print(f"    department=Engineering/year=2022/")
print(f"    department=Engineering/year=2023/")
print(f"    department=HR/year=2022/")
print(f"    department=HR/year=2023/")
print(f"    department=Marketing/year=2022/")
print(f"    department=Marketing/year=2023/")

# ── Predicate Pushdown (partition pruning) ────────────────────────────────────
print("\n" + "=" * 55)
print("PREDICATE PUSHDOWN — Partition Pruning")
print("=" * 55)

pruned_df = spark.read.parquet(partitioned_path) \
    .filter(col("department") == "Engineering") \
    .filter(col("year") == 2023)

print("  Filter: department=Engineering AND year=2023")
print("  Spark reads ONLY the matching partition folder — skips all others.")
pruned_df.show()

# ── Column Projection (read only needed columns) ──────────────────────────────
print("=" * 55)
print("COLUMN PROJECTION — Read only selected columns")
print("=" * 55)

projected = spark.read.parquet(parquet_path).select("name", "salary")
print("  Selected: name, salary only (other columns never read from disk)")
projected.show()

# ── Append Mode ───────────────────────────────────────────────────────────────
print("=" * 55)
print("APPEND MODE — Add new records to existing Parquet")
print("=" * 55)

new_data = [(8, "Hank", "Finance", "Austin", 85000.0, 2023)]
new_schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("name",       StringType(),  False),
    StructField("department", StringType(),  True),
    StructField("city",       StringType(),  True),
    StructField("salary",     DoubleType(),  True),
    StructField("year",       IntegerType(), True),
])
new_df = spark.createDataFrame(new_data, new_schema)
new_df.write.mode("append").parquet(parquet_path)

total = spark.read.parquet(parquet_path).count()
print(f"  Original rows : 7")
print(f"  Appended rows : 1")
print(f"  Total rows    : {total}")

spark.stop()
