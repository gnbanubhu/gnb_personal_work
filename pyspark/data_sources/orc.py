"""
data_sources/orc.py
--------------------
Demonstrates reading and writing ORC (Optimized Row Columnar) files
in PySpark including compression, partitioning, predicate pushdown,
and comparison with Parquet.

Usage:
    python orc.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("ORCDataSource") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

output_dir = "/tmp/pyspark_orc"
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

# ── Write ORC ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE ORC — Default (zlib compression)")
print("=" * 55)

orc_path = f"{output_dir}/employees"
df.write.mode("overwrite").orc(orc_path)
print(f"  Written to  : {orc_path}")
print(f"  Format      : columnar with built-in indexes and bloom filters")
print(f"  Compression : zlib (default for ORC)")

# ── Read ORC ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("READ ORC")
print("=" * 55)

orc_df = spark.read.orc(orc_path)
print("  Schema:")
orc_df.printSchema()
orc_df.show()

# ── Compression Codecs ────────────────────────────────────────────────────────
print("=" * 55)
print("WRITE ORC — Compression Codecs")
print("=" * 55)

for codec in ["zlib", "snappy", "lzo", "none"]:
    path = f"{output_dir}/employees_{codec}"
    try:
        df.write.mode("overwrite").option("compression", codec).orc(path)
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(path)
            for f in fn if f.endswith(".orc")
        )
        print(f"  {codec:<8} → {size:>8,} bytes")
    except Exception as e:
        print(f"  {codec:<8} → not supported: {e}")

# ── Partitioned ORC ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PARTITIONED ORC — by department and year")
print("=" * 55)

partitioned_path = f"{output_dir}/employees_partitioned"
df.write.mode("overwrite") \
    .partitionBy("department", "year") \
    .orc(partitioned_path)
print(f"  Written to: {partitioned_path}")
print(f"  Layout:")
print(f"    department=Engineering/year=2022/")
print(f"    department=Engineering/year=2023/")
print(f"    department=HR/year=2022/")
print(f"    department=HR/year=2023/")
print(f"    department=Marketing/year=2022/")
print(f"    department=Marketing/year=2023/")

# ── Predicate Pushdown ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PREDICATE PUSHDOWN — Partition & Column Pruning")
print("=" * 55)

filtered = spark.read.orc(partitioned_path) \
    .filter(col("department") == "Engineering") \
    .filter(col("salary") > 90000)

print("  Filter: department=Engineering AND salary > 90000")
print("  Spark prunes non-Engineering partitions automatically.")
filtered.show()

# ── Column Projection ─────────────────────────────────────────────────────────
print("=" * 55)
print("COLUMN PROJECTION — Read only selected columns")
print("=" * 55)

projected = spark.read.orc(orc_path).select("name", "department", "salary")
print("  Selected: name, department, salary (other columns skipped on disk read)")
projected.orderBy(col("salary").desc()).show()

# ── ORC vs Parquet Comparison ─────────────────────────────────────────────────
print("=" * 55)
print("ORC vs PARQUET — Key Differences")
print("=" * 55)
print(f"  {'Feature':<30} {'ORC':<20} {'Parquet':<20}")
print(f"  {'-'*30} {'-'*20} {'-'*20}")
print(f"  {'Origin':<30} {'Hive / Hadoop':<20} {'Cloudera / Twitter':<20}")
print(f"  {'Default Compression':<30} {'zlib':<20} {'snappy':<20}")
print(f"  {'Built-in Indexes':<30} {'Yes (row-level)':<20} {'No':<20}")
print(f"  {'Bloom Filters':<30} {'Yes':<20} {'Yes':<20}")
print(f"  {'Nested Types':<30} {'Good':<20} {'Excellent':<20}")
print(f"  {'Hive Integration':<30} {'Native':<20} {'Supported':<20}")
print(f"  {'Spark Default':<30} {'No':<20} {'Yes':<20}")
print(f"  {'ACID Support':<30} {'Yes (Hive)':<20} {'No':<20}")

spark.stop()
