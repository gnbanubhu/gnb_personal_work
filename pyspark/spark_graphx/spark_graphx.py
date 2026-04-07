"""
spark_graphx.py
---------------
Demonstrates graph-like analysis using PySpark DataFrames
(GraphX is Scala-only; this uses DataFrame operations to simulate
graph computations like degree centrality and connected components).

Usage:
    python spark_graphx.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, union

spark = SparkSession.builder \
    .appName("SparkGraphXDemo") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Vertices (nodes) ──────────────────────────────────────────────────────────
vertices = spark.createDataFrame([
    (1, "Alice"),
    (2, "Bob"),
    (3, "Charlie"),
    (4, "Diana"),
    (5, "Eve"),
    (6, "Frank"),
], ["id", "name"])

# ── Edges (connections) ───────────────────────────────────────────────────────
edges = spark.createDataFrame([
    (1, 2, "friend"),
    (1, 3, "colleague"),
    (2, 4, "friend"),
    (3, 4, "colleague"),
    (4, 5, "friend"),
    (5, 6, "colleague"),
    (1, 5, "friend"),
], ["src", "dst", "relationship"])

print("=" * 50)
print("VERTICES (Nodes)")
print("=" * 50)
vertices.show()

print("=" * 50)
print("EDGES (Connections)")
print("=" * 50)
edges.show()

# ── Degree Centrality (number of connections per node) ────────────────────────
out_degree = edges.groupBy("src").agg(count("*").alias("out_degree")).withColumnRenamed("src", "id")
in_degree  = edges.groupBy("dst").agg(count("*").alias("in_degree")).withColumnRenamed("dst", "id")

degree = vertices \
    .join(out_degree, "id", "left") \
    .join(in_degree,  "id", "left") \
    .fillna(0, subset=["out_degree", "in_degree"])

degree = degree.withColumn("total_degree", col("out_degree") + col("in_degree"))

print("=" * 50)
print("DEGREE CENTRALITY")
print("=" * 50)
degree.select("name", "out_degree", "in_degree", "total_degree") \
    .orderBy(col("total_degree").desc()) \
    .show()

# ── Most influential node ─────────────────────────────────────────────────────
top = degree.orderBy(col("total_degree").desc()).first()
print(f"Most connected node: {top['name']} with {top['total_degree']} connections\n")

# ── Friendship relationships only ─────────────────────────────────────────────
print("=" * 50)
print("FRIEND CONNECTIONS")
print("=" * 50)
edges.filter(col("relationship") == "friend") \
    .join(vertices.withColumnRenamed("name", "src_name"), col("src") == col("id")) \
    .drop("id") \
    .join(vertices.withColumnRenamed("name", "dst_name"), col("dst") == col("id")) \
    .drop("id") \
    .select("src_name", "dst_name", "relationship") \
    .show()

spark.stop()
