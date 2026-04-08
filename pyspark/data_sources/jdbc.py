"""
data_sources/jdbc.py
---------------------
Demonstrates reading and writing data via JDBC in PySpark.
Connects to a SQLite database (no server required) to show
JDBC read/write, predicate pushdown, and partitioned reads.

Usage:
    pip install pyspark
    python jdbc.py

Note:
    For production databases (PostgreSQL, MySQL, Oracle),
    replace the JDBC URL and driver class accordingly.
    The SQLite JDBC driver (sqlite-jdbc) must be on the classpath.
"""

import os
import sqlite3
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col

# ── Create SQLite DB with sample data ─────────────────────────────────────────
db_path = "/tmp/pyspark_jdbc/employees.db"
os.makedirs("/tmp/pyspark_jdbc", exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS employees")
cursor.execute("""
    CREATE TABLE employees (
        id         INTEGER PRIMARY KEY,
        name       TEXT,
        department TEXT,
        city       TEXT,
        salary     REAL,
        year       INTEGER
    )
""")
cursor.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", [
    (1, "Alice",   "Engineering", "NYC", 95000.0,  2022),
    (2, "Bob",     "Marketing",   "LA",  72000.0,  2022),
    (3, "Charlie", "Engineering", "NYC", 88000.0,  2023),
    (4, "Diana",   "HR",          "NYC", 65000.0,  2022),
    (5, "Eve",     "Marketing",   "SF",  78000.0,  2023),
    (6, "Frank",   "Engineering", "SF",  102000.0, 2023),
    (7, "Grace",   "HR",          "LA",  68000.0,  2023),
])
conn.commit()
conn.close()
print(f"SQLite database created at: {db_path}")

# ── SQLite JDBC driver path ───────────────────────────────────────────────────
# Download sqlite-jdbc jar from: https://github.com/xerial/sqlite-jdbc
sqlite_jar = os.path.expanduser("~/.ivy2/jars/org.xerial_sqlite-jdbc-3.43.0.0.jar")
jar_config  = f"--jars {sqlite_jar}" if os.path.exists(sqlite_jar) else ""

spark = SparkSession.builder \
    .appName("JDBCDataSource") \
    .master("local[*]") \
    .config("spark.driver.extraClassPath", sqlite_jar) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

jdbc_url    = f"jdbc:sqlite:{db_path}"
jdbc_driver = "org.sqlite.JDBC"

# ── JDBC Connection Properties ────────────────────────────────────────────────
properties = {
    "driver": jdbc_driver,
}

# ── Read via JDBC ─────────────────────────────────────────────────────────────
print("=" * 55)
print("READ via JDBC — Full Table Scan")
print("=" * 55)

try:
    jdbc_df = spark.read.jdbc(
        url=jdbc_url,
        table="employees",
        properties=properties,
    )
    jdbc_df.show()
    jdbc_df.printSchema()

    # ── Predicate Pushdown ────────────────────────────────────────────────────
    print("=" * 55)
    print("PREDICATE PUSHDOWN — Filter pushed to DB")
    print("=" * 55)
    filtered_df = spark.read.jdbc(
        url=jdbc_url,
        table="(SELECT * FROM employees WHERE salary > 80000) AS t",
        properties=properties,
    )
    print("  Query: salary > 80000 (executed inside the database)")
    filtered_df.show()

    # ── Partitioned Read ──────────────────────────────────────────────────────
    print("=" * 55)
    print("PARTITIONED READ — Parallel reads using id range")
    print("=" * 55)
    partitioned_df = spark.read.jdbc(
        url=jdbc_url,
        table="employees",
        column="id",
        lowerBound=1,
        upperBound=7,
        numPartitions=4,
        properties=properties,
    )
    print(f"  Partitions : {partitioned_df.rdd.getNumPartitions()}")
    print(f"  id range   : 1 to 7, split into 4 parallel reads")
    partitioned_df.orderBy("id").show()

    # ── Write via JDBC ────────────────────────────────────────────────────────
    print("=" * 55)
    print("WRITE via JDBC — New table")
    print("=" * 55)

    summary_df = jdbc_df.groupBy("department") \
        .agg({"salary": "avg", "id": "count"}) \
        .withColumnRenamed("avg(salary)", "avg_salary") \
        .withColumnRenamed("count(id)", "headcount")

    summary_df.write.jdbc(
        url=jdbc_url,
        table="dept_summary",
        mode="overwrite",
        properties=properties,
    )
    print("  Written dept_summary table to SQLite:")
    spark.read.jdbc(url=jdbc_url, table="dept_summary", properties=properties).show()

except Exception as e:
    print(f"\n  NOTE: SQLite JDBC driver not found.")
    print(f"  To run this demo, download sqlite-jdbc jar and place at:")
    print(f"  {sqlite_jar}")
    print(f"\n  For production JDBC examples:")
    print(f"  PostgreSQL: jdbc:postgresql://host:5432/dbname")
    print(f"  MySQL     : jdbc:mysql://host:3306/dbname")
    print(f"  Oracle    : jdbc:oracle:thin:@host:1521:SID")
    print(f"  SQL Server: jdbc:sqlserver://host:1433;databaseName=dbname")

finally:
    spark.stop()
