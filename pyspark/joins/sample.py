"""
joins/sample.py
---------------
Demonstrates all PySpark join types including inner, left, right,
full outer, cross, semi, and anti joins with practical examples.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast

spark = SparkSession.builder \
    .appName("JoinsSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample DataFrames ─────────────────────────────────────────────────────────
employees = spark.createDataFrame([
    (1, "Alice",   "E01"),
    (2, "Bob",     "E02"),
    (3, "Charlie", "E03"),
    (4, "Diana",   "E99"),   # no matching department
    (5, "Eve",     None),    # null department id
], ["emp_id", "name", "dept_id"])

departments = spark.createDataFrame([
    ("E01", "Engineering", "San Francisco"),
    ("E02", "Marketing",   "New York"),
    ("E03", "HR",          "Chicago"),
    ("E04", "Finance",     "Austin"),    # no matching employee
], ["dept_id", "dept_name", "location"])

print("EMPLOYEES:")
employees.show()
print("DEPARTMENTS:")
departments.show()

# ── Inner Join ────────────────────────────────────────────────────────────────
print("=" * 55)
print("INNER JOIN — Only matching rows from both sides")
print("=" * 55)
employees.join(departments, "dept_id", "inner").show()

# ── Left Join ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("LEFT JOIN — All employees, nulls for unmatched dept")
print("=" * 55)
employees.join(departments, "dept_id", "left").show()

# ── Right Join ────────────────────────────────────────────────────────────────
print("=" * 55)
print("RIGHT JOIN — All departments, nulls for unmatched emp")
print("=" * 55)
employees.join(departments, "dept_id", "right").show()

# ── Full Outer Join ───────────────────────────────────────────────────────────
print("=" * 55)
print("FULL OUTER JOIN — All rows from both sides")
print("=" * 55)
employees.join(departments, "dept_id", "full").show()

# ── Left Semi Join ────────────────────────────────────────────────────────────
print("=" * 55)
print("LEFT SEMI JOIN — Employees WITH a matching department")
print("=" * 55)
employees.join(departments, "dept_id", "left_semi").show()

# ── Left Anti Join ────────────────────────────────────────────────────────────
print("=" * 55)
print("LEFT ANTI JOIN — Employees WITHOUT a matching department")
print("=" * 55)
employees.join(departments, "dept_id", "left_anti").show()

# ── Cross Join ────────────────────────────────────────────────────────────────
print("=" * 55)
print("CROSS JOIN — Every employee paired with every department")
print("=" * 55)
employees.select("emp_id", "name").crossJoin(
    departments.select("dept_name")
).orderBy("emp_id").show()

# ── Broadcast Join ────────────────────────────────────────────────────────────
print("=" * 55)
print("BROADCAST JOIN — Small table broadcast to all partitions")
print("=" * 55)
employees.join(
    broadcast(departments), "dept_id", "inner"
).show()
print("  departments (4 rows) broadcast → no shuffle join")

# ── Join on Multiple Conditions ───────────────────────────────────────────────
print("=" * 55)
print("JOIN ON MULTIPLE CONDITIONS")
print("=" * 55)

orders = spark.createDataFrame([
    (1, "E01", "Laptop",  1200),
    (2, "E01", "Phone",    800),
    (3, "E02", "Tablet",   600),
    (4, "E03", "Monitor",  400),
], ["order_id", "dept_id", "product", "amount"])

result = departments.join(
    orders,
    (departments["dept_id"] == orders["dept_id"]) & (orders["amount"] > 500),
    "inner"
).select(
    departments["dept_name"],
    departments["location"],
    orders["product"],
    orders["amount"]
)
result.show()

spark.stop()
