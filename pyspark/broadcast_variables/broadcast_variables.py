"""
broadcast_variables/sample.py
------------------------------
Demonstrates Spark Broadcast Variables — read-only shared data
distributed efficiently to all executors, avoiding repeated
data transfer per task.

Usage:
    python sample.py
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("BroadcastVariableSample") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# ── Lookup table to broadcast ─────────────────────────────────────────────────
department_info = {
    "Engineering" : {"location": "San Francisco", "budget": 500000},
    "Marketing"   : {"location": "New York",       "budget": 300000},
    "HR"          : {"location": "Chicago",         "budget": 200000},
    "Finance"     : {"location": "Austin",          "budget": 250000},
}

# ── Broadcast the lookup table to all executors ───────────────────────────────
broadcast_dept = sc.broadcast(department_info)

# ── Employee data ─────────────────────────────────────────────────────────────
employees = sc.parallelize([
    ("Alice",   "Engineering", 95000),
    ("Bob",     "Marketing",   72000),
    ("Charlie", "Engineering", 88000),
    ("Diana",   "HR",          65000),
    ("Eve",     "Finance",     81000),
    ("Frank",   "Engineering", 102000),
    ("Grace",   "HR",          68000),
])

# ── Enrich employees using broadcast variable (no shuffle needed) ──────────────
def enrich(record):
    name, dept, salary = record
    info = broadcast_dept.value.get(dept, {})
    return (
        name,
        dept,
        salary,
        info.get("location", "Unknown"),
        info.get("budget",   0),
    )

enriched = employees.map(enrich)

# ── Actions ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("ENRICHED EMPLOYEE DATA (via Broadcast Variable)")
print("=" * 55)
print(f"  {'Name':<10} {'Department':<14} {'Salary':>8}  {'Location':<15} {'Dept Budget':>12}")
print(f"  {'-'*10} {'-'*14} {'-'*8}  {'-'*15} {'-'*12}")
for name, dept, salary, location, budget in sorted(enriched.collect(), key=lambda x: x[2], reverse=True):
    print(f"  {name:<10} {dept:<14} ${salary:>7,}  {location:<15} ${budget:>11,}")

print("\n" + "=" * 55)
print("BROADCAST VARIABLE STATS")
print("=" * 55)
print(f"  Departments broadcast : {len(broadcast_dept.value)}")
print(f"  Employees processed   : {employees.count()}")
print(f"  Data sent to executor : once (not once per task)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
broadcast_dept.unpersist()

spark.stop()
