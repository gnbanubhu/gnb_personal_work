"""
spark_sql.py
------------
Demonstrates Spark SQL operations including creating temp views,
running SQL queries, joins, and aggregations.

Usage:
    python spark_sql.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum, count, round

spark = SparkSession.builder \
    .appName("SparkSQLDemo") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample Data ───────────────────────────────────────────────────────────────
employees = [
    (1, "Alice",   "Engineering", 95000),
    (2, "Bob",     "Marketing",   72000),
    (3, "Charlie", "Engineering", 88000),
    (4, "Diana",   "HR",          65000),
    (5, "Eve",     "Marketing",   78000),
    (6, "Frank",   "Engineering", 102000),
    (7, "Grace",   "HR",          68000),
]

departments = [
    ("Engineering", "San Francisco"),
    ("Marketing",   "New York"),
    ("HR",          "Chicago"),
]

emp_df  = spark.createDataFrame(employees,  ["id", "name", "department", "salary"])
dept_df = spark.createDataFrame(departments, ["department", "location"])

emp_df.createOrReplaceTempView("employees")
dept_df.createOrReplaceTempView("departments")

# ── SQL Queries ───────────────────────────────────────────────────────────────
print("=" * 55)
print("ALL EMPLOYEES")
print("=" * 55)
spark.sql("SELECT * FROM employees ORDER BY salary DESC").show()

print("=" * 55)
print("AVERAGE SALARY BY DEPARTMENT")
print("=" * 55)
spark.sql("""
    SELECT department,
           COUNT(*)        AS headcount,
           ROUND(AVG(salary), 0) AS avg_salary,
           SUM(salary)     AS total_salary
    FROM employees
    GROUP BY department
    ORDER BY avg_salary DESC
""").show()

print("=" * 55)
print("EMPLOYEES WITH DEPARTMENT LOCATION (JOIN)")
print("=" * 55)
spark.sql("""
    SELECT e.name, e.department, e.salary, d.location
    FROM employees e
    JOIN departments d ON e.department = d.department
    ORDER BY e.salary DESC
""").show()

print("=" * 55)
print("HIGH EARNERS (salary > 80,000)")
print("=" * 55)
spark.sql("SELECT name, department, salary FROM employees WHERE salary > 80000 ORDER BY salary DESC").show()

spark.stop()
