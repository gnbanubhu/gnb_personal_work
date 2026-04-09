"""
rdds/rdd_to_df_converter.py
----------------------------
Demonstrates conversion between RDD and DataFrame in PySpark.

  - rdd_to_dataframe() : converts an RDD of Row/tuple/dict to a DataFrame
  - dataframe_to_rdd() : converts a DataFrame back to an RDD of Row objects

Usage:
    python rdd_to_df_converter.py
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .appName("RDDDataFrameConverter") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext


# ── Conversion Functions ──────────────────────────────────────────────────────

def rdd_to_dataframe(rdd, schema=None):
    """
    Convert an RDD to a DataFrame.

    Parameters
    ----------
    rdd    : RDD of Row, tuple, or dict objects
    schema : StructType (optional) — if provided, applies explicit schema;
             if None, Spark infers the schema from the data

    Returns
    -------
    DataFrame
    """
    if schema:
        return spark.createDataFrame(rdd, schema=schema)
    return spark.createDataFrame(rdd)


def dataframe_to_rdd(df):
    """
    Convert a DataFrame to an RDD of Row objects.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    RDD[Row]
    """
    return df.rdd


# ── Sample Data ───────────────────────────────────────────────────────────────

employee_data = [
    Row(name="Alice",   department="Engineering", salary=95000),
    Row(name="Bob",     department="Marketing",   salary=72000),
    Row(name="Charlie", department="Engineering", salary=88000),
    Row(name="Diana",   department="HR",          salary=65000),
    Row(name="Eve",     department="Marketing",   salary=78000),
]

employee_rdd = sc.parallelize(employee_data)

# Explicit schema
employee_schema = StructType([
    StructField("name",       StringType(),  nullable=False),
    StructField("department", StringType(),  nullable=False),
    StructField("salary",     IntegerType(), nullable=False),
])


# ── RDD → DataFrame ───────────────────────────────────────────────────────────

print("=" * 55)
print("STEP 1 : RDD → DataFrame (inferred schema)")
print("=" * 55)
df_inferred = rdd_to_dataframe(employee_rdd)
df_inferred.printSchema()
df_inferred.show()

print("=" * 55)
print("STEP 2 : RDD → DataFrame (explicit schema)")
print("=" * 55)
df_explicit = rdd_to_dataframe(employee_rdd, schema=employee_schema)
df_explicit.printSchema()
df_explicit.show()


# ── DataFrame → RDD ───────────────────────────────────────────────────────────

print("=" * 55)
print("STEP 3 : DataFrame → RDD")
print("=" * 55)
back_to_rdd = dataframe_to_rdd(df_explicit)
print(f"  Type           : {type(back_to_rdd)}")
print(f"  Partition count: {back_to_rdd.getNumPartitions()}")
print(f"  First element  : {back_to_rdd.first()}")
print()
print("  All elements:")
for row in back_to_rdd.collect():
    print(f"    {row}")


# ── RDD Transformations on converted RDD ─────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4 : Apply RDD transformations on converted RDD")
print("=" * 55)

# Filter Engineering employees
eng_rdd = back_to_rdd.filter(lambda r: r.department == "Engineering")
print("\n  Engineering employees (filtered from RDD):")
for row in eng_rdd.collect():
    print(f"    {row.name:<10} | {row.department:<15} | ${row.salary:,}")

# Average salary using RDD aggregation
total_salary = back_to_rdd.map(lambda r: r.salary).sum()
count        = back_to_rdd.count()
avg_salary   = total_salary / count
print(f"\n  Average Salary : ${avg_salary:,.2f}")

spark.stop()
