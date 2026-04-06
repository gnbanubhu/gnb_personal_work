"""
integration_test_spark_session.py
----------------------------------
Integration tests for spark_session.py.

Unlike unit tests, these tests exercise the full Spark execution engine —
verifying end-to-end behavior such as SQL queries, aggregations, filtering,
writing/reading data, and DataFrame transformations on a real SparkSession.

Usage:
    pytest pyspark/test/integration_test_spark_session.py -v
"""

import os
import sys
import tempfile

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

# Allow imports from the pyspark folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spark_session import create_sample_dataframe, create_spark_session


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    """
    Module-scoped SparkSession fixture.
    A single session is shared across all integration tests and stopped at teardown.
    """
    session = create_spark_session("IntegrationTestApp")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def employee_df(spark):
    """
    Module-scoped employee DataFrame fixture.
    Reused across tests to avoid re-creating data on every test.
    """
    return create_sample_dataframe(spark)


# ---------------------------------------------------------------------------
# Integration: SparkSession lifecycle
# ---------------------------------------------------------------------------

class TestSparkSessionLifecycle:
    """Verify the SparkSession is fully operational end-to-end."""

    def test_spark_version_is_present(self, spark):
        """Spark version string should be non-empty."""
        assert spark.version and len(spark.version) > 0

    def test_spark_conf_master(self, spark):
        """Session should be configured to run in local mode."""
        master = spark.sparkContext.master
        assert master.startswith("local")

    def test_sql_context_available(self, spark):
        """SparkSession should expose a working SQL context via selectExpr."""
        result = spark.range(1).selectExpr("1 + 1 AS result").collect()
        assert result[0]["result"] == 2

    def test_create_temp_view_and_query(self, spark, employee_df):
        """DataFrame registered as a temp view should be queryable via DataFrame API."""
        employee_df.createOrReplaceTempView("employees")
        result = spark.table("employees").count()
        assert result == 5

    def test_range_dataframe(self, spark):
        """spark.range() should produce the correct number of rows."""
        df = spark.range(10)
        assert df.count() == 10


# ---------------------------------------------------------------------------
# Integration: DataFrame transformations
# ---------------------------------------------------------------------------

class TestDataFrameTransformations:
    """End-to-end tests for filtering, selecting, and transforming data."""

    def test_filter_by_department(self, employee_df):
        """Filtering by department should return only matching rows."""
        engineering = employee_df.filter(F.col("department") == "Engineering")
        assert engineering.count() == 2

    def test_filter_by_age_range(self, employee_df):
        """Filtering employees aged between 28 and 32 should return 3 rows."""
        filtered = employee_df.filter((F.col("age") >= 28) & (F.col("age") <= 32))
        assert filtered.count() == 3

    def test_select_columns(self, employee_df):
        """Selecting a subset of columns should reduce the column count."""
        subset = employee_df.select("name", "department")
        assert subset.columns == ["name", "department"]

    def test_add_derived_column(self, employee_df):
        """Adding a derived column should appear in the resulting DataFrame."""
        df = employee_df.withColumn("senior", F.col("age") > 30)
        assert "senior" in df.columns
        senior_count = df.filter(F.col("senior")).count()
        assert senior_count == 2  # Charlie (35) and Eve (32)

    def test_rename_column(self, employee_df):
        """withColumnRenamed should produce the new column name."""
        df = employee_df.withColumnRenamed("name", "full_name")
        assert "full_name" in df.columns
        assert "name" not in df.columns

    def test_sort_by_age_ascending(self, employee_df):
        """Sorting by age ascending should place the youngest employee first."""
        sorted_df = employee_df.orderBy(F.col("age").asc())
        youngest = sorted_df.first()
        assert youngest["name"] == "Bob"
        assert youngest["age"] == 25

    def test_sort_by_age_descending(self, employee_df):
        """Sorting by age descending should place the oldest employee first."""
        sorted_df = employee_df.orderBy(F.col("age").desc())
        oldest = sorted_df.first()
        assert oldest["name"] == "Charlie"
        assert oldest["age"] == 35

    def test_drop_column(self, employee_df):
        """Dropping a column should remove it from the DataFrame."""
        df = employee_df.drop("age")
        assert "age" not in df.columns
        assert len(df.columns) == 3


# ---------------------------------------------------------------------------
# Integration: Aggregations
# ---------------------------------------------------------------------------

class TestAggregations:
    """End-to-end tests for group-by and aggregate operations."""

    def test_count_by_department(self, employee_df):
        """Group by department should return correct per-department counts."""
        counts = (
            employee_df
            .groupBy("department")
            .count()
            .orderBy("department")
            .collect()
        )
        dept_map = {row["department"]: row["count"] for row in counts}
        assert dept_map["Engineering"] == 2
        assert dept_map["Marketing"] == 2
        assert dept_map["HR"] == 1

    def test_average_age(self, employee_df):
        """Average age across all employees should be 30.0."""
        avg_age = employee_df.agg(F.avg("age").alias("avg_age")).collect()[0]["avg_age"]
        assert avg_age == 30.0

    def test_max_age(self, employee_df):
        """Maximum age should be 35 (Charlie)."""
        max_age = employee_df.agg(F.max("age")).collect()[0]["max(age)"]
        assert max_age == 35

    def test_min_age(self, employee_df):
        """Minimum age should be 25 (Bob)."""
        min_age = employee_df.agg(F.min("age")).collect()[0]["min(age)"]
        assert min_age == 25

    def test_avg_age_per_department(self, employee_df):
        """Average age grouped by department should return correct values."""
        result = (
            employee_df
            .groupBy("department")
            .agg(F.avg("age").alias("avg_age"))
            .orderBy("department")
            .collect()
        )
        dept_avg = {row["department"]: row["avg_age"] for row in result}
        assert dept_avg["Engineering"] == 32.5   # (30 + 35) / 2
        assert dept_avg["Marketing"] == 28.5     # (25 + 32) / 2
        assert dept_avg["HR"] == 28.0


# ---------------------------------------------------------------------------
# Integration: SQL-style queries via Spark temp views and DataFrame API
# ---------------------------------------------------------------------------

class TestSQLQueries:
    """
    End-to-end tests that exercise SQL-style queries via Spark's temp view and
    DataFrame API. spark.sql() is replaced with spark.table() + DataFrame API
    to ensure compatibility with Spark 3.4.x / Py4J version constraints.
    """

    @pytest.fixture(autouse=True)
    def register_view(self, spark, employee_df):
        """Register the employee DataFrame as a SQL temp view before each test."""
        employee_df.createOrReplaceTempView("employees")

    def test_select_all_from_view(self, spark):
        """Reading all rows from a registered temp view should return 5 rows."""
        result = spark.table("employees").collect()
        assert len(result) == 5

    def test_filter_department_from_view(self, spark):
        """Filtering by department on a temp view should return only matching rows."""
        result = spark.table("employees") \
            .filter(F.col("department") == "Engineering") \
            .collect()
        assert len(result) == 2

    def test_aggregation_from_view(self, spark):
        """GROUP BY on a temp view should return correct per-department counts."""
        result = (
            spark.table("employees")
            .groupBy("department")
            .count()
            .orderBy("department")
            .collect()
        )
        dept_map = {row["department"]: row["count"] for row in result}
        assert dept_map["Engineering"] == 2
        assert dept_map["Marketing"] == 2
        assert dept_map["HR"] == 1

    def test_order_by_from_view(self, spark):
        """ORDER BY age DESC on a temp view should place Charlie first."""
        result = spark.table("employees") \
            .orderBy(F.col("age").desc()) \
            .select("name", "age") \
            .collect()
        assert result[0]["name"] == "Charlie"

    def test_avg_age_from_view(self, spark):
        """AVG age on a temp view should return 30.0."""
        result = spark.table("employees") \
            .agg(F.avg("age").alias("avg_age")) \
            .collect()
        assert result[0]["avg_age"] == 30.0


# ---------------------------------------------------------------------------
# Integration: Read / Write (Parquet)
# ---------------------------------------------------------------------------

class TestReadWriteParquet:
    """End-to-end tests for writing and reading back a DataFrame as Parquet."""

    def test_write_and_read_parquet(self, spark, employee_df):
        """DataFrame written as Parquet should be readable with identical contents."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "employees.parquet")
            employee_df.write.mode("overwrite").parquet(path)

            loaded_df = spark.read.parquet(path)
            assert loaded_df.count() == employee_df.count()
            assert set(loaded_df.columns) == set(employee_df.columns)

    def test_parquet_schema_preserved(self, spark, employee_df):
        """Schema should be preserved after a Parquet write/read round-trip."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "employees_schema.parquet")
            employee_df.write.mode("overwrite").parquet(path)

            loaded_df = spark.read.parquet(path)
            assert loaded_df.schema == employee_df.schema

    def test_filtered_write_and_read_parquet(self, spark, employee_df):
        """Only filtered rows should be present after a partial write."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "engineering.parquet")
            employee_df.filter(F.col("department") == "Engineering") \
                       .write.mode("overwrite").parquet(path)

            loaded_df = spark.read.parquet(path)
            assert loaded_df.count() == 2
            departments = {row["department"] for row in loaded_df.collect()}
            assert departments == {"Engineering"}
