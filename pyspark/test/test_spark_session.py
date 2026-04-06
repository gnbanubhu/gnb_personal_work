"""
test_spark_session.py
---------------------
Unit tests for spark_session.py.

Tests cover:
    - SparkSession creation and configuration
    - Sample DataFrame schema validation
    - Sample DataFrame content and row count validation

Usage:
    pytest pyspark/test/test_spark_session.py
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

import sys
import os

# Allow imports from the pyspark folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spark_session import create_sample_dataframe, create_spark_session


@pytest.fixture(scope="module")
def spark():
    """
    Fixture that provides a shared SparkSession for all tests in this module.
    The session is stopped once all tests complete.
    """
    session = create_spark_session("TestApp")
    yield session
    session.stop()


class TestCreateSparkSession:
    """Tests for the create_spark_session() function."""

    def test_returns_spark_session_instance(self, spark):
        """create_spark_session() should return a SparkSession instance."""
        assert isinstance(spark, SparkSession)

    def test_app_name(self, spark):
        """SparkSession should have the correct application name."""
        assert spark.sparkContext.appName == "TestApp"

    def test_spark_context_is_active(self, spark):
        """SparkContext should be active after session creation."""
        assert spark.sparkContext is not None

    def test_default_app_name(self):
        """
        create_spark_session() default app name is 'SampleApp'.
        PySpark's getOrCreate() reuses any active session, so we verify
        the default parameter value directly rather than spinning up a new session.
        """
        import inspect
        sig = inspect.signature(create_spark_session)
        assert sig.parameters["app_name"].default == "SampleApp"


class TestCreateSampleDataframe:
    """Tests for the create_sample_dataframe() function."""

    def test_returns_dataframe(self, spark):
        """create_sample_dataframe() should return a PySpark DataFrame."""
        from pyspark.sql import DataFrame
        df = create_sample_dataframe(spark)
        assert isinstance(df, DataFrame)

    def test_row_count(self, spark):
        """DataFrame should contain exactly 5 rows."""
        df = create_sample_dataframe(spark)
        assert df.count() == 5

    def test_column_names(self, spark):
        """DataFrame should have the correct column names."""
        df = create_sample_dataframe(spark)
        assert df.columns == ["id", "name", "age", "department"]

    def test_schema(self, spark):
        """DataFrame schema should match the expected types."""
        df = create_sample_dataframe(spark)
        expected_schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("age", LongType(), True),
            StructField("department", StringType(), True),
        ])
        assert df.schema == expected_schema

    def test_data_values(self, spark):
        """DataFrame should contain the expected employee records."""
        df = create_sample_dataframe(spark)
        rows = df.orderBy("id").collect()

        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alice"
        assert rows[0]["age"] == 30
        assert rows[0]["department"] == "Engineering"

        assert rows[1]["id"] == 2
        assert rows[1]["name"] == "Bob"
        assert rows[1]["age"] == 25
        assert rows[1]["department"] == "Marketing"

        assert rows[4]["id"] == 5
        assert rows[4]["name"] == "Eve"
        assert rows[4]["age"] == 32
        assert rows[4]["department"] == "Marketing"

    def test_no_null_values(self, spark):
        """DataFrame should not contain any null values."""
        df = create_sample_dataframe(spark)
        for col in df.columns:
            null_count = df.filter(df[col].isNull()).count()
            assert null_count == 0, f"Column '{col}' contains null values"

    def test_unique_ids(self, spark):
        """All id values in the DataFrame should be unique."""
        df = create_sample_dataframe(spark)
        total = df.count()
        distinct_ids = df.select("id").distinct().count()
        assert total == distinct_ids

    def test_departments(self, spark):
        """DataFrame should only contain expected department values."""
        df = create_sample_dataframe(spark)
        valid_departments = {"Engineering", "Marketing", "HR"}
        actual_departments = {row["department"] for row in df.select("department").collect()}
        assert actual_departments.issubset(valid_departments)
