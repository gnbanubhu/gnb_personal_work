"""
spark_session.py
----------------
Demonstrates how to create a SparkSession and use it to build a DataFrame
from in-memory sample data.

Usage:
    python spark_session.py
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "SampleApp") -> SparkSession:
    """
    Create and return a SparkSession configured for local execution.

    Args:
        app_name (str): Name of the Spark application. Defaults to "SampleApp".

    Returns:
        SparkSession: An active SparkSession instance running on all local cores.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def create_sample_dataframe(spark: SparkSession):
    """
    Create a sample employee DataFrame using the provided SparkSession.

    The DataFrame contains 5 records with the following columns:
        - id (int): Unique employee identifier.
        - name (str): Employee name.
        - age (int): Employee age.
        - department (str): Department the employee belongs to.

    Args:
        spark (SparkSession): An active SparkSession used to create the DataFrame.

    Returns:
        pyspark.sql.DataFrame: A DataFrame containing sample employee data.
    """
    data = [
        (1, "Alice", 30, "Engineering"),
        (2, "Bob", 25, "Marketing"),
        (3, "Charlie", 35, "Engineering"),
        (4, "Diana", 28, "HR"),
        (5, "Eve", 32, "Marketing"),
    ]
    columns = ["id", "name", "age", "department"]
    return spark.createDataFrame(data, schema=columns)


if __name__ == "__main__":
    # Initialize SparkSession
    spark = create_spark_session("SparkSessionDemo")
    print(f"Spark version: {spark.version}\n")

    # Create sample DataFrame
    df = create_sample_dataframe(spark)

    # Display schema
    print("=== DataFrame Schema ===")
    df.printSchema()

    # Display contents
    print("=== DataFrame Contents ===")
    df.show()

    # Display row count
    print("=== Row Count ===")
    print(f"Total rows: {df.count()}\n")

    # Stop the SparkSession to release resources
    spark.stop()
