"""
spark_mllib.py
--------------
Demonstrates Spark MLlib with a Linear Regression pipeline
to predict house prices based on size and number of rooms.

Usage:
    python spark_mllib.py
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("SparkMLlibDemo") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Sample Data ───────────────────────────────────────────────────────────────
data = [
    (750,  2, 150000.0),
    (1000, 3, 200000.0),
    (1200, 3, 230000.0),
    (1500, 4, 280000.0),
    (1800, 4, 330000.0),
    (2000, 4, 370000.0),
    (2500, 5, 450000.0),
    (3000, 5, 540000.0),
    (1100, 3, 210000.0),
    (1600, 4, 295000.0),
]

df = spark.createDataFrame(data, ["size_sqft", "num_rooms", "price"])

# ── Train / Test Split ────────────────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ── Pipeline ──────────────────────────────────────────────────────────────────
assembler = VectorAssembler(inputCols=["size_sqft", "num_rooms"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="price")
pipeline = Pipeline(stages=[assembler, lr])

# ── Train ─────────────────────────────────────────────────────────────────────
model = pipeline.fit(train_df)

# ── Predict ───────────────────────────────────────────────────────────────────
predictions = model.transform(test_df)

print("=" * 55)
print("HOUSE PRICE PREDICTION — SPARK MLLIB")
print("=" * 55)
predictions.select("size_sqft", "num_rooms", "price", "prediction").show()

# ── Evaluate ──────────────────────────────────────────────────────────────────
evaluator_r2   = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
evaluator_rmse = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

print("=" * 55)
print("MODEL METRICS")
print("=" * 55)
print(f"  R² Score : {evaluator_r2.evaluate(predictions):.4f}")
print(f"  RMSE     : ${evaluator_rmse.evaluate(predictions):,.2f}")

lr_model = model.stages[-1]
print(f"\n  Coefficients : {lr_model.coefficients}")
print(f"  Intercept    : {lr_model.intercept:.2f}")

spark.stop()
