"""
sql/sakila_mysql.py
--------------------
Connects to MySQL Sakila schema using PySpark JDBC and demonstrates
reading tables, running SQL queries, and basic analysis.

Usage:
    python sakila_mysql.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum, round, desc

JAR_PATH = "/Users/nareshbabugorantla/gnb_personal_work/jars/mysql-connector-j-8.3.0.jar"

spark = SparkSession.builder \
    .appName("SakilaMySQL") \
    .master("local[*]") \
    .config("spark.jars", JAR_PATH) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── JDBC Connection Properties ────────────────────────────────────────────────
jdbc_url = "jdbc:mysql://localhost:3306/sakila"
properties = {
    "user"    : "root",
    "password": "Naresh!@12",
    "driver"  : "com.mysql.cj.jdbc.Driver",
}

def read_table(table_name):
    return spark.read.jdbc(url=jdbc_url, table=table_name, properties=properties)

# ── Load Sakila Tables ────────────────────────────────────────────────────────
print("=" * 55)
print("CONNECTING TO MYSQL — sakila schema")
print("=" * 55)

film        = read_table("film")
actor       = read_table("actor")
customer    = read_table("customer")
rental      = read_table("rental")
payment     = read_table("payment")
category    = read_table("category")
film_actor  = read_table("film_actor")
film_category = read_table("film_category")
inventory   = read_table("inventory")
store       = read_table("store")

print("  Tables loaded successfully from sakila schema.")

# ── Table Overview ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TABLE ROW COUNTS")
print("=" * 55)

tables = {
    "film"         : film,
    "actor"        : actor,
    "customer"     : customer,
    "rental"       : rental,
    "payment"      : payment,
    "category"     : category,
    "film_category": film_category,
    "film_actor"   : film_actor,
    "inventory"    : inventory,
    "store"        : store,
}

for name, df in tables.items():
    print(f"  {name:<20} : {df.count():>5} rows")

# ── Film Table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FILM TABLE — Sample + Schema")
print("=" * 55)

film.printSchema()
film.select("film_id", "title", "release_year", "rating", "rental_rate", "length").show(5)

# ── Top 10 Longest Films ──────────────────────────────────────────────────────
print("=" * 55)
print("TOP 10 LONGEST FILMS")
print("=" * 55)

film.select("title", "length", "rating", "rental_rate") \
    .orderBy(desc("length")) \
    .show(10)

# ── Films by Rating ───────────────────────────────────────────────────────────
print("=" * 55)
print("FILMS COUNT BY RATING")
print("=" * 55)

film.groupBy("rating") \
    .agg(
        count("*").alias("total_films"),
        round(avg("rental_rate"), 2).alias("avg_rental_rate"),
        round(avg("length"), 1).alias("avg_length_mins"),
    ) \
    .orderBy(desc("total_films")) \
    .show()

# ── Top 10 Actors by Number of Films ─────────────────────────────────────────
print("=" * 55)
print("TOP 10 ACTORS BY FILM COUNT")
print("=" * 55)

actor.join(film_actor, "actor_id") \
    .groupBy("actor_id", "first_name", "last_name") \
    .agg(count("film_id").alias("film_count")) \
    .orderBy(desc("film_count")) \
    .select("first_name", "last_name", "film_count") \
    .show(10)

# ── Revenue by Category ───────────────────────────────────────────────────────
print("=" * 55)
print("REVENUE BY FILM CATEGORY")
print("=" * 55)

film.join(film_category, "film_id") \
    .join(category, "category_id") \
    .join(inventory, "film_id") \
    .join(rental, "inventory_id") \
    .join(payment, "rental_id") \
    .groupBy("name") \
    .agg(
        count("payment_id").alias("total_rentals"),
        round(sum("amount"), 2).alias("total_revenue"),
        round(avg("amount"), 2).alias("avg_payment"),
    ) \
    .orderBy(desc("total_revenue")) \
    .show()

# ── Top 10 Customers by Spending ──────────────────────────────────────────────
print("=" * 55)
print("TOP 10 CUSTOMERS BY TOTAL SPENDING")
print("=" * 55)

customer.join(payment, "customer_id") \
    .groupBy("customer_id", "first_name", "last_name", "email") \
    .agg(
        count("payment_id").alias("total_rentals"),
        round(sum("amount"), 2).alias("total_spent"),
    ) \
    .orderBy(desc("total_spent")) \
    .select("first_name", "last_name", "email", "total_rentals", "total_spent") \
    .show(10)

# ── Monthly Rental Trend ──────────────────────────────────────────────────────
print("=" * 55)
print("MONTHLY RENTAL TREND")
print("=" * 55)

from pyspark.sql.functions import year, month, to_date

rental.withColumn("rental_date", col("rental_date").cast("timestamp")) \
    .withColumn("year",  year(col("rental_date"))) \
    .withColumn("month", month(col("rental_date"))) \
    .groupBy("year", "month") \
    .agg(count("rental_id").alias("total_rentals")) \
    .orderBy("year", "month") \
    .show()

spark.stop()
