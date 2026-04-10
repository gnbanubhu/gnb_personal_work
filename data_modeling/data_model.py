"""
data_modeling/data_model.py
----------------------------
Retail System Data Model using PySpark.

Implements a complete Retail data model with:
  - Dimension Tables : customers, products, stores, employees, dates, promotions
  - Fact Tables      : sales_transactions, inventory, returns
  - Relationships    : Foreign key constraints via schema definitions
  - Star Schema      : Fact tables at the center, dimensions surrounding them

Schema Design:
                        dim_dates
                           │
    dim_promotions    dim_stores    dim_employees
           │               │               │
           └───────── fact_sales ──────────┘
                           │
    dim_customers ─────────┘────── dim_products
                           │
                    fact_inventory
                           │
                    fact_returns

Usage:
    python data_model.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
    DateType, BooleanType
)
from datetime import date

spark = SparkSession.builder \
    .appName("RetailDataModel") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION TABLES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. dim_customers ──────────────────────────────────────────────────────────
section("DIM_CUSTOMERS")

dim_customers_schema = StructType([
    StructField("customer_id",    StringType(),  nullable=False),
    StructField("first_name",     StringType(),  nullable=False),
    StructField("last_name",      StringType(),  nullable=False),
    StructField("email",          StringType(),  nullable=True),
    StructField("phone",          StringType(),  nullable=True),
    StructField("gender",         StringType(),  nullable=True),
    StructField("date_of_birth",  DateType(),    nullable=True),
    StructField("city",           StringType(),  nullable=True),
    StructField("state",          StringType(),  nullable=True),
    StructField("country",        StringType(),  nullable=True),
    StructField("zip_code",       StringType(),  nullable=True),
    StructField("segment",        StringType(),  nullable=True),   # Premium, Regular, Budget
    StructField("loyalty_points", IntegerType(), nullable=True),
    StructField("is_active",      BooleanType(), nullable=False),
    StructField("created_at",     DateType(),    nullable=False),
])

dim_customers_data = [
    ("C001", "Alice",   "Johnson",  "alice@email.com",   "555-1001", "F", date(1990, 5, 15),  "New York",    "NY", "USA", "10001", "Premium", 2500, True,  date(2020, 1, 10)),
    ("C002", "Bob",     "Smith",    "bob@email.com",     "555-1002", "M", date(1985, 8, 22),  "Los Angeles", "CA", "USA", "90001", "Regular", 800,  True,  date(2021, 3, 5)),
    ("C003", "Charlie", "Brown",    "charlie@email.com", "555-1003", "M", date(1992, 11, 3),  "Chicago",     "IL", "USA", "60601", "Budget",  150,  True,  date(2022, 6, 18)),
    ("C004", "Diana",   "Williams", "diana@email.com",   "555-1004", "F", date(1988, 2, 28),  "Houston",     "TX", "USA", "77001", "Premium", 4100, True,  date(2019, 9, 14)),
    ("C005", "Eve",     "Davis",    "eve@email.com",     "555-1005", "F", date(1995, 7, 19),  "Phoenix",     "AZ", "USA", "85001", "Regular", 620,  False, date(2021, 11, 22)),
]

dim_customers = spark.createDataFrame(dim_customers_data, schema=dim_customers_schema)
print(f"\n  Rows: {dim_customers.count()} | Columns: {len(dim_customers.columns)}")
dim_customers.show(truncate=False)


# ── 2. dim_products ───────────────────────────────────────────────────────────
section("DIM_PRODUCTS")

dim_products_schema = StructType([
    StructField("product_id",    StringType(),  nullable=False),
    StructField("product_name",  StringType(),  nullable=False),
    StructField("category",      StringType(),  nullable=False),
    StructField("sub_category",  StringType(),  nullable=True),
    StructField("brand",         StringType(),  nullable=True),
    StructField("supplier_id",   StringType(),  nullable=True),
    StructField("unit_cost",     DoubleType(),  nullable=False),
    StructField("unit_price",    DoubleType(),  nullable=False),
    StructField("weight_kg",     DoubleType(),  nullable=True),
    StructField("is_perishable", BooleanType(), nullable=False),
    StructField("is_active",     BooleanType(), nullable=False),
])

dim_products_data = [
    ("P001", "Laptop Pro 15",       "Electronics", "Computers",   "TechBrand",  "S001", 750.00, 1299.99, 2.1,  False, True),
    ("P002", "Wireless Headphones", "Electronics", "Audio",       "SoundMax",   "S002", 45.00,  149.99,  0.3,  False, True),
    ("P003", "Running Shoes",       "Footwear",    "Sports",      "SpeedRun",   "S003", 35.00,  89.99,   0.8,  False, True),
    ("P004", "Organic Milk 1L",     "Grocery",     "Dairy",       "FreshFarm",  "S004", 1.20,   3.49,    1.0,  True,  True),
    ("P005", "Office Chair",        "Furniture",   "Office",      "ComfortSit", "S005", 120.00, 299.99,  12.5, False, True),
    ("P006", "Python Cookbook",     "Books",       "Programming", "TechPress",  "S006", 15.00,  49.99,   0.6,  False, True),
    ("P007", "Yoga Mat",            "Sports",      "Fitness",     "FlexFit",    "S003", 12.00,  34.99,   1.2,  False, True),
    ("P008", "Coffee Beans 500g",   "Grocery",     "Beverages",   "BeanMaster", "S004", 5.00,   14.99,   0.5,  True,  True),
]

dim_products = spark.createDataFrame(dim_products_data, schema=dim_products_schema)
print(f"\n  Rows: {dim_products.count()} | Columns: {len(dim_products.columns)}")
dim_products.show(truncate=False)


# ── 3. dim_stores ─────────────────────────────────────────────────────────────
section("DIM_STORES")

dim_stores_schema = StructType([
    StructField("store_id",      StringType(),  nullable=False),
    StructField("store_name",    StringType(),  nullable=False),
    StructField("store_type",    StringType(),  nullable=False),  # Flagship, Standard, Express
    StructField("city",          StringType(),  nullable=False),
    StructField("state",         StringType(),  nullable=False),
    StructField("country",       StringType(),  nullable=False),
    StructField("zip_code",      StringType(),  nullable=True),
    StructField("region",        StringType(),  nullable=False),  # North, South, East, West
    StructField("area_sqft",     IntegerType(), nullable=True),
    StructField("num_employees", IntegerType(), nullable=True),
    StructField("opened_date",   DateType(),    nullable=False),
    StructField("is_active",     BooleanType(), nullable=False),
])

dim_stores_data = [
    ("ST001", "NYC Flagship",    "Flagship", "New York",    "NY", "USA", "10001", "East",  12000, 85, date(2015, 3, 1),  True),
    ("ST002", "LA Downtown",     "Standard", "Los Angeles", "CA", "USA", "90001", "West",  8000,  52, date(2017, 6, 15), True),
    ("ST003", "Chicago Central", "Standard", "Chicago",     "IL", "USA", "60601", "North", 7500,  48, date(2018, 9, 10), True),
    ("ST004", "Houston Hub",     "Express",  "Houston",     "TX", "USA", "77001", "South", 4000,  28, date(2020, 1, 20), True),
    ("ST005", "Phoenix Express", "Express",  "Phoenix",     "AZ", "USA", "85001", "West",  3500,  22, date(2021, 5, 5),  True),
]

dim_stores = spark.createDataFrame(dim_stores_data, schema=dim_stores_schema)
print(f"\n  Rows: {dim_stores.count()} | Columns: {len(dim_stores.columns)}")
dim_stores.show(truncate=False)


# ── 4. dim_employees ──────────────────────────────────────────────────────────
section("DIM_EMPLOYEES")

dim_employees_schema = StructType([
    StructField("employee_id", StringType(),  nullable=False),
    StructField("first_name",  StringType(),  nullable=False),
    StructField("last_name",   StringType(),  nullable=False),
    StructField("role",        StringType(),  nullable=False),  # Cashier, Manager, Associate
    StructField("store_id",    StringType(),  nullable=False),  # FK → dim_stores
    StructField("hire_date",   DateType(),    nullable=False),
    StructField("salary",      DoubleType(),  nullable=True),
    StructField("is_active",   BooleanType(), nullable=False),
])

dim_employees_data = [
    ("E001", "Frank",  "Miller",   "Manager",   "ST001", date(2015, 3, 1),  75000.0, True),
    ("E002", "Grace",  "Wilson",   "Cashier",   "ST001", date(2019, 7, 15), 38000.0, True),
    ("E003", "Henry",  "Moore",    "Associate", "ST002", date(2018, 2, 10), 42000.0, True),
    ("E004", "Isabel", "Taylor",   "Manager",   "ST003", date(2018, 9, 10), 72000.0, True),
    ("E005", "Jack",   "Anderson", "Cashier",   "ST004", date(2020, 3, 5),  37000.0, True),
]

dim_employees = spark.createDataFrame(dim_employees_data, schema=dim_employees_schema)
print(f"\n  Rows: {dim_employees.count()} | Columns: {len(dim_employees.columns)}")
dim_employees.show(truncate=False)


# ── 5. dim_promotions ─────────────────────────────────────────────────────────
section("DIM_PROMOTIONS")

dim_promotions_schema = StructType([
    StructField("promo_id",      StringType(),  nullable=True),
    StructField("promo_name",    StringType(),  nullable=False),
    StructField("promo_type",    StringType(),  nullable=False),  # Discount, BOGO, Clearance
    StructField("discount_pct",  DoubleType(),  nullable=True),
    StructField("start_date",    DateType(),    nullable=False),
    StructField("end_date",      DateType(),    nullable=False),
    StructField("applicable_to", StringType(),  nullable=True),   # All, Electronics, Grocery
    StructField("is_active",     BooleanType(), nullable=False),
])

dim_promotions_data = [
    ("PR001", "Summer Sale",     "Discount",  20.0, date(2026, 6, 1),  date(2026, 6, 30), "All",         True),
    ("PR002", "Electronics Week","Discount",  15.0, date(2026, 4, 1),  date(2026, 4, 7),  "Electronics", True),
    ("PR003", "Buy One Get One", "BOGO",      None, date(2026, 3, 15), date(2026, 3, 31), "Grocery",     False),
    ("PR004", "Clearance Event", "Clearance", 40.0, date(2026, 5, 1),  date(2026, 5, 15), "Footwear",    True),
    (None,    "No Promotion",    "None",       0.0, date(2000, 1, 1),  date(2099, 12, 31),"All",         True),
]

dim_promotions = spark.createDataFrame(dim_promotions_data, schema=dim_promotions_schema)
print(f"\n  Rows: {dim_promotions.count()} | Columns: {len(dim_promotions.columns)}")
dim_promotions.show(truncate=False)


# ── 6. dim_dates ──────────────────────────────────────────────────────────────
section("DIM_DATES")

dim_dates_schema = StructType([
    StructField("date_id",      StringType(),  nullable=False),  # YYYYMMDD
    StructField("full_date",    DateType(),    nullable=False),
    StructField("day",          IntegerType(), nullable=False),
    StructField("month",        IntegerType(), nullable=False),
    StructField("month_name",   StringType(),  nullable=False),
    StructField("quarter",      IntegerType(), nullable=False),
    StructField("year",         IntegerType(), nullable=False),
    StructField("week_of_year", IntegerType(), nullable=False),
    StructField("day_of_week",  StringType(),  nullable=False),
    StructField("is_weekend",   BooleanType(), nullable=False),
    StructField("is_holiday",   BooleanType(), nullable=False),
    StructField("fiscal_year",  IntegerType(), nullable=False),
])

dim_dates_data = [
    ("20260401", date(2026, 4, 1), 1, 4, "April", 2, 2026, 14, "Wednesday", False, False, 2026),
    ("20260402", date(2026, 4, 2), 2, 4, "April", 2, 2026, 14, "Thursday",  False, False, 2026),
    ("20260403", date(2026, 4, 3), 3, 4, "April", 2, 2026, 14, "Friday",    False, False, 2026),
    ("20260404", date(2026, 4, 4), 4, 4, "April", 2, 2026, 14, "Saturday",  True,  False, 2026),
    ("20260405", date(2026, 4, 5), 5, 4, "April", 2, 2026, 14, "Sunday",    True,  False, 2026),
    ("20260406", date(2026, 4, 6), 6, 4, "April", 2, 2026, 15, "Monday",    False, False, 2026),
    ("20260407", date(2026, 4, 7), 7, 4, "April", 2, 2026, 15, "Tuesday",   False, False, 2026),
    ("20260408", date(2026, 4, 8), 8, 4, "April", 2, 2026, 15, "Wednesday", False, False, 2026),
    ("20260409", date(2026, 4, 9), 9, 4, "April", 2, 2026, 15, "Thursday",  False, False, 2026),
]

dim_dates = spark.createDataFrame(dim_dates_data, schema=dim_dates_schema)
print(f"\n  Rows: {dim_dates.count()} | Columns: {len(dim_dates.columns)}")
dim_dates.show(truncate=False)


# ══════════════════════════════════════════════════════════════════════════════
# FACT TABLES
# ══════════════════════════════════════════════════════════════════════════════

# ── 7. fact_sales ─────────────────────────────────────────────────────────────
section("FACT_SALES")

fact_sales_schema = StructType([
    StructField("sale_id",         StringType(),  nullable=False),
    StructField("date_id",         StringType(),  nullable=False),  # FK → dim_dates
    StructField("customer_id",     StringType(),  nullable=False),  # FK → dim_customers
    StructField("product_id",      StringType(),  nullable=False),  # FK → dim_products
    StructField("store_id",        StringType(),  nullable=False),  # FK → dim_stores
    StructField("employee_id",     StringType(),  nullable=False),  # FK → dim_employees
    StructField("promo_id",        StringType(),  nullable=True),   # FK → dim_promotions
    StructField("quantity",        IntegerType(), nullable=False),
    StructField("unit_price",      DoubleType(),  nullable=False),
    StructField("discount_amount", DoubleType(),  nullable=False),
    StructField("tax_amount",      DoubleType(),  nullable=False),
    StructField("gross_amount",    DoubleType(),  nullable=False),
    StructField("net_amount",      DoubleType(),  nullable=False),
    StructField("payment_method",  StringType(),  nullable=True),   # Cash, Card, Online
    StructField("channel",         StringType(),  nullable=False),  # In-store, Online
])

fact_sales_data = [
    ("S001", "20260401", "C001", "P001", "ST001", "E001", "PR002", 1, 1299.99, 195.00, 98.40,  1299.99, 1007.39, "Card",   "In-store"),
    ("S002", "20260401", "C002", "P003", "ST002", "E003", None,    2, 89.99,   0.00,   14.40,  179.98,  165.58,  "Cash",   "In-store"),
    ("S003", "20260402", "C003", "P008", "ST003", "E004", "PR003", 4, 14.99,   0.00,   4.80,   59.96,   55.16,   "Card",   "In-store"),
    ("S004", "20260403", "C004", "P002", "ST001", "E002", "PR002", 1, 149.99,  22.50,  10.20,  149.99,  117.29,  "Online", "Online"),
    ("S005", "20260404", "C001", "P006", "ST001", "E001", None,    3, 49.99,   0.00,   12.00,  149.97,  137.97,  "Card",   "In-store"),
    ("S006", "20260405", "C005", "P007", "ST005", "E005", "PR004", 2, 34.99,   27.99,  3.36,   69.98,   38.63,   "Cash",   "In-store"),
    ("S007", "20260406", "C002", "P004", "ST002", "E003", "PR003", 6, 3.49,    0.00,   1.68,   20.94,   19.26,   "Card",   "In-store"),
    ("S008", "20260407", "C004", "P005", "ST001", "E001", None,    1, 299.99,  0.00,   24.00,  299.99,  275.99,  "Online", "Online"),
    ("S009", "20260408", "C003", "P001", "ST003", "E004", "PR002", 1, 1299.99, 195.00, 88.40,  1299.99, 1016.59, "Card",   "In-store"),
    ("S010", "20260409", "C001", "P008", "ST001", "E002", None,    2, 14.99,   0.00,   2.40,   29.98,   27.58,   "Cash",   "In-store"),
]

fact_sales = spark.createDataFrame(fact_sales_data, schema=fact_sales_schema)
print(f"\n  Rows: {fact_sales.count()} | Columns: {len(fact_sales.columns)}")
fact_sales.show(truncate=False)


# ── 8. fact_inventory ─────────────────────────────────────────────────────────
section("FACT_INVENTORY")

fact_inventory_schema = StructType([
    StructField("inventory_id",      StringType(),  nullable=False),
    StructField("date_id",           StringType(),  nullable=False),  # FK → dim_dates
    StructField("product_id",        StringType(),  nullable=False),  # FK → dim_products
    StructField("store_id",          StringType(),  nullable=False),  # FK → dim_stores
    StructField("quantity_on_hand",  IntegerType(), nullable=False),
    StructField("quantity_sold",     IntegerType(), nullable=False),
    StructField("quantity_received", IntegerType(), nullable=False),
    StructField("reorder_level",     IntegerType(), nullable=False),
    StructField("reorder_quantity",  IntegerType(), nullable=False),
    StructField("is_below_reorder",  BooleanType(), nullable=False),
    StructField("unit_cost",         DoubleType(),  nullable=False),
    StructField("inventory_value",   DoubleType(),  nullable=False),
])

fact_inventory_data = [
    ("I001", "20260409", "P001", "ST001", 25,  3,  10, 10, 20, False, 750.00, 18750.00),
    ("I002", "20260409", "P001", "ST002", 8,   1,  0,  10, 20, True,  750.00, 6000.00),
    ("I003", "20260409", "P002", "ST001", 42,  5,  20, 15, 30, False, 45.00,  1890.00),
    ("I004", "20260409", "P003", "ST002", 60,  8,  0,  20, 40, False, 35.00,  2100.00),
    ("I005", "20260409", "P004", "ST003", 120, 24, 50, 50, 100, False, 1.20,  144.00),
    ("I006", "20260409", "P005", "ST001", 15,  2,  5,  5,  10, False, 120.00, 1800.00),
    ("I007", "20260409", "P007", "ST005", 35,  4,  0,  10, 20, False, 12.00,  420.00),
    ("I008", "20260409", "P008", "ST001", 200, 12, 100, 50, 100, False, 5.00, 1000.00),
]

fact_inventory = spark.createDataFrame(fact_inventory_data, schema=fact_inventory_schema)
print(f"\n  Rows: {fact_inventory.count()} | Columns: {len(fact_inventory.columns)}")
fact_inventory.show(truncate=False)


# ── 9. fact_returns ───────────────────────────────────────────────────────────
section("FACT_RETURNS")

fact_returns_schema = StructType([
    StructField("return_id",     StringType(),  nullable=False),
    StructField("sale_id",       StringType(),  nullable=False),  # FK → fact_sales
    StructField("date_id",       StringType(),  nullable=False),  # FK → dim_dates
    StructField("customer_id",   StringType(),  nullable=False),  # FK → dim_customers
    StructField("product_id",    StringType(),  nullable=False),  # FK → dim_products
    StructField("store_id",      StringType(),  nullable=False),  # FK → dim_stores
    StructField("quantity",      IntegerType(), nullable=False),
    StructField("return_amount", DoubleType(),  nullable=False),
    StructField("return_reason", StringType(),  nullable=True),   # Defective, Wrong Item, Changed Mind
    StructField("return_status", StringType(),  nullable=False),  # Approved, Pending, Rejected
])

fact_returns_data = [
    ("R001", "S001", "20260405", "C001", "P001", "ST001", 1, 1007.39, "Defective",    "Approved"),
    ("R002", "S003", "20260406", "C003", "P008", "ST003", 2, 27.58,   "Wrong Item",   "Approved"),
    ("R003", "S006", "20260408", "C005", "P007", "ST005", 1, 19.32,   "Changed Mind", "Pending"),
]

fact_returns = spark.createDataFrame(fact_returns_data, schema=fact_returns_schema)
print(f"\n  Rows: {fact_returns.count()} | Columns: {len(fact_returns.columns)}")
fact_returns.show(truncate=False)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL QUERIES ON THE DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

dim_customers.createOrReplaceTempView("dim_customers")
dim_products.createOrReplaceTempView("dim_products")
dim_stores.createOrReplaceTempView("dim_stores")
dim_employees.createOrReplaceTempView("dim_employees")
dim_promotions.createOrReplaceTempView("dim_promotions")
dim_dates.createOrReplaceTempView("dim_dates")
fact_sales.createOrReplaceTempView("fact_sales")
fact_inventory.createOrReplaceTempView("fact_inventory")
fact_returns.createOrReplaceTempView("fact_returns")


# ── Query 1: Total sales by store ─────────────────────────────────────────────
section("QUERY 1 — Total Sales by Store")
spark.sql("""
    SELECT
        s.store_name,
        s.city,
        s.region,
        COUNT(f.sale_id)                 AS total_transactions,
        SUM(f.quantity)                  AS total_units_sold,
        ROUND(SUM(f.gross_amount),  2)   AS total_gross_sales,
        ROUND(SUM(f.net_amount),    2)   AS total_net_sales,
        ROUND(SUM(f.discount_amount),2)  AS total_discounts
    FROM fact_sales f
    JOIN dim_stores s ON f.store_id = s.store_id
    GROUP BY s.store_name, s.city, s.region
    ORDER BY total_net_sales DESC
""").show(truncate=False)


# ── Query 2: Top selling products ─────────────────────────────────────────────
section("QUERY 2 — Top Selling Products by Revenue")
spark.sql("""
    SELECT
        p.product_name,
        p.category,
        SUM(f.quantity)              AS units_sold,
        ROUND(SUM(f.net_amount), 2)  AS total_revenue,
        ROUND(AVG(f.unit_price), 2)  AS avg_selling_price
    FROM fact_sales f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.product_name, p.category
    ORDER BY total_revenue DESC
""").show(truncate=False)


# ── Query 3: Customer purchase summary ────────────────────────────────────────
section("QUERY 3 — Customer Purchase Summary")
spark.sql("""
    SELECT
        CONCAT(c.first_name, ' ', c.last_name)  AS customer_name,
        c.segment,
        c.city,
        COUNT(f.sale_id)             AS total_orders,
        SUM(f.quantity)              AS total_items,
        ROUND(SUM(f.net_amount), 2)  AS total_spent,
        ROUND(AVG(f.net_amount), 2)  AS avg_order_value
    FROM fact_sales f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY c.first_name, c.last_name, c.segment, c.city
    ORDER BY total_spent DESC
""").show(truncate=False)


# ── Query 4: Sales by category ────────────────────────────────────────────────
section("QUERY 4 — Sales Revenue by Product Category")
spark.sql("""
    SELECT
        p.category,
        COUNT(DISTINCT f.product_id)    AS unique_products,
        SUM(f.quantity)                 AS total_units,
        ROUND(SUM(f.net_amount),    2)  AS total_revenue,
        ROUND(SUM(f.discount_amount),2) AS total_discounts,
        ROUND(SUM(f.tax_amount),    2)  AS total_tax
    FROM fact_sales f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.category
    ORDER BY total_revenue DESC
""").show(truncate=False)


# ── Query 5: Inventory below reorder level ────────────────────────────────────
section("QUERY 5 — Inventory Below Reorder Level")
spark.sql("""
    SELECT
        p.product_name,
        p.category,
        s.store_name,
        i.quantity_on_hand,
        i.reorder_level,
        i.reorder_quantity,
        ROUND(i.inventory_value, 2) AS inventory_value
    FROM fact_inventory i
    JOIN dim_products p ON i.product_id = p.product_id
    JOIN dim_stores   s ON i.store_id   = s.store_id
    WHERE i.is_below_reorder = TRUE
    ORDER BY i.quantity_on_hand ASC
""").show(truncate=False)


# ── Query 6: Promotion effectiveness ─────────────────────────────────────────
section("QUERY 6 — Promotion Effectiveness")
spark.sql("""
    SELECT
        COALESCE(pr.promo_name, 'No Promotion')  AS promotion,
        pr.promo_type,
        COUNT(f.sale_id)                AS transactions,
        SUM(f.quantity)                 AS units_sold,
        ROUND(SUM(f.discount_amount),2) AS total_discounts_given,
        ROUND(SUM(f.net_amount), 2)     AS total_net_revenue
    FROM fact_sales f
    LEFT JOIN dim_promotions pr ON f.promo_id = pr.promo_id
    GROUP BY pr.promo_name, pr.promo_type
    ORDER BY total_net_revenue DESC
""").show(truncate=False)


# ── Query 7: Returns analysis ─────────────────────────────────────────────────
section("QUERY 7 — Returns Analysis")
spark.sql("""
    SELECT
        p.product_name,
        p.category,
        r.return_reason,
        r.return_status,
        COUNT(r.return_id)             AS total_returns,
        ROUND(SUM(r.return_amount), 2) AS total_return_value
    FROM fact_returns r
    JOIN dim_products p ON r.product_id = p.product_id
    GROUP BY p.product_name, p.category, r.return_reason, r.return_status
    ORDER BY total_return_value DESC
""").show(truncate=False)


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("RETAIL DATA MODEL SUMMARY")
print("""
  Table               Type       Columns   Rows   Primary Key
  ──────────────────────────────────────────────────────────────
  dim_customers       Dimension    15        5     customer_id
  dim_products        Dimension    11        8     product_id
  dim_stores          Dimension    12        5     store_id
  dim_employees       Dimension     8        5     employee_id
  dim_promotions      Dimension     8        5     promo_id
  dim_dates           Dimension    12        9     date_id
  fact_sales          Fact         15       10     sale_id
  fact_inventory      Fact         12        8     inventory_id
  fact_returns        Fact         10        3     return_id
  ──────────────────────────────────────────────────────────────

  Schema  : Star Schema
  Pattern : Fact tables reference dimension tables via foreign keys

  Relationships:
    fact_sales      → dim_customers, dim_products, dim_stores,
                      dim_employees, dim_promotions, dim_dates
    fact_inventory  → dim_products, dim_stores, dim_dates
    fact_returns    → fact_sales, dim_customers, dim_products,
                      dim_stores, dim_dates
""")

spark.stop()
