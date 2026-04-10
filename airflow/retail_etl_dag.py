"""
airflow/retail_etl_dag.py
--------------------------
A sample Apache Airflow DAG for a Retail ETL Pipeline.

Demonstrates:
  1. DAG definition with schedule, retries, and default args
  2. PythonOperator  — extract, transform, load tasks
  3. BashOperator    — shell command tasks
  4. BranchOperator  — conditional branching
  5. EmailOperator   — success/failure notifications (stubbed)
  6. TaskGroup       — grouping related tasks
  7. XCom            — passing data between tasks
  8. Task dependencies (>> and << operators)

Pipeline Flow:
  start
    │
    ▼
  extract_customers ──┐
  extract_products  ──┤──► validate_data ──► branch_on_quality
  extract_sales     ──┘                          │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              quality_passed           quality_failed
                                    │                         │
                                    ▼                         ▼
                           [transform group]          notify_quality_issue
                     transform_customers
                     transform_products
                     transform_sales
                                    │
                                    ▼
                           [load group]
                     load_customers
                     load_products
                     load_sales
                                    │
                                    ▼
                           generate_report
                                    │
                                    ▼
                                  end

Usage:
    Place this file in your Airflow DAGs folder:
        $AIRFLOW_HOME/dags/retail_etl_dag.py

    Or test locally:
        python retail_etl_dag.py
"""

from datetime import datetime, timedelta
import logging
import random

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

default_args = {
    "owner":            "data_engineering",
    "depends_on_past":  False,
    "email":            ["data-team@company.com"],
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Extract ───────────────────────────────────────────────────────────────────

def extract_customers(**context):
    """Extract customer data from source database."""
    log.info("Extracting customers from source database...")
    customers = [
        {"customer_id": "C001", "name": "Alice Johnson",  "segment": "Premium", "city": "New York"},
        {"customer_id": "C002", "name": "Bob Smith",      "segment": "Regular", "city": "Los Angeles"},
        {"customer_id": "C003", "name": "Charlie Brown",  "segment": "Budget",  "city": "Chicago"},
        {"customer_id": "C004", "name": "Diana Williams", "segment": "Premium", "city": "Houston"},
        {"customer_id": "C005", "name": "Eve Davis",      "segment": "Regular", "city": "Phoenix"},
    ]
    log.info(f"Extracted {len(customers)} customers.")
    context["ti"].xcom_push(key="customers_raw", value=customers)
    context["ti"].xcom_push(key="customers_count", value=len(customers))
    return len(customers)


def extract_products(**context):
    """Extract product data from source database."""
    log.info("Extracting products from source database...")
    products = [
        {"product_id": "P001", "name": "Laptop Pro 15",       "category": "Electronics", "price": 1299.99},
        {"product_id": "P002", "name": "Wireless Headphones",  "category": "Electronics", "price": 149.99},
        {"product_id": "P003", "name": "Running Shoes",        "category": "Footwear",    "price": 89.99},
        {"product_id": "P004", "name": "Organic Milk 1L",      "category": "Grocery",     "price": 3.49},
        {"product_id": "P005", "name": "Office Chair",         "category": "Furniture",   "price": 299.99},
    ]
    log.info(f"Extracted {len(products)} products.")
    context["ti"].xcom_push(key="products_raw", value=products)
    context["ti"].xcom_push(key="products_count", value=len(products))
    return len(products)


def extract_sales(**context):
    """Extract sales transaction data from source database."""
    log.info("Extracting sales transactions from source database...")
    sales = [
        {"sale_id": "S001", "customer_id": "C001", "product_id": "P001", "quantity": 1, "amount": 1007.39},
        {"sale_id": "S002", "customer_id": "C002", "product_id": "P003", "quantity": 2, "amount": 165.58},
        {"sale_id": "S003", "customer_id": "C003", "product_id": "P004", "quantity": 4, "amount": 55.16},
        {"sale_id": "S004", "customer_id": "C004", "product_id": "P002", "quantity": 1, "amount": 117.29},
        {"sale_id": "S005", "customer_id": "C001", "product_id": "P005", "quantity": 1, "amount": 275.99},
        {"sale_id": "S006", "customer_id": "C005", "product_id": "P001", "quantity": 1, "amount": 1016.59},
        {"sale_id": "S007", "customer_id": "C002", "product_id": "P004", "quantity": 6, "amount": 19.26},
    ]
    log.info(f"Extracted {len(sales)} sales records.")
    context["ti"].xcom_push(key="sales_raw", value=sales)
    context["ti"].xcom_push(key="sales_count", value=len(sales))
    return len(sales)


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_data(**context):
    """
    Validate extracted data quality.
    Checks: record counts, null fields, duplicate IDs.
    Pushes a quality_score (0-100) to XCom.
    """
    ti = context["ti"]
    customers_count = ti.xcom_pull(task_ids="extract_customers", key="customers_count")
    products_count  = ti.xcom_pull(task_ids="extract_products",  key="products_count")
    sales_count     = ti.xcom_pull(task_ids="extract_sales",     key="sales_count")

    log.info(f"Validating data — customers={customers_count}, products={products_count}, sales={sales_count}")

    issues = []
    if customers_count == 0: issues.append("No customers extracted")
    if products_count  == 0: issues.append("No products extracted")
    if sales_count     == 0: issues.append("No sales extracted")

    quality_score = 100 - (len(issues) * 30)
    quality_score = max(0, quality_score)

    log.info(f"Data quality score: {quality_score}/100 | Issues: {issues or 'None'}")
    ti.xcom_push(key="quality_score", value=quality_score)
    ti.xcom_push(key="quality_issues", value=issues)
    return quality_score


# ── Branch ────────────────────────────────────────────────────────────────────

def branch_on_quality(**context):
    """
    Branch based on data quality score.
    score >= 70 → proceed to transform
    score <  70 → notify quality issue and stop
    """
    ti            = context["ti"]
    quality_score = ti.xcom_pull(task_ids="validate_data", key="quality_score")
    log.info(f"Branching on quality score: {quality_score}")

    if quality_score >= 70:
        log.info("Quality check PASSED → proceeding to transform.")
        return "quality_passed"
    else:
        log.info("Quality check FAILED → routing to notify_quality_issue.")
        return "quality_failed"


def notify_quality_issue(**context):
    """Notify team of data quality failure."""
    ti     = context["ti"]
    issues = ti.xcom_pull(task_ids="validate_data", key="quality_issues")
    log.error(f"DATA QUALITY ALERT: Pipeline halted due to issues: {issues}")
    raise ValueError(f"Data quality check failed: {issues}")


# ── Transform ─────────────────────────────────────────────────────────────────

def transform_customers(**context):
    """Apply transformations to customer data."""
    ti        = context["ti"]
    customers = ti.xcom_pull(task_ids="extract_customers", key="customers_raw")

    transformed = []
    for c in customers:
        transformed.append({
            **c,
            "name_upper":  c["name"].upper(),
            "city_cleaned": c["city"].strip().title(),
            "is_premium":  c["segment"] == "Premium",
            "loaded_at":   datetime.utcnow().isoformat(),
        })

    log.info(f"Transformed {len(transformed)} customer records.")
    ti.xcom_push(key="customers_transformed", value=transformed)
    return len(transformed)


def transform_products(**context):
    """Apply transformations to product data."""
    ti       = context["ti"]
    products = ti.xcom_pull(task_ids="extract_products", key="products_raw")

    transformed = []
    for p in products:
        transformed.append({
            **p,
            "price_with_tax":   round(p["price"] * 1.08, 2),
            "category_upper":   p["category"].upper(),
            "is_high_value":    p["price"] > 200,
            "loaded_at":        datetime.utcnow().isoformat(),
        })

    log.info(f"Transformed {len(transformed)} product records.")
    ti.xcom_push(key="products_transformed", value=transformed)
    return len(transformed)


def transform_sales(**context):
    """Apply transformations to sales data."""
    ti    = context["ti"]
    sales = ti.xcom_pull(task_ids="extract_sales", key="sales_raw")

    transformed = []
    total_revenue = 0.0
    for s in sales:
        total_revenue += s["amount"]
        transformed.append({
            **s,
            "amount_rounded": round(s["amount"], 2),
            "is_bulk_order":  s["quantity"] > 3,
            "loaded_at":      datetime.utcnow().isoformat(),
        })

    log.info(f"Transformed {len(transformed)} sales records. Total revenue: ${total_revenue:,.2f}")
    ti.xcom_push(key="sales_transformed", value=transformed)
    ti.xcom_push(key="total_revenue",     value=round(total_revenue, 2))
    return len(transformed)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_customers(**context):
    """Load transformed customers into data warehouse."""
    ti        = context["ti"]
    customers = ti.xcom_pull(task_ids="transform.transform_customers", key="customers_transformed")
    log.info(f"Loading {len(customers)} customers into data warehouse...")
    for c in customers:
        log.info(f"  INSERT INTO dim_customers: {c['customer_id']} | {c['name']} | premium={c['is_premium']}")
    log.info("Customers loaded successfully.")
    return len(customers)


def load_products(**context):
    """Load transformed products into data warehouse."""
    ti       = context["ti"]
    products = ti.xcom_pull(task_ids="transform.transform_products", key="products_transformed")
    log.info(f"Loading {len(products)} products into data warehouse...")
    for p in products:
        log.info(f"  INSERT INTO dim_products: {p['product_id']} | {p['name']} | tax_price=${p['price_with_tax']}")
    log.info("Products loaded successfully.")
    return len(products)


def load_sales(**context):
    """Load transformed sales into data warehouse."""
    ti    = context["ti"]
    sales = ti.xcom_pull(task_ids="transform.transform_sales", key="sales_transformed")
    log.info(f"Loading {len(sales)} sales into data warehouse...")
    for s in sales:
        log.info(f"  INSERT INTO fact_sales: {s['sale_id']} | ${s['amount_rounded']} | bulk={s['is_bulk_order']}")
    log.info("Sales loaded successfully.")
    return len(sales)


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(**context):
    """Generate a pipeline execution summary report."""
    ti            = context["ti"]
    total_revenue = ti.xcom_pull(task_ids="transform.transform_sales", key="total_revenue")
    quality_score = ti.xcom_pull(task_ids="validate_data",             key="quality_score")
    run_date      = context["ds"]

    report = {
        "run_date":          run_date,
        "pipeline":          "retail_etl",
        "status":            "SUCCESS",
        "quality_score":     quality_score,
        "total_revenue":     total_revenue,
        "customers_loaded":  5,
        "products_loaded":   5,
        "sales_loaded":      7,
    }

    log.info("=" * 50)
    log.info("  RETAIL ETL PIPELINE — EXECUTION REPORT")
    log.info("=" * 50)
    for key, val in report.items():
        log.info(f"  {key:<22}: {val}")
    log.info("=" * 50)

    ti.xcom_push(key="pipeline_report", value=report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="retail_etl_pipeline",
    description="Retail ETL pipeline — extract, validate, transform, load, report",
    default_args=default_args,
    schedule="0 2 * * *",                  # daily at 02:00 AM
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["retail", "etl", "data-engineering"],
) as dag:

    # ── Start ─────────────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Extract tasks ─────────────────────────────────────────────────────────
    t_extract_customers = PythonOperator(
        task_id="extract_customers",
        python_callable=extract_customers,
    )

    t_extract_products = PythonOperator(
        task_id="extract_products",
        python_callable=extract_products,
    )

    t_extract_sales = PythonOperator(
        task_id="extract_sales",
        python_callable=extract_sales,
    )

    # ── Validate ──────────────────────────────────────────────────────────────
    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # ── Branch ────────────────────────────────────────────────────────────────
    t_branch = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=branch_on_quality,
    )

    t_quality_passed = EmptyOperator(task_id="quality_passed")

    t_quality_failed = PythonOperator(
        task_id="quality_failed",
        python_callable=notify_quality_issue,
    )

    # ── Transform TaskGroup ───────────────────────────────────────────────────
    with TaskGroup(group_id="transform") as tg_transform:

        t_transform_customers = PythonOperator(
            task_id="transform_customers",
            python_callable=transform_customers,
        )

        t_transform_products = PythonOperator(
            task_id="transform_products",
            python_callable=transform_products,
        )

        t_transform_sales = PythonOperator(
            task_id="transform_sales",
            python_callable=transform_sales,
        )

    # ── Load TaskGroup ────────────────────────────────────────────────────────
    with TaskGroup(group_id="load") as tg_load:

        t_load_customers = PythonOperator(
            task_id="load_customers",
            python_callable=load_customers,
        )

        t_load_products = PythonOperator(
            task_id="load_products",
            python_callable=load_products,
        )

        t_load_sales = PythonOperator(
            task_id="load_sales",
            python_callable=load_sales,
        )

    # ── Bash task — archive logs ──────────────────────────────────────────────
    t_archive_logs = BashOperator(
        task_id="archive_logs",
        bash_command=(
            "echo 'Archiving ETL logs for run date: {{ ds }}' && "
            "echo 'Log archive complete.'"
        ),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    t_report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    # ── End ───────────────────────────────────────────────────────────────────
    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TASK DEPENDENCIES
    # ══════════════════════════════════════════════════════════════════════════

    # Extract in parallel → validate
    start >> [t_extract_customers, t_extract_products, t_extract_sales] >> t_validate

    # Validate → branch
    t_validate >> t_branch

    # Branch → quality_passed or quality_failed
    t_branch >> [t_quality_passed, t_quality_failed]

    # quality_passed → transform (in parallel)
    t_quality_passed >> tg_transform

    # transform → load (in parallel)
    tg_transform >> tg_load

    # load → archive logs → report → end
    tg_load >> t_archive_logs >> t_report >> end

    # quality_failed also links to end
    t_quality_failed >> end


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST (run without Airflow scheduler)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S"
    )

    print("=" * 60)
    print("  RETAIL ETL DAG — LOCAL TEST RUN")
    print("=" * 60)
    print(f"  DAG ID   : {dag.dag_id}")
    print(f"  Schedule : {dag.schedule}")
    print(f"  Tasks    : {len(dag.tasks)}")
    print(f"  Tags     : {dag.tags}")
    print()

    # Simulate a simple sequential run for local testing
    ctx = {
        "ti":     type("TI", (), {
            "_store": {},
            "xcom_push": lambda self, key, value: self._store.update({key: value}),
            "xcom_pull": lambda self, task_ids, key: self._store.get(key),
        })(),
        "ds": datetime.utcnow().strftime("%Y-%m-%d"),
    }

    print("── Step 1: Extract")
    extract_customers(**ctx)
    extract_products(**ctx)
    extract_sales(**ctx)

    print("\n── Step 2: Validate")
    validate_data(**ctx)

    print("\n── Step 3: Branch")
    branch = branch_on_quality(**ctx)
    print(f"   Branch result → {branch}")

    print("\n── Step 4: Transform")
    transform_customers(**ctx)
    transform_products(**ctx)
    transform_sales(**ctx)

    print("\n── Step 5: Load")
    load_customers(**ctx)
    load_products(**ctx)
    load_sales(**ctx)

    print("\n── Step 6: Report")
    generate_report(**ctx)

    print("\n" + "=" * 60)
    print("  DAG TASK DEPENDENCY GRAPH")
    print("=" * 60)
    for task in dag.topological_sort():
        upstream   = [t.task_id for t in task.upstream_list]
        downstream = [t.task_id for t in task.downstream_list]
        print(f"  {task.task_id:<40} upstream={upstream}")
