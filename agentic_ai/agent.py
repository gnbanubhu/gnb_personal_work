"""
agentic_ai/agent.py
--------------------
A sample Agentic AI application using the Anthropic Claude API.

Demonstrates a Data Engineering Assistant Agent that:
  - Uses tools (functions) to perform tasks autonomously
  - Decides which tool to call based on the user's request
  - Handles multi-step tool use in an agentic loop
  - Responds with final answers after gathering tool results

Tools available to the agent:
  1. get_spark_job_status   — check status of a Spark job
  2. get_pipeline_metrics   — fetch pipeline performance metrics
  3. list_data_sources      — list available data sources
  4. calculate_data_quality — calculate data quality score for a dataset
  5. get_table_schema       — retrieve schema of a table

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python agent.py
"""

import os
import json
import anthropic

# ── Initialize client ─────────────────────────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )

client = anthropic.Anthropic(api_key=api_key)
MODEL  = "claude-opus-4-6"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — What the agent can do
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_spark_job_status",
        "description": (
            "Check the current status of a Spark job by job ID. "
            "Returns the job state, duration, and number of stages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The Spark job ID to check (e.g. 'job_001')"
                }
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "get_pipeline_metrics",
        "description": (
            "Fetch performance metrics for a data pipeline. "
            "Returns throughput, latency, error rate, and last run time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pipeline_name": {
                    "type": "string",
                    "description": "Name of the data pipeline (e.g. 'sales_etl')"
                }
            },
            "required": ["pipeline_name"]
        }
    },
    {
        "name": "list_data_sources",
        "description": "List all available data sources in the data platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "enum": ["all", "database", "streaming", "file"],
                    "description": "Filter by source type. Use 'all' to list everything."
                }
            },
            "required": ["source_type"]
        }
    },
    {
        "name": "calculate_data_quality",
        "description": (
            "Calculate the data quality score for a given dataset. "
            "Checks completeness, accuracy, consistency, and timeliness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the dataset to evaluate"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "get_table_schema",
        "description": "Retrieve the schema (columns and data types) of a table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table (e.g. 'sales_transactions')"
                }
            },
            "required": ["table_name"]
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS — Simulated responses (mock data)
# ══════════════════════════════════════════════════════════════════════════════

def get_spark_job_status(job_id: str) -> dict:
    jobs = {
        "job_001": {"job_id": "job_001", "status": "RUNNING",   "duration_mins": 12, "stages": 8,  "progress": "75%"},
        "job_002": {"job_id": "job_002", "status": "COMPLETED", "duration_mins": 45, "stages": 12, "progress": "100%"},
        "job_003": {"job_id": "job_003", "status": "FAILED",    "duration_mins": 3,  "stages": 2,  "progress": "20%", "error": "OutOfMemoryError"},
    }
    return jobs.get(job_id, {"job_id": job_id, "status": "NOT_FOUND"})


def get_pipeline_metrics(pipeline_name: str) -> dict:
    pipelines = {
        "sales_etl": {
            "pipeline": "sales_etl",
            "throughput_rows_per_sec": 15000,
            "avg_latency_ms": 320,
            "error_rate_pct": 0.02,
            "last_run": "2026-04-09 18:00:00",
            "status": "healthy"
        },
        "customer_sync": {
            "pipeline": "customer_sync",
            "throughput_rows_per_sec": 4200,
            "avg_latency_ms": 850,
            "error_rate_pct": 1.5,
            "last_run": "2026-04-09 17:30:00",
            "status": "degraded"
        },
        "inventory_stream": {
            "pipeline": "inventory_stream",
            "throughput_rows_per_sec": 32000,
            "avg_latency_ms": 95,
            "error_rate_pct": 0.0,
            "last_run": "2026-04-09 18:05:00",
            "status": "healthy"
        }
    }
    return pipelines.get(pipeline_name, {"pipeline": pipeline_name, "status": "NOT_FOUND"})


def list_data_sources(source_type: str) -> dict:
    sources = {
        "database": [
            {"name": "postgres_prod",    "type": "database",  "tables": 48, "size_gb": 320},
            {"name": "mysql_warehouse",  "type": "database",  "tables": 22, "size_gb": 180},
        ],
        "streaming": [
            {"name": "kafka_events",     "type": "streaming", "topics": 12, "throughput": "50k/sec"},
            {"name": "kinesis_logs",     "type": "streaming", "topics": 5,  "throughput": "10k/sec"},
        ],
        "file": [
            {"name": "s3_data_lake",     "type": "file",      "buckets": 8, "size_tb": 12.5},
            {"name": "hdfs_archive",     "type": "file",      "buckets": 3, "size_tb": 45.0},
        ]
    }
    if source_type == "all":
        return {"sources": sources["database"] + sources["streaming"] + sources["file"]}
    return {"sources": sources.get(source_type, [])}


def calculate_data_quality(dataset_name: str) -> dict:
    datasets = {
        "sales_transactions": {
            "dataset": "sales_transactions",
            "overall_score": 94.5,
            "completeness": 98.2,
            "accuracy": 96.1,
            "consistency": 91.3,
            "timeliness": 92.4,
            "total_records": 5_200_000,
            "failed_records": 285_000
        },
        "customer_profiles": {
            "dataset": "customer_profiles",
            "overall_score": 78.3,
            "completeness": 82.5,
            "accuracy": 75.0,
            "consistency": 79.1,
            "timeliness": 76.6,
            "total_records": 1_800_000,
            "failed_records": 391_000
        }
    }
    return datasets.get(dataset_name, {"dataset": dataset_name, "status": "NOT_FOUND"})


def get_table_schema(table_name: str) -> dict:
    schemas = {
        "sales_transactions": {
            "table": "sales_transactions",
            "columns": [
                {"name": "transaction_id", "type": "STRING",    "nullable": False},
                {"name": "customer_id",    "type": "STRING",    "nullable": False},
                {"name": "product_id",     "type": "STRING",    "nullable": False},
                {"name": "amount",         "type": "DOUBLE",    "nullable": False},
                {"name": "quantity",       "type": "INTEGER",   "nullable": False},
                {"name": "status",         "type": "STRING",    "nullable": True},
                {"name": "created_at",     "type": "TIMESTAMP", "nullable": False},
            ],
            "partition_by": "created_at",
            "row_count": 5_200_000
        },
        "customer_profiles": {
            "table": "customer_profiles",
            "columns": [
                {"name": "customer_id",  "type": "STRING",    "nullable": False},
                {"name": "name",         "type": "STRING",    "nullable": False},
                {"name": "email",        "type": "STRING",    "nullable": True},
                {"name": "region",       "type": "STRING",    "nullable": True},
                {"name": "segment",      "type": "STRING",    "nullable": True},
                {"name": "created_at",   "type": "TIMESTAMP", "nullable": False},
            ],
            "partition_by": "region",
            "row_count": 1_800_000
        }
    }
    return schemas.get(table_name, {"table": table_name, "status": "NOT_FOUND"})


# ── Tool dispatcher ───────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return its result as JSON string."""
    dispatch = {
        "get_spark_job_status":   lambda: get_spark_job_status(**tool_input),
        "get_pipeline_metrics":   lambda: get_pipeline_metrics(**tool_input),
        "list_data_sources":      lambda: list_data_sources(**tool_input),
        "calculate_data_quality": lambda: calculate_data_quality(**tool_input),
        "get_table_schema":       lambda: get_table_schema(**tool_input),
    }
    result = dispatch[tool_name]()
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP — Core agent execution
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(user_query: str) -> str:
    """
    Run the agent with an agentic loop:
      1. Send query + tools to Claude
      2. If Claude requests a tool → execute it → send result back
      3. Repeat until Claude returns a final text response
    """
    print(f"\n  User     : {user_query}")
    print(f"  {'─' * 54}")

    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are a Data Engineering Assistant Agent. "
                "You have access to tools that provide real-time information "
                "about Spark jobs, pipelines, data sources, data quality, and table schemas. "
                "Always use the available tools to fetch accurate information before answering. "
                "Be concise and structured in your responses."
            ),
            tools=TOOLS,
            messages=messages
        )

        # ── Tool use requested ────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Add assistant's response (with tool calls) to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each requested tool
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool     : {block.name}({json.dumps(block.input)})")
                    result = execute_tool(block.name, block.input)
                    print(f"  Result   : {result[:120]}{'...' if len(result) > 120 else ''}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        # ── Final response ────────────────────────────────────────────────────
        elif response.stop_reason == "end_turn":
            final_answer = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            print(f"\n  Agent    :\n")
            for line in final_answer.splitlines():
                print(f"    {line}")
            return final_answer


# ══════════════════════════════════════════════════════════════════════════════
# RUN AGENT WITH SAMPLE QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def header(title):
    print("\n" + "█" * 60)
    print(f"  {title}")
    print("█" * 60)


header("DATA ENGINEERING ASSISTANT AGENT")
print(f"\n  Model : {MODEL}")
print(f"  Tools : {', '.join(t['name'] for t in TOOLS)}")

# Query 1 — Single tool use
header("QUERY 1 : Spark Job Status")
run_agent("What is the status of Spark job job_002?")

# Query 2 — Single tool use
header("QUERY 2 : Pipeline Health Check")
run_agent("Check the metrics for the sales_etl pipeline and tell me if it's healthy.")

# Query 3 — Multi-tool use (agent uses multiple tools in one query)
header("QUERY 3 : Multi-tool — Data Quality + Schema")
run_agent(
    "I want a full report on the sales_transactions dataset. "
    "Give me its schema and data quality score."
)

# Query 4 — List + recommend
header("QUERY 4 : List Data Sources")
run_agent("List all available streaming data sources on the platform.")

# Query 5 — Complex multi-step reasoning
header("QUERY 5 : Complex — Pipeline + Job Investigation")
run_agent(
    "The customer_sync pipeline seems slow. "
    "Check its metrics and also check the status of job_003. "
    "Based on the findings, what do you recommend?"
)

print("\n" + "═" * 60)
print("  Agent session complete.")
print("═" * 60)
