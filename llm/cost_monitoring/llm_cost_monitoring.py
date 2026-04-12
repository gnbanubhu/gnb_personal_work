"""
llm/cost_monitoring/llm_cost_monitoring.py
-------------------------------------------
Comprehensive LLM Cost Monitoring guide using the Anthropic Claude API.

Topics covered:
  1.  Token Counting               — count tokens before sending a request
  2.  Usage Tracking               — capture input/output tokens per call
  3.  Cost Calculation             — compute USD cost from token counts
  4.  Request Logger               — log every API call with cost metadata
  5.  Budget Enforcement           — hard-stop when budget limit is reached
  6.  Cost by Model Comparison     — compare costs across Claude model tiers
  7.  Prompt Caching               — reduce costs with cache_control
  8.  Streaming Token Counting     — track tokens in streaming responses
  9.  Batch Cost Estimation        — estimate cost before running a job
  10. Cost Dashboard               — aggregate stats: calls, tokens, spend
  11. Cost Optimization Tips       — techniques to reduce spend
  12. Cost Alert System            — warn when spend crosses thresholds

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python llm_cost_monitoring.py
"""

import os
import json
import time
from datetime        import datetime
from dataclasses     import dataclass, field
from typing          import List, Optional
from collections     import defaultdict

import anthropic

# ── Client ────────────────────────────────────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )

client = anthropic.Anthropic(api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════════
# PRICING TABLE  (USD per million tokens, as of 2026)
# ══════════════════════════════════════════════════════════════════════════════

PRICING = {
    # model_id : (input_per_1M, output_per_1M, cache_write_per_1M, cache_read_per_1M)
    "claude-opus-4-6":     (15.00,  75.00,  18.75,  1.50),
    "claude-sonnet-4-6":   ( 3.00,  15.00,   3.75,  0.30),
    "claude-haiku-4-5-20251001": ( 0.80,   4.00,   1.00,  0.08),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int,
                   cache_write_tokens: int = 0,
                   cache_read_tokens:  int = 0) -> float:
    """Return total USD cost for a single API call."""
    if model not in PRICING:
        model = "claude-opus-4-6"   # default
    inp, out, cw, cr = PRICING[model]
    return (
        input_tokens       / 1_000_000 * inp +
        output_tokens      / 1_000_000 * out +
        cache_write_tokens / 1_000_000 * cw  +
        cache_read_tokens  / 1_000_000 * cr
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN COUNTING
# ══════════════════════════════════════════════════════════════════════════════

section("1. TOKEN COUNTING")
print("""
  Count tokens BEFORE sending a request to estimate cost upfront.
  Uses client.messages.count_tokens() — does NOT make a generation call.

  Why it matters:
    • Avoid surprise costs on large prompts
    • Validate prompts fit within model context limits
    • Estimate batch job cost before running
""")

MODEL = "claude-opus-4-6"

messages_to_count = [
    {"role": "user", "content": "What is Apache Spark and how does it work?"},
]

token_response = client.messages.count_tokens(
    model=MODEL,
    messages=messages_to_count,
)

print(f"\n  Message  : \"{messages_to_count[0]['content']}\"")
print(f"  Model    : {MODEL}")
print(f"  Input tokens (count_tokens) : {token_response.input_tokens}")

# Larger prompt
large_prompt = """You are a data engineering expert. Please provide a comprehensive analysis
of the following Apache Spark job log and identify performance bottlenecks, root causes,
and actionable optimization recommendations with code examples where appropriate.

Log excerpt:
Stage 0: 45 tasks, 12 failed, avg duration 340ms, max 8200ms
Stage 1: 200 tasks, 0 failed, shuffle write 14.2 GB, avg 1200ms
Stage 2: 200 tasks, 0 failed, shuffle read 14.2 GB, avg 890ms
GC time: 18% of executor wall clock time
Spill (memory): 3.8 GB across 24 tasks
Skew detected: partition 47 = 2.1 GB vs avg 72 MB"""

large_count = client.messages.count_tokens(
    model=MODEL,
    messages=[{"role": "user", "content": large_prompt}],
)
est_cost = calculate_cost(MODEL, large_count.input_tokens, 500)

print(f"\n  Large prompt token count : {large_count.input_tokens}")
print(f"  Estimated cost (500 out) : ${est_cost:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. USAGE TRACKING
# ══════════════════════════════════════════════════════════════════════════════

section("2. USAGE TRACKING")
print("""
  Every API response includes a usage object with:
    input_tokens          — tokens in the prompt
    output_tokens         — tokens in the response
    cache_creation_input_tokens  — tokens written to cache
    cache_read_input_tokens      — tokens read from cache
""")

response = client.messages.create(
    model=MODEL,
    max_tokens=200,
    messages=[{"role": "user", "content": "Explain what a data lakehouse is in 2 sentences."}]
)

usage    = response.usage
cost     = calculate_cost(MODEL, usage.input_tokens, usage.output_tokens)

print(f"\n  Response text    : {response.content[0].text[:80]}...")
print(f"\n  Usage breakdown:")
print(f"    input_tokens   : {usage.input_tokens:>8,}")
print(f"    output_tokens  : {usage.output_tokens:>8,}")
print(f"    total_tokens   : {usage.input_tokens + usage.output_tokens:>8,}")
print(f"\n  Cost             : ${cost:.6f}")
print(f"  Stop reason      : {response.stop_reason}")
print(f"  Model            : {response.model}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COST CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

section("3. COST CALCULATION")
print("""
  Pricing breakdown for Claude models (USD per million tokens):
""")

print(f"  {'Model':<28} {'Input/1M':>10} {'Output/1M':>10} {'CacheWr/1M':>12} {'CacheRd/1M':>12}")
print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
for model, (inp, out, cw, cr) in PRICING.items():
    print(f"  {model:<28} ${inp:>9.2f} ${out:>9.2f} ${cw:>11.2f} ${cr:>11.2f}")

# Cost scenarios
subsection("Cost scenarios — 1,000 API calls")

scenarios = [
    ("Simple Q&A",         "claude-haiku-4-5-20251001", 500,   200),
    ("Code generation",    "claude-sonnet-4-6",          2000,  800),
    ("Document analysis",  "claude-opus-4-6",            8000, 1200),
    ("RAG generation",     "claude-sonnet-4-6",          3000,  500),
]

print(f"\n  {'Scenario':<22} {'Model':<28} {'In Tok':>8} {'Out Tok':>8} {'Per Call':>10} {'1K Calls':>10}")
print(f"  {'─'*22} {'─'*28} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
for name, model, inp, out in scenarios:
    per_call  = calculate_cost(model, inp, out)
    per_1k    = per_call * 1000
    print(f"  {name:<22} {model:<28} {inp:>8,} {out:>8,} ${per_call:>9.4f} ${per_1k:>9.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REQUEST LOGGER
# ══════════════════════════════════════════════════════════════════════════════

section("4. REQUEST LOGGER")

@dataclass
class APICallRecord:
    """Stores metadata for a single API call."""
    call_id:       str
    timestamp:     str
    model:         str
    prompt:        str
    response_text: str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    float
    purpose:       str = "general"

class RequestLogger:
    """Logs all API calls with cost and token metadata."""

    def __init__(self):
        self.records: List[APICallRecord] = []
        self._call_counter = 0

    def call(self, prompt: str, model: str = MODEL,
             max_tokens: int = 300, purpose: str = "general") -> str:
        """Make an API call and log all metadata."""
        self._call_counter += 1
        call_id = f"call_{self._call_counter:04d}"
        t_start = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        latency  = (time.time() - t_start) * 1000
        usage    = response.usage
        cost     = calculate_cost(model, usage.input_tokens, usage.output_tokens)
        text     = response.content[0].text

        self.records.append(APICallRecord(
            call_id       = call_id,
            timestamp     = datetime.now().isoformat(timespec="seconds"),
            model         = model,
            prompt        = prompt[:80],
            response_text = text[:80],
            input_tokens  = usage.input_tokens,
            output_tokens = usage.output_tokens,
            cost_usd      = cost,
            latency_ms    = latency,
            purpose       = purpose,
        ))
        return text

    def summary(self) -> None:
        if not self.records:
            print("  No calls logged.")
            return
        total_in   = sum(r.input_tokens  for r in self.records)
        total_out  = sum(r.output_tokens for r in self.records)
        total_cost = sum(r.cost_usd      for r in self.records)
        avg_lat    = sum(r.latency_ms    for r in self.records) / len(self.records)

        print(f"\n  {'Call ID':<12} {'Purpose':<16} {'In':>6} {'Out':>6} {'Cost':>10} {'Latency':>10}")
        print(f"  {'─'*12} {'─'*16} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")
        for r in self.records:
            print(f"  {r.call_id:<12} {r.purpose:<16} {r.input_tokens:>6} "
                  f"{r.output_tokens:>6} ${r.cost_usd:>9.4f} {r.latency_ms:>8.0f}ms")
        print(f"  {'─'*12} {'─'*16} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")
        print(f"  {'TOTAL':<12} {len(self.records)} calls    {total_in:>6} "
              f"{total_out:>6} ${total_cost:>9.4f} {avg_lat:>8.0f}ms avg")


logger = RequestLogger()

logger.call("What is ETL?",                  purpose="qa",       max_tokens=100)
logger.call("List 3 benefits of data lakes", purpose="qa",       max_tokens=150)
logger.call("Write a haiku about Apache Spark", purpose="creative", max_tokens=80)

subsection("Logged API Calls")
logger.summary()


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUDGET ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

section("5. BUDGET ENFORCEMENT")
print("""
  Hard-stop API calls when cumulative spend reaches a budget limit.
  Raises BudgetExceededError before making the call.
""")

class BudgetExceededError(Exception):
    pass

class BudgetedClient:
    """Wraps Anthropic client with a hard budget cap."""

    def __init__(self, budget_usd: float, model: str = MODEL):
        self.budget_usd   = budget_usd
        self.model        = model
        self.spent_usd    = 0.0
        self.call_count   = 0

    @property
    def remaining(self) -> float:
        return self.budget_usd - self.spent_usd

    def call(self, prompt: str, max_tokens: int = 300) -> str:
        # Pre-flight cost estimate
        token_count = client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        est_cost = calculate_cost(self.model, token_count.input_tokens, max_tokens)

        if self.spent_usd + est_cost > self.budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded! Spent ${self.spent_usd:.4f} / ${self.budget_usd:.4f}. "
                f"Estimated call cost: ${est_cost:.4f}"
            )

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        usage     = response.usage
        actual    = calculate_cost(self.model, usage.input_tokens, usage.output_tokens)
        self.spent_usd  += actual
        self.call_count += 1

        print(f"  Call {self.call_count}: ${actual:.4f} | "
              f"Spent ${self.spent_usd:.4f} / ${self.budget_usd:.4f} | "
              f"Remaining: ${self.remaining:.4f}")
        return response.content[0].text

# Demo: small budget
budgeted = BudgetedClient(budget_usd=0.05, model="claude-haiku-4-5-20251001")

prompts = [
    "What is a data warehouse? (1 sentence)",
    "What is a data lake? (1 sentence)",
    "What is a data lakehouse? (1 sentence)",
    "What is Delta Lake? (1 sentence)",
    "What is Apache Iceberg? (1 sentence)",
]

print(f"\n  Budget: ${budgeted.budget_usd:.2f}  |  Model: {budgeted.model}\n")
for p in prompts:
    try:
        budgeted.call(p, max_tokens=80)
    except BudgetExceededError as e:
        print(f"\n  ⚠  {e}")
        break

print(f"\n  Final: {budgeted.call_count} calls | "
      f"${budgeted.spent_usd:.4f} spent | "
      f"${budgeted.remaining:.4f} remaining")


# ══════════════════════════════════════════════════════════════════════════════
# 6. COST BY MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

section("6. COST BY MODEL COMPARISON")

same_prompt = "Summarize the CAP theorem in data engineering in 3 bullet points."

print(f"\n  Prompt: \"{same_prompt}\"\n")
print(f"  {'Model':<28} {'In Tok':>8} {'Out Tok':>8} {'Cost':>10} {'Preview'}")
print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*10} {'─'*35}")

model_results = {}
for model_id in PRICING:
    resp = client.messages.create(
        model=model_id,
        max_tokens=200,
        messages=[{"role": "user", "content": same_prompt}]
    )
    u    = resp.usage
    cost = calculate_cost(model_id, u.input_tokens, u.output_tokens)
    model_results[model_id] = (u.input_tokens, u.output_tokens, cost, resp.content[0].text)
    print(f"  {model_id:<28} {u.input_tokens:>8} {u.output_tokens:>8} "
          f"${cost:>9.4f} {resp.content[0].text[:35].replace(chr(10),' ')}")

# Cost ratio vs Haiku
haiku_cost = model_results["claude-haiku-4-5-20251001"][2]
print(f"\n  Cost multipliers relative to Haiku:")
for model_id, (_, _, cost, _) in model_results.items():
    ratio = cost / haiku_cost if haiku_cost > 0 else 1.0
    print(f"    {model_id:<28} {ratio:>5.1f}×")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PROMPT CACHING
# ══════════════════════════════════════════════════════════════════════════════

section("7. PROMPT CACHING")
print("""
  Prompt caching stores a prefix of the prompt on Anthropic's servers.
  Subsequent calls that reuse the same prefix pay only the cache_read rate
  instead of the full input rate — up to 90% savings on repeated prefixes.

  Cache write : 25% more than base input price (one-time)
  Cache read  : 10% of base input price (every subsequent use)

  Requirements:
    • Minimum 1,024 tokens in the cached block
    • Add cache_control: {"type": "ephemeral"} to the content block
    • Cache TTL: 5 minutes (refreshed on each cache hit)
""")

SYSTEM_PROMPT = """You are a senior data engineering consultant with 15 years of experience.
You specialize in Apache Spark, cloud data platforms (AWS, Azure, GCP), real-time streaming
with Kafka, data warehouse design, pipeline orchestration with Airflow, and ML infrastructure.

When answering questions:
- Be precise and technical
- Provide concrete code examples where relevant
- Reference industry best practices
- Consider trade-offs and scalability implications
- Structure responses with clear sections

The following is the architecture documentation for our data platform:
Our platform processes 500TB of data daily across three zones:
  Landing Zone (S3/ADLS): Raw ingestion, schema-on-read, 90-day retention
  Processing Zone (Spark/Databricks): ETL, validation, enrichment, 30-day retention
  Serving Zone (Snowflake/BigQuery): Analytics-ready, partitioned, 3-year retention

Pipeline SLAs: Bronze layer < 15 min, Silver < 30 min, Gold < 60 min after source arrival.
Critical pipelines: sales_etl, customer_sync, inventory_stream, fraud_detection."""

questions = [
    "How should we optimize our Spark jobs to meet the Bronze layer SLA?",
    "What monitoring should we add to the fraud_detection pipeline?",
]

print(f"\n  System prompt size : {len(SYSTEM_PROMPT.split())} words")

# Call WITH cache_control on the system prompt
print(f"\n  Calling with prompt caching enabled:")
print(f"  {'Call':<6} {'In Tok':>8} {'Out Tok':>8} {'Cache Write':>12} "
      f"{'Cache Read':>11} {'Cost':>10}")
print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*12} {'─'*11} {'─'*10}")

for i, question in enumerate(questions, 1):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{"role": "user", "content": question}]
    )
    u  = resp.usage
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens",     0) or 0
    cost = calculate_cost(MODEL, u.input_tokens, u.output_tokens, cw, cr)
    print(f"  {i:<6} {u.input_tokens:>8} {u.output_tokens:>8} {cw:>12} "
          f"{cr:>11} ${cost:>9.4f}")

# Cost without caching (manual estimate)
base_input = 800   # approximate system prompt tokens
per_call_no_cache = calculate_cost(MODEL, base_input + 50, 200)
per_call_cached   = calculate_cost(MODEL, 50, 200, cache_read_tokens=base_input)
print(f"\n  Estimated savings per call (after first): "
      f"${per_call_no_cache - per_call_cached:.4f}  "
      f"({(1 - per_call_cached/per_call_no_cache)*100:.0f}% cheaper)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. STREAMING TOKEN COUNTING
# ══════════════════════════════════════════════════════════════════════════════

section("8. STREAMING TOKEN COUNTING")
print("""
  Streaming responses yield text incrementally.
  The final message_delta event contains usage stats for cost tracking.
""")

input_tokens_stream  = 0
output_tokens_stream = 0
full_text            = []

with client.messages.stream(
    model=MODEL,
    max_tokens=150,
    messages=[{"role": "user",
               "content": "List 3 key differences between batch and stream processing."}]
) as stream:
    for event in stream:
        if hasattr(event, "type"):
            if event.type == "message_start":
                input_tokens_stream = event.message.usage.input_tokens
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    full_text.append(event.delta.text)
            elif event.type == "message_delta":
                output_tokens_stream = event.usage.output_tokens

stream_cost = calculate_cost(MODEL, input_tokens_stream, output_tokens_stream)

print(f"\n  Streamed response:")
for line in ("".join(full_text)).splitlines():
    print(f"    {line}")

print(f"\n  Token usage (from stream events):")
print(f"    input_tokens   : {input_tokens_stream:>8,}")
print(f"    output_tokens  : {output_tokens_stream:>8,}")
print(f"    cost           : ${stream_cost:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. BATCH COST ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

section("9. BATCH COST ESTIMATION")
print("""
  Estimate the total cost of a batch job BEFORE running it.
  Use count_tokens() on each prompt, then sum up.
""")

batch_prompts = [
    ("Summarize Apache Spark in one paragraph.",                    300),
    ("Explain Kafka consumer groups.",                              250),
    ("What are the layers of the Medallion Architecture?",          200),
    ("Write a Python function to read Parquet files with PySpark.", 400),
    ("What is the difference between OLTP and OLAP?",              200),
    ("Explain data partitioning strategies in distributed systems.",300),
    ("What is Apache Airflow and what are its core components?",   250),
    ("Describe the CAP theorem.",                                   150),
]

print(f"\n  Estimating cost for {len(batch_prompts)} prompts on {MODEL}:\n")
print(f"  {'#':<4} {'Input Tokens':>13} {'Max Output':>11} {'Est Cost':>10}  {'Prompt'}")
print(f"  {'─'*4} {'─'*13} {'─'*11} {'─'*10}  {'─'*45}")

total_input  = 0
total_output = 0
total_est    = 0.0

for i, (prompt, max_out) in enumerate(batch_prompts, 1):
    count = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    est = calculate_cost(MODEL, count.input_tokens, max_out)
    total_input  += count.input_tokens
    total_output += max_out
    total_est    += est
    print(f"  {i:<4} {count.input_tokens:>13,} {max_out:>11,} ${est:>9.4f}  {prompt[:45]}")

print(f"  {'─'*4} {'─'*13} {'─'*11} {'─'*10}")
print(f"  {'TOT':<4} {total_input:>13,} {total_output:>11,} ${total_est:>9.4f}")
print(f"\n  Estimated batch cost : ${total_est:.4f}")
print(f"  If run daily (30d)   : ${total_est * 30:.4f}/month")


# ══════════════════════════════════════════════════════════════════════════════
# 10. COST DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

section("10. COST DASHBOARD")

class CostDashboard:
    """Aggregates cost metrics across multiple API calls."""

    def __init__(self):
        self.calls:         List[dict] = []
        self.by_model:      dict       = defaultdict(lambda: {"calls":0,"input":0,"output":0,"cost":0.0})
        self.by_purpose:    dict       = defaultdict(lambda: {"calls":0,"cost":0.0})

    def record(self, model: str, input_tok: int, output_tok: int,
               cost: float, purpose: str = "general") -> None:
        self.calls.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model, "input": input_tok,
            "output": output_tok, "cost": cost, "purpose": purpose
        })
        self.by_model[model]["calls"]  += 1
        self.by_model[model]["input"]  += input_tok
        self.by_model[model]["output"] += output_tok
        self.by_model[model]["cost"]   += cost
        self.by_purpose[purpose]["calls"] += 1
        self.by_purpose[purpose]["cost"]  += cost

    def print_dashboard(self) -> None:
        total_cost  = sum(c["cost"]   for c in self.calls)
        total_input = sum(c["input"]  for c in self.calls)
        total_out   = sum(c["output"] for c in self.calls)

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │               LLM COST DASHBOARD                    │")
        print(f"  ├─────────────────────────────────────────────────────┤")
        print(f"  │  Total calls         : {len(self.calls):<28}│")
        print(f"  │  Total input tokens  : {total_input:<28,}│")
        print(f"  │  Total output tokens : {total_out:<28,}│")
        print(f"  │  Total spend         : ${total_cost:<27.4f}│")
        print(f"  └─────────────────────────────────────────────────────┘")

        print(f"\n  By Model:")
        print(f"  {'Model':<28} {'Calls':>7} {'Input':>10} {'Output':>8} {'Cost':>10} {'Avg/Call':>10}")
        print(f"  {'─'*28} {'─'*7} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
        for model, stats in self.by_model.items():
            avg = stats["cost"] / stats["calls"] if stats["calls"] else 0
            print(f"  {model:<28} {stats['calls']:>7} {stats['input']:>10,} "
                  f"{stats['output']:>8,} ${stats['cost']:>9.4f} ${avg:>9.4f}")

        print(f"\n  By Purpose:")
        print(f"  {'Purpose':<20} {'Calls':>7} {'Cost':>10} {'% of Total':>12}")
        print(f"  {'─'*20} {'─'*7} {'─'*10} {'─'*12}")
        for purpose, stats in self.by_purpose.items():
            pct = stats["cost"] / total_cost * 100 if total_cost > 0 else 0
            print(f"  {purpose:<20} {stats['calls']:>7} ${stats['cost']:>9.4f} {pct:>11.1f}%")


# Simulate a session with mixed models and purposes
dashboard = CostDashboard()

session_calls = [
    ("claude-haiku-4-5-20251001", 420,  180, "qa"),
    ("claude-haiku-4-5-20251001", 380,  150, "qa"),
    ("claude-sonnet-4-6",          1800, 600, "analysis"),
    ("claude-sonnet-4-6",          2200, 800, "code_gen"),
    ("claude-opus-4-6",            5000, 1200,"deep_analysis"),
    ("claude-haiku-4-5-20251001", 300,  100, "qa"),
    ("claude-sonnet-4-6",          900,  400, "summarization"),
    ("claude-opus-4-6",            3200, 900, "deep_analysis"),
]

for model, inp, out, purpose in session_calls:
    cost = calculate_cost(model, inp, out)
    dashboard.record(model, inp, out, cost, purpose)

dashboard.print_dashboard()


# ══════════════════════════════════════════════════════════════════════════════
# 11. COST OPTIMIZATION TIPS
# ══════════════════════════════════════════════════════════════════════════════

section("11. COST OPTIMIZATION TIPS")
print("""
  ┌──────────────────────────────────┬───────────────────────────────────────┐
  │ Technique                        │ Savings Potential                     │
  ├──────────────────────────────────┼───────────────────────────────────────┤
  │ Use Haiku for simple tasks       │ 10–19× cheaper than Opus              │
  │ Prompt caching (repeated prefix) │ Up to 90% on cached tokens            │
  │ Reduce max_tokens                │ Limits worst-case output cost         │
  │ Trim unnecessary context         │ Every 1K tokens = ~$0.015 (Opus)      │
  │ Batch count_tokens first         │ Pre-screen expensive prompts          │
  │ Summarize long chat history      │ Compress context before each call     │
  │ Use streaming + early stop       │ Stop generation when answer found     │
  │ Cache responses client-side      │ Re-use identical query responses      │
  │ Select model per task complexity │ Route simple → Haiku, hard → Opus     │
  │ Chunk documents, not full docs   │ Send only relevant passages (RAG)     │
  └──────────────────────────────────┴───────────────────────────────────────┘
""")

# Demonstrate model routing
subsection("Model Routing — match task complexity to model tier")

def route_model(prompt: str) -> str:
    """Simple heuristic routing based on prompt complexity."""
    tokens = len(prompt.split())
    has_code    = any(kw in prompt.lower() for kw in ["write", "code", "function", "implement"])
    has_complex = any(kw in prompt.lower() for kw in ["analyze", "compare", "trade-off", "design"])

    if has_complex or has_code or tokens > 80:
        return "claude-sonnet-4-6"
    elif tokens > 40:
        return "claude-haiku-4-5-20251001"
    else:
        return "claude-haiku-4-5-20251001"

routed_prompts = [
    "What is ETL?",
    "List the top 5 benefits of using a data lakehouse architecture.",
    "Analyze the trade-offs between row-oriented and columnar storage formats for OLAP workloads.",
    "Write a PySpark function to deduplicate records based on a composite key.",
]

print(f"\n  {'Prompt':<58} {'Routed Model'}")
print(f"  {'─'*58} {'─'*28}")
for p in routed_prompts:
    routed = route_model(p)
    print(f"  {p[:56]:<58} {routed}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. COST ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

section("12. COST ALERT SYSTEM")
print("""
  Automatically warn when cumulative spend crosses configurable thresholds.
  Supports INFO / WARNING / CRITICAL alert levels.
""")

class CostAlertSystem:
    """Monitors spend and fires alerts at configurable thresholds."""

    LEVELS = [
        (0.01,  "INFO",     "Spend crossed $0.01"),
        (0.05,  "WARNING",  "Spend crossed $0.05 — approaching limit"),
        (0.10,  "CRITICAL", "Spend crossed $0.10 — review immediately"),
    ]

    def __init__(self):
        self.total_spent = 0.0
        self.fired       = set()
        self.alerts_log  = []

    def record(self, cost: float, call_info: str = "") -> None:
        self.total_spent += cost
        self._check_thresholds(call_info)

    def _check_thresholds(self, call_info: str) -> None:
        for threshold, level, message in self.LEVELS:
            key = f"{level}_{threshold}"
            if self.total_spent >= threshold and key not in self.fired:
                self.fired.add(key)
                alert = {
                    "level":     level,
                    "threshold": threshold,
                    "spent":     self.total_spent,
                    "message":   message,
                    "trigger":   call_info,
                    "ts":        datetime.now().isoformat(timespec="seconds"),
                }
                self.alerts_log.append(alert)
                print(f"\n  🔔 [{level}] {message} | "
                      f"Current spend: ${self.total_spent:.4f}")

    def report(self) -> None:
        print(f"\n  Total spend  : ${self.total_spent:.4f}")
        print(f"  Alerts fired : {len(self.alerts_log)}")
        for a in self.alerts_log:
            print(f"    [{a['level']:8s}] ${a['threshold']:.2f} threshold at "
                  f"${a['spent']:.4f} total — {a['ts']}")


alert_sys = CostAlertSystem()

# Simulate incrementally growing spend
simulated_spend = [0.003, 0.005, 0.004, 0.008, 0.012, 0.015, 0.020, 0.025, 0.030]
print(f"\n  Simulating {len(simulated_spend)} API calls...\n")
for i, spend in enumerate(simulated_spend, 1):
    print(f"  Call {i:02d}: cost ${spend:.3f} | running total ${alert_sys.total_spent + spend:.4f}")
    alert_sys.record(spend, call_info=f"call_{i:02d}")

alert_sys.report()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
  LLM Cost Monitoring Topics Demonstrated:
  ──────────────────────────────────────────────────────────────────
   1.  Token Counting         count_tokens() — estimate cost before calling
   2.  Usage Tracking         response.usage — input/output token capture
   3.  Cost Calculation       USD cost from token counts + pricing table
   4.  Request Logger         APICallRecord dataclass + per-call log table
   5.  Budget Enforcement     Hard-stop with BudgetedClient + pre-flight check
   6.  Model Comparison       Same prompt across Haiku / Sonnet / Opus
   7.  Prompt Caching         cache_control header — up to 90% savings
   8.  Streaming Tokens       Track usage via message_start/message_delta events
   9.  Batch Estimation       Pre-compute total cost before running a job
  10.  Cost Dashboard         Aggregate stats by model and purpose
  11.  Cost Optimization      10 techniques to reduce LLM spend
  12.  Alert System           Threshold-based INFO/WARNING/CRITICAL alerts
  ──────────────────────────────────────────────────────────────────
  Model     : {MODEL}
  Library   : anthropic {anthropic.__version__}

  Pricing quick reference (USD / 1M tokens):
  {'Model':<28} {'Input':>8}  {'Output':>8}  {'CacheRd':>8}
  {'─'*28} {'─'*8}  {'─'*8}  {'─'*8}""")

for m, (inp, out, _, cr) in PRICING.items():
    print(f"  {m:<28} ${inp:>7.2f}  ${out:>7.2f}  ${cr:>7.2f}")

print()
