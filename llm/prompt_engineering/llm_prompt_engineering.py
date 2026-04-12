"""
llm/prompt_engineering/llm_prompt_engineering.py
--------------------------------------------------
Comprehensive Prompt Engineering guide using the Anthropic Claude API.

Techniques demonstrated:
  1.  Zero-Shot Prompting          — direct task with no examples
  2.  Few-Shot Prompting           — provide examples to guide output
  3.  Chain-of-Thought (CoT)       — ask model to reason step by step
  4.  System Prompt Customization  — set model persona and behavior
  5.  Role Prompting               — assign a domain expert role
  6.  Instruction Prompting        — explicit structured instructions
  7.  Output Format Control        — JSON, markdown, bullet points
  8.  Temperature & Sampling       — control creativity vs determinism
  9.  Prompt Chaining              — break complex tasks into steps
  10. Self-Consistency Prompting   — multiple reasoning paths
  11. Negative Prompting           — tell the model what NOT to do
  12. Contextual Prompting         — inject relevant context/documents

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python llm_prompt_engineering.py
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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)

def ask(prompt: str, system: str = None, max_tokens: int = 512,
        temperature: float = 1.0) -> str:
    """Send a message to Claude and return the text response."""
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if temperature != 1.0:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    return response.content[0].text

def print_response(label: str, text: str) -> None:
    print(f"\n  [{label}]")
    for line in text.strip().splitlines():
        print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ZERO-SHOT PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("1. ZERO-SHOT PROMPTING")
print("""
  Definition:
    Ask the model to perform a task with NO examples.
    Relies entirely on the model's pre-trained knowledge.

  Best for:
    • Simple, well-defined tasks
    • Tasks the model commonly encounters in training
    • Quick, direct answers
""")

prompt = "Classify the sentiment of this review as Positive, Negative, or Neutral:\n'The product quality is decent but delivery was extremely slow.'"
response = ask(prompt)
print_response("Prompt", prompt)
print_response("Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEW-SHOT PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("2. FEW-SHOT PROMPTING")
print("""
  Definition:
    Provide 2–5 input/output examples before the actual task.
    The model infers the pattern and applies it.

  Best for:
    • Custom classification labels
    • Domain-specific formatting
    • Tasks where the desired output style is hard to describe
""")

prompt = """Classify each customer message into one of: [Billing, Technical, General]

Message: "I was charged twice for my subscription."
Category: Billing

Message: "The app crashes every time I open it."
Category: Technical

Message: "What are your business hours?"
Category: General

Message: "My internet connection keeps dropping after your update."
Category:"""

response = ask(prompt, max_tokens=50)
print_response("Prompt (with examples)", prompt)
print_response("Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHAIN-OF-THOUGHT (CoT) PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("3. CHAIN-OF-THOUGHT (CoT) PROMPTING")
print("""
  Definition:
    Instruct the model to show its reasoning steps before the final answer.
    Dramatically improves accuracy on math, logic, and multi-step problems.

  Variants:
    • "Think step by step" — zero-shot CoT
    • Provide worked examples with reasoning — few-shot CoT
    • "Let's think carefully" — encourages deliberate reasoning
""")

prompt = """A data pipeline processes 1.2 million records per hour.
Due to a schema change, processing speed drops by 35%.
After optimization, speed recovers by 20% of the original speed.
What is the final processing rate in records per hour?

Think step by step before giving the final answer."""

response = ask(prompt, max_tokens=300)
print_response("Prompt", prompt)
print_response("Response (step-by-step reasoning)", response)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SYSTEM PROMPT CUSTOMIZATION
# ══════════════════════════════════════════════════════════════════════════════

section("4. SYSTEM PROMPT CUSTOMIZATION")
print("""
  Definition:
    The system prompt sets the model's persistent persona, tone,
    rules, and constraints for the entire conversation.

  Best practices:
    • Define the role clearly at the start
    • Specify output format and length expectations
    • Set constraints (e.g., "always respond in JSON")
    • Use it to inject domain knowledge or company context
""")

system = """You are a Senior Data Engineer at a Fortune 500 company.
You specialize in Apache Spark, cloud data platforms, and pipeline architecture.
Always respond in a concise, technical manner suitable for an engineering audience.
Format responses with clear sections using bullet points."""

prompt = "What are the key considerations when designing a real-time data ingestion pipeline?"

response = ask(prompt, system=system, max_tokens=400)
print_response("System Prompt", system)
print_response("User Prompt", prompt)
print_response("Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROLE PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("5. ROLE PROMPTING")
print("""
  Definition:
    Assign a specific expert persona directly in the user prompt.
    Shapes the depth, vocabulary, and perspective of the response.

  Tip:
    Combine with system prompts for strongest effect.
    Use specific roles: "Principal ML Engineer at Google" beats "expert".
""")

prompt = """Act as a database architect with 15 years of experience in
data warehousing. Explain the differences between Star Schema and
Snowflake Schema, and when you would choose one over the other."""

response = ask(prompt, max_tokens=400)
print_response("Prompt", prompt)
print_response("Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# 6. INSTRUCTION PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("6. INSTRUCTION PROMPTING")
print("""
  Definition:
    Provide explicit, structured instructions about HOW to complete the task.
    Use numbered steps, constraints, and output specifications.

  Best for:
    • Complex tasks requiring specific structure
    • Ensuring consistency across multiple runs
    • Production prompts where reliability matters
""")

prompt = """Analyze the following Spark job log and provide a structured diagnosis.

Follow these instructions exactly:
1. Identify the root cause of the failure (1 sentence)
2. List the contributing factors as bullet points (max 3)
3. Provide a recommended fix (2-3 sentences)
4. Rate the severity: Critical / High / Medium / Low

Log excerpt:
ERROR SparkContext: Error initializing SparkContext
java.lang.OutOfMemoryError: GC overhead limit exceeded
  at org.apache.spark.executor.Executor.run(Executor.scala:478)
Caused by: Shuffle spill (memory) detected: 2.8 GB spilled to disk
Task stage 45 failed 4 times; most recent failure: lost task 12.3"""

response = ask(prompt, max_tokens=400)
print_response("Prompt", prompt)
print_response("Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# 7. OUTPUT FORMAT CONTROL
# ══════════════════════════════════════════════════════════════════════════════

section("7. OUTPUT FORMAT CONTROL")

# ── JSON output ───────────────────────────────────────────────────────────────
print("\n  7a. JSON Output")
prompt_json = """Extract the following information from the job description and return ONLY valid JSON.
No explanation, no markdown, just the raw JSON object.

Schema: {"job_title": str, "skills": [str], "experience_years": int, "location": str}

Job Description:
"We are looking for a Senior Data Engineer with 5+ years of experience in
Apache Spark, Python, and AWS. The role is based in Seattle, WA and requires
expertise in building scalable ETL pipelines and data lakes." """

response_json = ask(prompt_json, max_tokens=200)
print_response("Prompt", prompt_json)
print_response("Raw Response", response_json)

try:
    parsed = json.loads(response_json)
    print(f"\n  [Parsed JSON]")
    print(f"    job_title        : {parsed.get('job_title')}")
    print(f"    skills           : {parsed.get('skills')}")
    print(f"    experience_years : {parsed.get('experience_years')}")
    print(f"    location         : {parsed.get('location')}")
except json.JSONDecodeError:
    print("    (response was not pure JSON — add stricter instructions if needed)")

# ── Markdown table output ──────────────────────────────────────────────────────
print("\n  7b. Markdown Table Output")
prompt_table = """Compare Apache Kafka, Apache Pulsar, and AWS Kinesis across these dimensions:
Throughput, Latency, Retention, Ordering Guarantee, and Managed Service availability.

Format the response as a markdown table only. No additional text."""

response_table = ask(prompt_table, max_tokens=300)
print_response("Prompt", prompt_table)
print_response("Response (Markdown Table)", response_table)


# ══════════════════════════════════════════════════════════════════════════════
# 8. TEMPERATURE & SAMPLING CONTROL
# ══════════════════════════════════════════════════════════════════════════════

section("8. TEMPERATURE & SAMPLING CONTROL")
print("""
  Temperature controls randomness / creativity of output:

    temperature = 0.0   → Deterministic, focused, factual (same answer every run)
    temperature = 0.3   → Slightly creative, consistent
    temperature = 0.7   → Balanced creativity and coherence  (default)
    temperature = 1.0   → More varied, creative responses
    temperature = 1.0+  → Very creative, may be less coherent

  Use cases:
    Low temp  (0.0–0.3) : SQL generation, classification, data extraction
    Mid temp  (0.5–0.7) : Summarization, Q&A, analysis
    High temp (0.8–1.0) : Brainstorming, creative writing, ideation
""")

task = "List 3 creative names for a real-time data streaming platform."

print("  Running same prompt at 3 different temperatures:\n")
for temp in [0.0, 0.5, 1.0]:
    resp = ask(task, max_tokens=100, temperature=temp)
    print(f"  [temperature={temp}]")
    for line in resp.strip().splitlines():
        print(f"    {line}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 9. PROMPT CHAINING
# ══════════════════════════════════════════════════════════════════════════════

section("9. PROMPT CHAINING")
print("""
  Definition:
    Break a complex task into sequential prompts where the output
    of one step becomes the input to the next.

  Benefits:
    • Easier to debug (isolate which step failed)
    • Each step can be optimized independently
    • Reduces hallucination on complex tasks
    • Allows branching/conditional logic between steps
""")

raw_data = """
date,store,product,units_sold,revenue
2026-01-01,NYC,Laptop,12,14400
2026-01-01,NYC,Phone,45,31500
2026-01-02,LA,Laptop,8,9600
2026-01-02,LA,Tablet,20,10000
2026-01-03,NYC,Phone,60,42000
2026-01-03,Chicago,Laptop,5,6000
"""

# Step 1 — Extract insights
step1_prompt = f"""Analyze this sales CSV data and extract the top 3 key insights.
Be specific with numbers. Output as a numbered list.

{raw_data}"""

step1_result = ask(step1_prompt, max_tokens=200)
print_response("Step 1 — Extract Insights", step1_result)

# Step 2 — Generate recommendations based on step 1 output
step2_prompt = f"""Based on these sales insights:
{step1_result}

Generate 3 specific, actionable business recommendations for the sales team.
Format as: Recommendation | Expected Impact"""

step2_result = ask(step2_prompt, max_tokens=250)
print_response("Step 2 — Generate Recommendations", step2_result)

# Step 3 — Summarize into executive brief
step3_prompt = f"""Create a 3-sentence executive summary combining these insights and recommendations:

Insights:
{step1_result}

Recommendations:
{step2_result}"""

step3_result = ask(step3_prompt, max_tokens=150)
print_response("Step 3 — Executive Summary", step3_result)


# ══════════════════════════════════════════════════════════════════════════════
# 10. SELF-CONSISTENCY PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("10. SELF-CONSISTENCY PROMPTING")
print("""
  Definition:
    Sample multiple independent reasoning paths for the same problem
    and aggregate (majority vote) the final answers.
    Improves reliability on reasoning-heavy tasks.

  When to use:
    • High-stakes decisions (architecture choices, root cause analysis)
    • Math and logic problems
    • When a single response may be unreliable
""")

question = """A Spark job reads 500 GB of data, applies a filter that keeps 40% of records,
then joins with a 2 GB lookup table, and finally aggregates by 10 unique keys.
Estimate the approximate shuffle data volume in GB. Show your reasoning."""

print("  Sampling 3 independent reasoning paths:\n")
answers = []
for i in range(3):
    resp = ask(question, max_tokens=200, temperature=0.7)
    print(f"  [Path {i+1}]")
    for line in resp.strip().splitlines():
        print(f"    {line}")
    print()
    answers.append(resp)

# Aggregate using a final consolidation prompt
consolidate_prompt = f"""Three independent analyses estimated Spark shuffle volume for the same problem.
Synthesize them and provide the most reliable final answer with brief justification.

Analysis 1: {answers[0]}
Analysis 2: {answers[1]}
Analysis 3: {answers[2]}"""

final = ask(consolidate_prompt, max_tokens=150)
print_response("Consolidated Answer (majority reasoning)", final)


# ══════════════════════════════════════════════════════════════════════════════
# 11. NEGATIVE PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("11. NEGATIVE PROMPTING")
print("""
  Definition:
    Explicitly tell the model what NOT to do, include, or assume.
    Prevents common failure modes like verbosity, hallucination,
    irrelevant caveats, or undesired formatting.

  Common negatives:
    • "Do not add disclaimers or caveats"
    • "Do not repeat the question"
    • "Do not use bullet points"
    • "Do not make up data — if unknown, say 'unknown'"
    • "Do not explain what you are about to do"
""")

# Without negative constraints
prompt_without = "Explain what a data lakehouse is."
resp_without = ask(prompt_without, max_tokens=200)
print_response("Without negative prompting", resp_without)

# With negative constraints
prompt_with = """Explain what a data lakehouse is.
Do NOT repeat the question. Do NOT add introductory phrases like 'Certainly' or 'Sure'.
Do NOT use bullet points. Keep it under 4 sentences. Be direct and technical."""

resp_with = ask(prompt_with, max_tokens=200)
print_response("With negative prompting (constrained)", resp_with)


# ══════════════════════════════════════════════════════════════════════════════
# 12. CONTEXTUAL PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

section("12. CONTEXTUAL PROMPTING")
print("""
  Definition:
    Inject relevant background context, documents, or data directly
    into the prompt so the model answers based on YOUR information,
    not just its training data.

  Use cases:
    • Document Q&A (inject the document)
    • Code review (inject the code)
    • Incident analysis (inject the log/metrics)
    • Personalized responses (inject user profile)

  This is the foundation of RAG (Retrieval-Augmented Generation).
""")

context_doc = """
PIPELINE INCIDENT REPORT — 2026-04-10
Pipeline: customer_sync
Status: DEGRADED
Symptoms:
  - Throughput dropped from 4,200 rows/sec to 820 rows/sec at 14:32 UTC
  - Average latency increased from 850ms to 4,200ms
  - Error rate spiked to 12.4% (from baseline 1.5%)
  - 3 executor nodes reported GC pause > 10 seconds
  - Shuffle spill detected: 18 GB to disk across 6 partitions
Root Cause: Upstream schema change added 4 new nullable columns; downstream
  consumers were not updated, causing deserialization failures and retry storms.
Resolution: Schema registry update deployed at 15:10 UTC. Throughput recovering.
"""

prompt = f"""Based on the incident report below, answer these questions:
1. What was the primary symptom that indicated degradation?
2. What caused the GC pauses?
3. What was the root cause of the incident?
4. What should be done to prevent this in the future?

Incident Report:
{context_doc}"""

response = ask(prompt, max_tokens=400)
print_response("Context (Incident Report)", context_doc.strip())
print_response("Questions + Response", response)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING BEST PRACTICES SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("PROMPT ENGINEERING BEST PRACTICES")
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │              PROMPT ENGINEERING CHEAT SHEET                     │
  ├──────────────────────────┬──────────────────────────────────────┤
  │ Technique                │ When to Use                          │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ Zero-Shot                │ Simple, well-known tasks             │
  │ Few-Shot                 │ Custom format / domain labels        │
  │ Chain-of-Thought         │ Math, logic, multi-step reasoning    │
  │ System Prompt            │ Persistent persona, rules, format    │
  │ Role Prompting           │ Domain expertise needed              │
  │ Instruction Prompting    │ Complex structured output required   │
  │ Output Format Control    │ JSON, tables, code, structured data  │
  │ Temperature Control      │ Balance creativity vs determinism    │
  │ Prompt Chaining          │ Complex multi-step workflows         │
  │ Self-Consistency         │ High-stakes or ambiguous problems    │
  │ Negative Prompting       │ Prevent unwanted model behaviors     │
  │ Contextual Prompting     │ Ground answers in your own data      │
  └──────────────────────────┴──────────────────────────────────────┘

  Golden Rules:
    1. Be specific — vague prompts produce vague answers
    2. Show don't just tell — examples outperform instructions alone
    3. One task per prompt — complex tasks → prompt chaining
    4. Control the format — specify JSON/markdown/bullets explicitly
    5. Set temperature deliberately — 0 for extraction, 0.7 for analysis
    6. Use system prompts for persistent behavior, user prompts for tasks
    7. Iterate — prompt engineering is empirical; test and refine
""")

print("  Model :", MODEL)
print("  All 12 prompt engineering techniques demonstrated successfully.")
print("═" * 64)
