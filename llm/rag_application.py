"""
llm/rag_application.py
-----------------------
End-to-End RAG (Retrieval-Augmented Generation) Application.

This application demonstrates a complete RAG pipeline for a
Data Engineering Knowledge Base assistant.

Pipeline stages:
  Stage 1 — Document Loading     : Load raw documents with metadata
  Stage 2 — Text Chunking         : Split documents into smaller passages
  Stage 3 — Embedding             : Convert chunks to dense vectors
  Stage 4 — Indexing              : Store vectors in FAISS index
  Stage 5 — Retrieval             : Find top-K relevant chunks for a query
  Stage 6 — Augmented Generation  : Inject context into LLM prompt
  Stage 7 — Conversational RAG    : Multi-turn Q&A with memory
  Stage 8 — Source Citation       : Show which documents were used

Setup:
    pip install anthropic sentence-transformers faiss-cpu numpy
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python rag_application.py
"""

import os
import re
import json
import numpy as np
import faiss
from dataclasses             import dataclass, field
from typing                  import List, Optional
from sentence_transformers   import SentenceTransformer
import anthropic


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LLM_MODEL    = "claude-opus-4-6"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 150       # words per chunk
CHUNK_OVERLAP = 30       # overlapping words between chunks
TOP_K        = 4         # retrieved chunks per query
MIN_SCORE    = 0.30      # minimum similarity score threshold


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def print_answer(answer: str) -> None:
    print()
    for line in answer.strip().splitlines():
        print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """A source document with metadata."""
    doc_id:   str
    title:    str
    category: str
    content:  str
    source:   str = "knowledge_base"

@dataclass
class Chunk:
    """A text chunk derived from a document."""
    chunk_id:  str
    doc_id:    str
    title:     str
    category:  str
    text:      str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 1 — DOCUMENT LOADING")

DOCUMENTS = [
    Document(
        doc_id="DE_001", title="Apache Spark Architecture",
        category="data_engineering",
        content="""Apache Spark is an open-source distributed computing system designed
for large-scale data processing. Spark provides a unified analytics engine for
big data workloads with built-in modules for SQL, streaming, machine learning,
and graph processing.

Spark's core abstraction is the Resilient Distributed Dataset (RDD), which is
an immutable distributed collection of objects. RDDs support two types of operations:
transformations (which create a new dataset from an existing one) and actions
(which return a value to the driver program after running a computation on the dataset).

Spark uses a master-worker architecture. The driver program runs on the master node
and coordinates execution. Executors run on worker nodes and perform actual computation.
The Cluster Manager (YARN, Mesos, Kubernetes, or Spark Standalone) allocates resources.

Key Spark components include: Spark Core, Spark SQL, Spark Streaming, MLlib, and GraphX.
Spark SQL provides a programming interface for structured data using DataFrames and SQL.
Spark Streaming processes live data streams using micro-batches.
MLlib provides scalable machine learning algorithms.
GraphX provides graph computation capabilities."""
    ),
    Document(
        doc_id="DE_002", title="Apache Kafka Fundamentals",
        category="data_engineering",
        content="""Apache Kafka is a distributed event streaming platform used for
high-performance data pipelines, streaming analytics, and data integration.
Kafka was originally developed at LinkedIn and open-sourced in 2011.

Kafka's architecture consists of Producers, Brokers, Topics, Partitions, and Consumers.
Producers publish messages to topics. Brokers store messages in partitioned, replicated logs.
Consumers subscribe to topics and process messages. Consumer Groups allow parallel consumption.

Topics in Kafka are divided into partitions for scalability. Each partition is an ordered,
immutable sequence of messages identified by an offset. Kafka guarantees message ordering
within a partition but not across partitions.

Kafka uses ZooKeeper (or KRaft in newer versions) for cluster coordination and leader election.
Replication factor determines how many copies of each partition are maintained across brokers
for fault tolerance. A replication factor of 3 means each partition has 3 replicas.

Key Kafka use cases include: log aggregation, real-time analytics, event sourcing,
stream processing, messaging, metrics collection, and microservices communication.
Kafka Connect integrates Kafka with external systems. Kafka Streams provides a client library
for building stream processing applications."""
    ),
    Document(
        doc_id="DE_003", title="Data Warehouse Design",
        category="architecture",
        content="""A data warehouse is a centralized repository that integrates data from
multiple sources for reporting, analytics, and business intelligence. Unlike OLTP systems
optimized for transactions, data warehouses are optimized for analytical queries (OLAP).

The most common design patterns are Star Schema and Snowflake Schema. In a Star Schema,
a central Fact Table contains measurable business metrics (sales amount, order quantity)
and is surrounded by Dimension Tables (customer, product, time, location) containing
descriptive attributes. Star Schema queries are faster due to fewer joins.

Snowflake Schema normalizes dimension tables into multiple related tables, reducing
redundancy but increasing join complexity. Snowflake Schema is preferred when storage
efficiency matters more than query performance.

The Medallion Architecture organizes data in three layers:
Bronze Layer — raw ingested data, no transformation, preserves source fidelity
Silver Layer — cleaned, validated, and enriched data, deduplicated and standardized
Gold Layer  — aggregated, business-ready data for dashboards and reporting

Modern cloud data warehouses include Snowflake, Google BigQuery, Amazon Redshift,
and Azure Synapse Analytics. These platforms separate storage from compute,
enabling independent scaling and pay-per-query pricing models."""
    ),
    Document(
        doc_id="DE_004", title="Apache Airflow Pipeline Orchestration",
        category="orchestration",
        content="""Apache Airflow is an open-source platform for authoring, scheduling,
and monitoring data pipelines. Pipelines are defined as Directed Acyclic Graphs (DAGs)
in Python, making them version-controlled and testable.

An Airflow DAG defines the workflow logic: which tasks to run, in what order,
and under what conditions. Tasks represent individual units of work. Operators
are templates for tasks: PythonOperator runs Python functions, BashOperator runs
shell commands, and provider operators integrate with external services.

Airflow uses a Scheduler to trigger DAG runs based on a schedule (cron expression
or timedelta). The Executor determines how tasks are executed: SequentialExecutor runs
one task at a time, LocalExecutor runs tasks in parallel on the same machine, and
CeleryExecutor/KubernetesExecutor distribute tasks across multiple workers.

Key Airflow concepts include: Task Dependencies (set_upstream, set_downstream, >>),
XComs for inter-task data sharing, Sensors for waiting on external conditions,
Hooks for connecting to external systems, Variables for configuration, and
Connections for storing credentials.

Best practices for Airflow DAGs: keep tasks atomic and idempotent, use SLAs to
detect slow tasks, avoid heavy computation in the scheduler, use pools to limit
concurrency, and parameterize DAGs with Jinja templating."""
    ),
    Document(
        doc_id="DE_005", title="Delta Lake and Data Lakehouse",
        category="architecture",
        content="""Delta Lake is an open-source storage layer that brings ACID transaction
support to Apache Spark and big data workloads. Delta Lake runs on top of existing
cloud object storage (S3, ADLS, GCS) and adds reliability features missing from
raw data lakes.

Key Delta Lake features include:
ACID Transactions — atomic commits prevent partial writes and data corruption
Schema Enforcement — rejects writes that don't match the table schema
Schema Evolution — allows adding new columns without rewriting existing data
Time Travel — query previous versions of data using VERSION AS OF or TIMESTAMP AS OF
Unified Batch and Streaming — same table can be used for both batch writes and streaming reads
Z-Ordering — colocation of related data for faster queries on high-cardinality columns
OPTIMIZE — compacts small files into larger ones to improve read performance

A Data Lakehouse combines the low-cost storage of a data lake with the data management
and performance of a data warehouse. Delta Lake, Apache Iceberg, and Apache Hudi are
the three main open table formats enabling the Lakehouse architecture.

Databricks built Delta Lake and offers Databricks Lakehouse Platform as a managed solution.
Key Lakehouse use cases include: unified analytics, ML feature stores, streaming analytics,
and replacing legacy ETL pipelines with more flexible ELT patterns."""
    ),
    Document(
        doc_id="DE_006", title="Data Quality and Validation",
        category="data_engineering",
        content="""Data quality is a critical concern in data engineering. Poor data quality
leads to incorrect analytics, failed ML models, and bad business decisions.
The four key dimensions of data quality are:

Completeness — are all required fields populated? Measured as percentage of non-null values.
Accuracy — does the data correctly represent the real-world entity? Requires domain validation.
Consistency — is the data consistent across systems and time? Check for referential integrity.
Timeliness — is the data available when needed? Monitor pipeline SLAs and data freshness.

Data validation techniques include:
Schema validation — enforce data types, nullability, and column names
Range checks — numeric values within expected min/max bounds
Uniqueness checks — primary keys and business identifiers have no duplicates
Referential integrity — foreign keys match existing dimension table records
Statistical checks — row counts, null rates, and distributions within expected ranges
Custom business rules — domain-specific constraints (e.g., sale date <= ship date)

Tools for data quality include Great Expectations, dbt tests, Soda Core, and Apache Griffin.
Great Expectations allows defining expectations as code and generates data documentation.
dbt tests run validation checks as part of the transformation pipeline.

Data quality monitoring should include alerting when quality scores drop below thresholds,
lineage tracking to identify the source of quality issues, and automated remediation
workflows to quarantine bad records."""
    ),
]

print(f"\n  Loaded {len(DOCUMENTS)} documents into knowledge base:\n")
print(f"  {'Doc ID':<10} {'Category':<18} {'Title':<40} {'Words':>6}")
print(f"  {'─'*10} {'─'*18} {'─'*40} {'─'*6}")
for doc in DOCUMENTS:
    words = len(doc.content.split())
    print(f"  {doc.doc_id:<10} {doc.category:<18} {doc.title:<40} {words:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TEXT CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 2 — TEXT CHUNKING")
print(f"""
  Strategy : Sliding window with overlap
  Chunk size   : {CHUNK_SIZE} words
  Overlap      : {CHUNK_OVERLAP} words

  Why chunk?
    • LLMs have context limits — can't fit entire documents
    • Smaller chunks improve retrieval precision
    • Overlap preserves context at chunk boundaries
""")

def chunk_document(doc: Document) -> List[Chunk]:
    """Split document into overlapping word-based chunks."""
    words  = doc.content.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        text        = " ".join(chunk_words)
        chunk_id    = f"{doc.doc_id}_C{len(chunks)+1:02d}"
        chunks.append(Chunk(
            chunk_id=chunk_id, doc_id=doc.doc_id,
            title=doc.title, category=doc.category, text=text
        ))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

ALL_CHUNKS: List[Chunk] = []
for doc in DOCUMENTS:
    doc_chunks = chunk_document(doc)
    ALL_CHUNKS.extend(doc_chunks)

print(f"  {'Doc ID':<10} {'Title':<38} {'Chunks':>7} {'Avg Words':>10}")
print(f"  {'─'*10} {'─'*38} {'─'*7} {'─'*10}")
for doc in DOCUMENTS:
    doc_chunks = [c for c in ALL_CHUNKS if c.doc_id == doc.doc_id]
    avg_words  = sum(len(c.text.split()) for c in doc_chunks) / len(doc_chunks)
    print(f"  {doc.doc_id:<10} {doc.title:<38} {len(doc_chunks):>7} {avg_words:>10.0f}")

print(f"\n  Total chunks : {len(ALL_CHUNKS)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 3 — EMBEDDING")
print(f"\n  Loading embedding model: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)
DIM      = embedder.get_sentence_embedding_dimension()
print(f"  Embedding dimension : {DIM}")

texts      = [c.text for c in ALL_CHUNKS]
embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

for chunk, emb in zip(ALL_CHUNKS, embeddings):
    chunk.embedding = emb

EMBED_MATRIX = np.vstack([c.embedding for c in ALL_CHUNKS]).astype("float32")
print(f"  Encoded {len(ALL_CHUNKS)} chunks → matrix {EMBED_MATRIX.shape}")
print(f"  Memory  : {EMBED_MATRIX.nbytes / 1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FAISS INDEXING
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 4 — FAISS INDEXING")

index = faiss.IndexFlatIP(DIM)
index.add(EMBED_MATRIX)

print(f"\n  Index type       : IndexFlatIP (exact cosine similarity)")
print(f"  Vectors indexed  : {index.ntotal}")
print(f"  Dimension        : {DIM}")
print(f"  Is trained       : {index.is_trained}")
print(f"""
  Why IndexFlatIP?
    • Exact nearest neighbour — 100% recall
    • Simple for a knowledge base of < 10K chunks
    • For larger corpora use IndexHNSWFlat or IndexIVFFlat
""")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 5 — RETRIEVAL")

def retrieve(query: str, top_k: int = TOP_K,
             category: Optional[str] = None) -> List[tuple]:
    """
    Retrieve the most relevant chunks for a query.
    Returns list of (score, chunk) tuples sorted by relevance.
    """
    q_emb = embedder.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")

    if category:
        # Filter to category before search
        filtered   = [c for c in ALL_CHUNKS if c.category == category]
        filt_matrix = np.vstack([c.embedding for c in filtered]).astype("float32")
        filt_index  = faiss.IndexFlatIP(DIM)
        filt_index.add(filt_matrix)
        scores, indices = filt_index.search(q_emb, k=min(top_k, len(filtered)))
        return [(float(scores[0][i]), filtered[indices[0][i]])
                for i in range(len(indices[0]))
                if float(scores[0][i]) >= MIN_SCORE]
    else:
        scores, indices = index.search(q_emb, k=top_k)
        return [(float(scores[0][i]), ALL_CHUNKS[indices[0][i]])
                for i in range(top_k)
                if float(scores[0][i]) >= MIN_SCORE]

# Demo retrieval
test_queries = [
    "How does Spark distribute data across workers?",
    "What are Kafka partitions and consumer groups?",
    "What are the layers in the Medallion Architecture?",
]

for query in test_queries:
    results = retrieve(query, top_k=3)
    print(f"\n  Query : \"{query}\"")
    print(f"  {'Rank':<5} {'Score':>7}  {'Chunk ID':<14} {'Source Document':<35}  {'Preview'}")
    print(f"  {'─'*5} {'─'*7}  {'─'*14} {'─'*35}  {'─'*30}")
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"  {rank:<5} {score:>7.4f}  {chunk.chunk_id:<14} {chunk.title:<35}  {chunk.text[:30]}...")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — AUGMENTED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 6 — AUGMENTED GENERATION")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )
llm_client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are a Data Engineering Knowledge Base Assistant.
Answer questions using ONLY the provided context passages.
Rules:
  - If the answer is in the context, answer clearly and concisely.
  - Always cite which source documents you used (mention the title).
  - If the context does not contain enough information, say "I don't have enough information in the knowledge base to answer this."
  - Do not make up information that is not in the context.
  - Format responses with clear structure when answering multi-part questions."""

def build_context(results: List[tuple]) -> str:
    """Build a formatted context string from retrieved chunks."""
    parts = []
    for rank, (score, chunk) in enumerate(results, 1):
        parts.append(
            f"[Source {rank} | {chunk.title} | {chunk.chunk_id}]\n{chunk.text}"
        )
    return "\n\n".join(parts)

def rag_query(question: str, top_k: int = TOP_K,
              category: Optional[str] = None) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks
      2. Build augmented prompt
      3. Generate grounded answer
      4. Return answer + sources used
    """
    # Step 1 — Retrieve
    results = retrieve(question, top_k=top_k, category=category)

    if not results:
        return {
            "answer":  "No relevant documents found in the knowledge base.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 2 — Build context
    context = build_context(results)

    # Step 3 — Generate
    prompt = f"""Context from Knowledge Base:
{context}

Question: {question}

Answer using only the context above. Cite source titles."""

    response = llm_client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    answer  = response.content[0].text
    sources = list({chunk.title for _, chunk in results})

    return {
        "answer":       answer,
        "sources":      sources,
        "chunks_used":  len(results),
        "input_tokens": response.usage.input_tokens,
        "output_tokens":response.usage.output_tokens,
    }


# Run RAG queries
rag_questions = [
    "What is Apache Spark and what are its key components?",
    "How does Kafka ensure fault tolerance with replication?",
    "What is the difference between Star Schema and Snowflake Schema?",
    "What is Delta Lake and what problems does it solve?",
    "What are the four dimensions of data quality?",
]

for q in rag_questions:
    result = rag_query(q)
    subsection(f"Q: {q}")
    print_answer(result["answer"])
    print(f"\n    Sources used    : {result['sources']}")
    print(f"    Chunks retrieved: {result['chunks_used']}")
    print(f"    Tokens (in/out) : {result['input_tokens']} / {result['output_tokens']}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — CONVERSATIONAL RAG (Multi-Turn Q&A)
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 7 — CONVERSATIONAL RAG (Multi-Turn Q&A)")
print("""
  Conversational RAG maintains chat history so follow-up questions
  can reference previous answers naturally.

  Pattern:
    1. Keep a running message history
    2. For each new question, retrieve fresh context
    3. Inject context + history into the LLM prompt
    4. Append question and answer to history for next turn
""")

def conversational_rag(turns: List[str]) -> None:
    """Run a multi-turn RAG conversation."""
    history  = []

    for i, question in enumerate(turns, 1):
        print(f"\n  Turn {i} ─────────────────────────────────────────────────")
        print(f"  User : {question}")

        # Build history string
        history_str = ""
        if history:
            history_str = "Previous conversation:\n"
            for user_q, asst_a in history:
                history_str += f"  User     : {user_q}\n"
                history_str += f"  Assistant: {asst_a[:120]}...\n\n"

        # Retrieve context for current question
        results = retrieve(question, top_k=3)
        context = build_context(results) if results else "No relevant context found."

        prompt = f"""{history_str}
Context from Knowledge Base:
{context}

Current question: {question}

Answer concisely using only the context. Reference prior answers when relevant."""

        response = llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=350,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text
        history.append((question, answer))

        print(f"\n  Assistant:")
        for line in answer.strip().splitlines():
            print(f"    {line}")

conversational_rag([
    "What is Apache Airflow and how does it schedule pipelines?",
    "What operators does it provide for different types of tasks?",
    "How should I structure my DAGs for best practices?",
])


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SOURCE CITATION & TRANSPARENCY
# ══════════════════════════════════════════════════════════════════════════════

section("STAGE 8 — SOURCE CITATION & TRANSPARENCY")
print("""
  A trustworthy RAG application always shows which documents were used
  to generate the answer, with relevance scores for transparency.
""")

def rag_with_citations(question: str) -> None:
    """RAG query with full source citation display."""
    results = retrieve(question, top_k=4)

    print(f"\n  Question : {question}")
    print(f"\n  Retrieved Sources:")
    print(f"  {'Rank':<5} {'Score':>7}  {'Chunk ID':<14} {'Document Title':<38} {'Category'}")
    print(f"  {'─'*5} {'─'*7}  {'─'*14} {'─'*38} {'─'*16}")
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"  {rank:<5} {score:>7.4f}  {chunk.chunk_id:<14} {chunk.title:<38} {chunk.category}")

    context = build_context(results)
    prompt  = f"""Context:
{context}

Question: {question}

Provide a structured answer. After your answer, add a "Sources:" section
listing the exact document titles you used."""

    response = llm_client.messages.create(
        model=LLM_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\n  Answer:")
    print_answer(response.content[0].text)

rag_with_citations(
    "How does Spark Streaming work and how does it differ from Kafka Streams?"
)


# ══════════════════════════════════════════════════════════════════════════════
# RAG APPLICATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("RAG APPLICATION SUMMARY")
print(f"""
  RAG Pipeline Stages Completed:
  ──────────────────────────────────────────────────────────────────
  Stage 1  Document Loading      {len(DOCUMENTS)} documents loaded with metadata
  Stage 2  Text Chunking         {len(ALL_CHUNKS)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})
  Stage 3  Embedding             {EMBED_MODEL} → {DIM}-dim vectors
  Stage 4  FAISS Indexing        IndexFlatIP, {index.ntotal} vectors indexed
  Stage 5  Retrieval             Cosine similarity, top-{TOP_K}, threshold={MIN_SCORE}
  Stage 6  Augmented Generation  {LLM_MODEL} with grounded prompts
  Stage 7  Conversational RAG    Multi-turn Q&A with chat history
  Stage 8  Source Citation       Chunk IDs, scores, and document titles
  ──────────────────────────────────────────────────────────────────

  Key design decisions:
    ✦ Chunk overlap ({CHUNK_OVERLAP} words) preserves context at boundaries
    ✦ Normalized embeddings → cosine similarity via dot product
    ✦ Minimum score threshold ({MIN_SCORE}) filters irrelevant results
    ✦ System prompt enforces grounding — no hallucination
    ✦ Source citations build user trust and enable debugging

  Scaling this application:
    Small  (< 10K chunks)   → IndexFlatIP (exact, current setup)
    Medium (10K – 1M)       → IndexHNSWFlat (fast ANN)
    Large  (1M+)            → IndexIVFPQ (memory-efficient ANN)
    Production              → Pinecone / Weaviate / pgvector (managed)

  LLM      : {LLM_MODEL}
  Embedder : {EMBED_MODEL}  ({DIM}-dim)
  Library  : anthropic · sentence-transformers · faiss-cpu · numpy
""")
