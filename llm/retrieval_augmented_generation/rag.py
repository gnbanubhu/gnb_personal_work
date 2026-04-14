"""
llm/rag/rag.py
---------------
Comprehensive Retrieval-Augmented Generation (RAG) pipeline using
sentence-transformers for retrieval and Anthropic Claude for generation.

Topics covered:
  1.  What is RAG                  — architecture overview
  2.  Document Ingestion           — load and chunk documents
  3.  Embedding & Indexing         — encode chunks into vector store
  4.  Retrieval                    — find top-K relevant chunks
  5.  Augmented Generation         — inject context into LLM prompt
  6.  End-to-End RAG Pipeline      — ingestion → retrieval → generation
  7.  Chunking Strategies          — fixed-size, sentence, overlap
  8.  Metadata Filtering           — filter by source, date, category
  9.  Multi-Document RAG           — query across multiple documents
  10. Conversational RAG           — multi-turn Q&A with memory
  11. RAG Evaluation               — faithfulness, relevance, context recall
  12. RAG vs Fine-Tuning           — when to use each approach

Libraries: sentence-transformers · anthropic · numpy · scikit-learn

Setup:
    pip install sentence-transformers anthropic numpy scikit-learn
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python rag.py
"""

import os
import re
import json
import numpy as np
import anthropic
from dataclasses                         import dataclass, field
from typing                              import List, Optional
from sentence_transformers               import SentenceTransformer, util
from sklearn.metrics.pairwise            import cosine_similarity

# ── Clients ───────────────────────────────────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )

client     = anthropic.Anthropic(api_key=api_key)
LLM_MODEL  = "claude-opus-4-6"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def ask_llm(prompt: str, system: str = None, max_tokens: int = 512) -> str:
    """Send a prompt to Claude and return the text response."""
    kwargs = dict(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs).content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """Represents a source document with metadata."""
    doc_id:   str
    title:    str
    content:  str
    source:   str
    category: str
    date:     str = "2026-01-01"

@dataclass
class Chunk:
    """A text chunk from a document with its embedding."""
    chunk_id:  str
    doc_id:    str
    title:     str
    text:      str
    source:    str
    category:  str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. WHAT IS RAG
# ══════════════════════════════════════════════════════════════════════════════

section("1. WHAT IS RAG (Retrieval-Augmented Generation)")
print("""
  RAG combines information retrieval with LLM generation to produce
  accurate, grounded answers from your own knowledge base.

  Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │                   RAG PIPELINE                                │
  │                                                               │
  │  INDEXING (offline)                                           │
  │  Documents → Chunking → Embedding → Vector Store             │
  │                                                               │
  │  RETRIEVAL (online)                                           │
  │  Query → Embed Query → Similarity Search → Top-K Chunks       │
  │                                                               │
  │  GENERATION (online)                                          │
  │  Query + Chunks → LLM Prompt → Grounded Answer               │
  └───────────────────────────────────────────────────────────────┘

  Why RAG instead of fine-tuning?
    ✦ No retraining needed — add new docs instantly
    ✦ Transparent — can cite sources
    ✦ Reduces hallucination — grounded in retrieved facts
    ✦ Cost-effective — no GPU training required
    ✦ Privacy-safe — data never leaves your infrastructure
""")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════════════════════

section("2. DOCUMENT INGESTION")

DOCUMENTS = [
    Document(
        doc_id="doc_001", title="Apache Spark Overview",
        source="internal_wiki", category="data_engineering",
        content="""Apache Spark is an open-source, distributed computing system designed for
big data processing and analytics. It provides an interface for programming entire clusters
with implicit data parallelism and fault tolerance. Spark was developed at UC Berkeley in 2009
and later donated to the Apache Software Foundation in 2013.

Spark supports multiple programming languages including Python (PySpark), Scala, Java, and R.
Its core abstraction is the Resilient Distributed Dataset (RDD), which allows data to be stored
in memory across a cluster. Spark also provides higher-level APIs including DataFrames, Datasets,
and Spark SQL for structured data processing.

Key components of Spark include: Spark Core (task scheduling, memory management), Spark SQL
(structured queries), Spark Streaming (real-time processing), MLlib (machine learning), and
GraphX (graph processing). Spark is widely used for ETL pipelines, machine learning at scale,
and interactive data analysis."""
    ),
    Document(
        doc_id="doc_002", title="Apache Kafka Guide",
        source="internal_wiki", category="data_engineering",
        content="""Apache Kafka is a distributed event streaming platform capable of handling
trillions of events per day. Originally developed at LinkedIn and open-sourced in 2011,
Kafka is used for building real-time streaming data pipelines and applications.

Kafka works through a publish-subscribe model. Producers write messages to topics, which are
partitioned and replicated across a cluster of brokers. Consumers subscribe to topics and
read messages at their own pace using consumer groups. Each partition maintains ordered,
immutable sequences of messages called offsets.

Key Kafka concepts: Topics (logical categories for messages), Partitions (ordered logs),
Brokers (Kafka servers), Consumer Groups (parallel consumers), Retention (configurable message
storage duration), and Replication (fault tolerance). Kafka is commonly used for log aggregation,
real-time analytics, event sourcing, and microservice communication."""
    ),
    Document(
        doc_id="doc_003", title="Data Warehouse Best Practices",
        source="engineering_blog", category="architecture",
        content="""A data warehouse is a central repository that integrates data from multiple
sources for reporting and analytics. Modern cloud data warehouses like Snowflake, BigQuery,
and Redshift have transformed how organizations store and query data.

Key design patterns for data warehouses include the Star Schema (fact tables surrounded by
dimension tables) and Snowflake Schema (normalized dimension tables). The Medallion Architecture
organizes data in Bronze (raw), Silver (cleaned), and Gold (aggregated) layers.

Performance best practices: Partition tables by date or high-cardinality columns. Use columnar
storage formats like Parquet and ORC. Implement query result caching. Leverage materialized
views for expensive aggregations. Use clustering keys in Snowflake for frequently filtered columns.
Monitor query performance with execution plans and optimize JOIN order."""
    ),
    Document(
        doc_id="doc_004", title="Machine Learning Pipeline Design",
        source="engineering_blog", category="machine_learning",
        content="""A machine learning pipeline automates the workflow from raw data to deployed
model. MLOps practices combine ML development with DevOps to ensure reliable, scalable model
deployment and monitoring.

A typical ML pipeline consists of: Data Ingestion (collect raw data), Feature Engineering
(transform raw data into model inputs), Model Training (fit model on training data),
Model Evaluation (measure performance on held-out data), Model Registry (version and store
trained models), Deployment (serve model via API or batch scoring), and Monitoring (track
model performance and data drift in production).

Tools in the ML ecosystem: MLflow for experiment tracking and model registry, Kubeflow for
Kubernetes-native ML workflows, Apache Airflow for pipeline orchestration, Feature Store
(Feast, Hopsworks) for feature sharing, and Seldon or BentoML for model serving."""
    ),
    Document(
        doc_id="doc_005", title="Cloud Data Platform Architecture",
        source="architecture_docs", category="architecture",
        content="""A modern cloud data platform integrates storage, compute, and orchestration
services to support analytics workloads at scale. The major cloud providers offer managed
services that reduce operational overhead.

AWS data platform components: S3 (object storage), Glue (serverless ETL), Redshift (data
warehouse), Kinesis (streaming), Athena (serverless SQL), and EMR (managed Hadoop/Spark).

Azure data platform components: ADLS Gen2 (storage), Data Factory (ETL), Synapse Analytics
(warehouse + Spark), Event Hubs (streaming), and Databricks (unified analytics).

GCP data platform components: Cloud Storage, BigQuery (serverless warehouse), Dataflow
(streaming/batch ETL), Pub/Sub (messaging), and Dataproc (managed Spark/Hadoop).

Key principles for cloud data platforms: Separate storage from compute (pay per query),
use managed services to reduce ops burden, implement data governance with cataloging and
lineage tracking, design for multi-region availability, and enforce column-level security."""
    ),
]

print(f"\n  Loaded {len(DOCUMENTS)} documents:")
for doc in DOCUMENTS:
    word_count = len(doc.content.split())
    print(f"    [{doc.doc_id}] {doc.title:<40} {word_count:>4} words  [{doc.category}]")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHUNKING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

section("3. CHUNKING STRATEGIES")
print("""
  Why chunk?
    LLMs have context limits. Splitting docs into smaller chunks:
      • Improves retrieval precision (small, focused chunks)
      • Fits within LLM context windows
      • Reduces noise injected into the prompt

  Strategies:
    Fixed-size    — split every N characters (fast, ignores semantics)
    Sentence      — split on sentence boundaries (preserves meaning)
    Overlap       — adjacent chunks share N characters (preserves context)
    Paragraph     — split on double newlines (natural document structure)
    Semantic      — split when topic shifts (advanced, uses embeddings)
""")

def chunk_by_paragraph(doc: Document, max_words: int = 120) -> List[Chunk]:
    """Split document into paragraph-based chunks with optional merging."""
    paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
    chunks, buffer, buf_words = [], [], 0

    for para in paragraphs:
        words = len(para.split())
        if buf_words + words > max_words and buffer:
            text = " ".join(buffer)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c{len(chunks)+1:02d}",
                doc_id=doc.doc_id, title=doc.title,
                text=text, source=doc.source, category=doc.category
            ))
            buffer, buf_words = [], 0
        buffer.append(para)
        buf_words += words

    if buffer:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_c{len(chunks)+1:02d}",
            doc_id=doc.doc_id, title=doc.title,
            text=" ".join(buffer), source=doc.source, category=doc.category
        ))
    return chunks

def chunk_with_overlap(doc: Document, chunk_size: int = 200,
                       overlap: int = 40) -> List[Chunk]:
    """Fixed-size word chunks with overlap between consecutive chunks."""
    words  = doc.content.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_ov{len(chunks)+1:02d}",
            doc_id=doc.doc_id, title=doc.title,
            text=" ".join(chunk_words), source=doc.source, category=doc.category
        ))
        i += chunk_size - overlap
    return chunks

# Apply paragraph chunking to all documents
ALL_CHUNKS: List[Chunk] = []
for doc in DOCUMENTS:
    doc_chunks = chunk_by_paragraph(doc, max_words=120)
    ALL_CHUNKS.extend(doc_chunks)

print(f"\n  Paragraph chunking results:")
for doc in DOCUMENTS:
    doc_chunks = [c for c in ALL_CHUNKS if c.doc_id == doc.doc_id]
    for c in doc_chunks:
        print(f"    [{c.chunk_id}]  {len(c.text.split()):>4} words  {c.text[:60]}...")

# Overlap example
subsection("Overlap Chunking (chunk_size=200, overlap=40)")
spark_doc    = DOCUMENTS[0]
overlap_chunks = chunk_with_overlap(spark_doc, chunk_size=200, overlap=40)
print(f"  Document: {spark_doc.title}")
for c in overlap_chunks:
    print(f"    [{c.chunk_id}]  {len(c.text.split()):>4} words  {c.text[:60]}...")

print(f"\n  Total chunks (paragraph strategy): {len(ALL_CHUNKS)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING & INDEXING
# ══════════════════════════════════════════════════════════════════════════════

section("4. EMBEDDING & INDEXING")

print(f"\n  Loading embedding model: {EMBED_MODEL} ...")
embed_model = SentenceTransformer(EMBED_MODEL)
DIM = embed_model.get_sentence_embedding_dimension()
print(f"  Embedding dimensions : {DIM}")

# Encode all chunks
texts = [c.text for c in ALL_CHUNKS]
embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

for chunk, emb in zip(ALL_CHUNKS, embeddings):
    chunk.embedding = emb

print(f"  Indexed {len(ALL_CHUNKS)} chunks")
print(f"  Index matrix shape   : ({len(ALL_CHUNKS)}, {DIM})")
INDEX_MATRIX = np.vstack([c.embedding for c in ALL_CHUNKS])


# ══════════════════════════════════════════════════════════════════════════════
# 5. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

section("5. RETRIEVAL")

def retrieve(query: str, top_k: int = 4,
             category_filter: Optional[str] = None) -> List[Chunk]:
    """Retrieve top-K most relevant chunks for a query."""
    q_emb = embed_model.encode(query, normalize_embeddings=True)

    # Apply metadata filter if specified
    candidates = ALL_CHUNKS
    cand_matrix = INDEX_MATRIX
    if category_filter:
        candidates  = [c for c in ALL_CHUNKS if c.category == category_filter]
        cand_matrix = np.vstack([c.embedding for c in candidates])

    scores = cand_matrix @ q_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], candidates[i]) for i in top_indices]

# Demo retrieval
queries = [
    "How does Kafka handle message ordering and fault tolerance?",
    "What are best practices for data warehouse performance?",
]

for query in queries:
    results = retrieve(query, top_k=3)
    print(f"\n  Query : \"{query}\"")
    print(f"  {'Rank':<5} {'Score':>7}  {'Chunk ID':<15} {'Preview'}")
    print(f"  {'─'*5} {'─'*7}  {'─'*15} {'─'*45}")
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"  {rank:<5} {score:>7.4f}  {chunk.chunk_id:<15} {chunk.text[:45]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 6. END-TO-END RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

section("6. END-TO-END RAG PIPELINE")

def rag(query: str, top_k: int = 4, category_filter: Optional[str] = None) -> str:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks
      2. Build augmented prompt with context
      3. Generate grounded answer from LLM
    """
    # Step 1 — Retrieve
    results = retrieve(query, top_k=top_k, category_filter=category_filter)

    # Step 2 — Build context
    context_parts = []
    for rank, (score, chunk) in enumerate(results, 1):
        context_parts.append(
            f"[Source {rank}: {chunk.title} | {chunk.chunk_id}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    # Step 3 — Augmented prompt
    system = (
        "You are a knowledgeable data engineering assistant. "
        "Answer questions using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't have enough information.' "
        "Always cite the source chunk IDs in your answer."
    )
    prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above. Cite source chunk IDs."""

    # Step 4 — Generate
    return ask_llm(prompt, system=system, max_tokens=400)


subsection("RAG Query 1 — Data Engineering")
q1 = "What are the key components of Apache Spark?"
print(f"  Query: {q1}\n")
answer1 = rag(q1)
for line in answer1.splitlines():
    print(f"    {line}")

subsection("RAG Query 2 — Architecture")
q2 = "What is the Medallion Architecture and its layers?"
print(f"  Query: {q2}\n")
answer2 = rag(q2)
for line in answer2.splitlines():
    print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. METADATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════

section("7. METADATA FILTERING")
print("""
  Metadata filters narrow the retrieval space before similarity search.
  This improves precision and reduces irrelevant context injected into prompts.

  Common filter dimensions:
    • category   — only search within a specific domain
    • source     — restrict to a particular document source
    • date range — recent documents only
    • doc_id     — restrict to specific documents
""")

subsection("Filter by category = 'machine_learning'")
q = "How is model performance monitored after deployment?"
print(f"  Query: {q}")
print(f"  Filter: category = machine_learning\n")

ml_results = retrieve(q, top_k=3, category_filter="machine_learning")
for rank, (score, chunk) in enumerate(ml_results, 1):
    print(f"    {rank}. [{score:.4f}] [{chunk.category}] {chunk.text[:70]}...")

subsection("Filter by category = 'architecture'")
q2 = "How is model performance monitored after deployment?"
arch_results = retrieve(q2, top_k=3, category_filter="architecture")
print(f"  Same query, different category filter:\n")
if arch_results:
    for rank, (score, chunk) in enumerate(arch_results, 1):
        print(f"    {rank}. [{score:.4f}] [{chunk.category}] {chunk.text[:70]}...")
else:
    print("    (no results in this category)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MULTI-DOCUMENT RAG
# ══════════════════════════════════════════════════════════════════════════════

section("8. MULTI-DOCUMENT RAG")
print("""
  Retrieve context spanning multiple documents to answer cross-cutting queries.
  The LLM synthesizes information from all retrieved chunks.
""")

q = "Compare AWS, Azure, and GCP data platform offerings."
print(f"  Query: {q}\n")

results = retrieve(q, top_k=5)
sources_used = list({chunk.title for _, chunk in results})
print(f"  Documents contributing context: {sources_used}\n")

answer = rag(q, top_k=5)
print("  Answer:")
for line in answer.splitlines():
    print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CONVERSATIONAL RAG (Multi-Turn Q&A)
# ══════════════════════════════════════════════════════════════════════════════

section("9. CONVERSATIONAL RAG (Multi-Turn Q&A with Memory)")
print("""
  Conversational RAG maintains chat history so the model can understand
  follow-up questions that reference prior answers.

  Pattern:
    • Keep a running list of (user, assistant) message pairs
    • For each new query, retrieve fresh context
    • Append context + chat history to the prompt
""")

def conversational_rag(questions: List[str], top_k: int = 3) -> None:
    """Run multi-turn RAG conversation."""
    history = []

    for i, question in enumerate(questions, 1):
        print(f"\n  Turn {i} — User: {question}")

        # Build history string
        history_str = ""
        if history:
            history_str = "Previous conversation:\n"
            for user_q, assistant_a in history:
                history_str += f"  User: {user_q}\n  Assistant: {assistant_a}\n\n"

        # Retrieve context for current question
        results = retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for _, chunk in results
        )

        system = (
            "You are a data engineering assistant. "
            "Answer based only on the provided context and conversation history. "
            "Be concise. Reference prior answers when relevant."
        )
        prompt = f"""{history_str}Context:
{context}

Current question: {question}

Answer concisely using the context above."""

        answer = ask_llm(prompt, system=system, max_tokens=250)
        history.append((question, answer))

        print(f"  Assistant:")
        for line in answer.strip().splitlines():
            print(f"    {line}")

conversational_rag([
    "What is Apache Kafka used for?",
    "How does it compare to Spark Streaming?",
    "Which one should I use for a real-time analytics pipeline?",
])


# ══════════════════════════════════════════════════════════════════════════════
# 10. RAG EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

section("10. RAG EVALUATION")
print("""
  Key RAG evaluation metrics:

  ┌──────────────────────┬──────────────────────────────────────────────────┐
  │ Metric               │ Definition                                       │
  ├──────────────────────┼──────────────────────────────────────────────────┤
  │ Context Recall       │ Are all relevant chunks retrieved?               │
  │ Context Precision    │ Are retrieved chunks actually relevant?          │
  │ Answer Faithfulness  │ Is the answer grounded in the context?           │
  │ Answer Relevance     │ Does the answer address the question?            │
  └──────────────────────┴──────────────────────────────────────────────────┘
""")

eval_cases = [
    {
        "question":      "What programming languages does Spark support?",
        "ground_truth":  "Python, Scala, Java, and R",
        "relevant_docs": ["doc_001"],
    },
    {
        "question":      "What is the Medallion Architecture?",
        "ground_truth":  "Bronze (raw), Silver (cleaned), Gold (aggregated) layers",
        "relevant_docs": ["doc_003"],
    },
    {
        "question":      "What are Kafka consumer groups?",
        "ground_truth":  "Groups of consumers that read from topics in parallel",
        "relevant_docs": ["doc_002"],
    },
]

subsection("Context Precision & Recall")
print(f"  {'Question':<50} {'C-Precision':>12} {'C-Recall':>10}")
print(f"  {'─'*50} {'─'*12} {'─'*10}")

for case in eval_cases:
    results = retrieve(case["question"], top_k=4)
    retrieved_docs = {chunk.doc_id for _, chunk in results}
    relevant_set   = set(case["relevant_docs"])

    true_pos  = len(retrieved_docs & relevant_set)
    precision = true_pos / len(retrieved_docs) if retrieved_docs else 0
    recall    = true_pos / len(relevant_set)    if relevant_set   else 0

    print(f"  {case['question'][:48]:<50} {precision:>12.2f} {recall:>10.2f}")

subsection("Answer Faithfulness (LLM-as-judge)")
eval_q = "What programming languages does Apache Spark support?"
retrieved = retrieve(eval_q, top_k=3)
context   = "\n".join(chunk.text for _, chunk in retrieved)
answer    = rag(eval_q, top_k=3)

judge_prompt = f"""You are an evaluator. Rate whether the answer is FAITHFUL to the context.

Context:
{context}

Question: {eval_q}
Answer: {answer}

Reply with JSON only: {{"faithful": true/false, "score": 0.0-1.0, "reason": "brief reason"}}"""

judge_result = ask_llm(judge_prompt, max_tokens=150)
print(f"\n  Question : {eval_q}")
print(f"  Answer   : {answer[:100]}...")
print(f"\n  Faithfulness evaluation:")
try:
    parsed = json.loads(judge_result)
    print(f"    Faithful : {parsed['faithful']}")
    print(f"    Score    : {parsed['score']}")
    print(f"    Reason   : {parsed['reason']}")
except Exception:
    for line in judge_result.splitlines():
        print(f"    {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. RAG vs FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

section("11. RAG vs FINE-TUNING")
print("""
  ┌──────────────────────┬───────────────────────────┬───────────────────────┐
  │ Dimension            │ RAG                       │ Fine-Tuning           │
  ├──────────────────────┼───────────────────────────┼───────────────────────┤
  │ Knowledge updates    │ Add docs instantly        │ Retrain required      │
  │ Source citation      │ Yes — traceable           │ No                    │
  │ Hallucination risk   │ Lower (grounded)          │ Higher                │
  │ Cost                 │ Low (inference only)      │ High (GPU training)   │
  │ Setup complexity     │ Moderate                  │ High                  │
  │ Latency              │ Higher (retrieval step)   │ Lower (direct)        │
  │ Style / tone control │ Limited                   │ Strong                │
  │ Domain adaptation    │ Via retrieved context     │ Baked into weights    │
  │ Data privacy         │ Stay on-prem              │ Data sent to trainer  │
  ├──────────────────────┼───────────────────────────┼───────────────────────┤
  │ Best for             │ Dynamic knowledge bases,  │ Fixed domain jargon,  │
  │                      │ Q&A over documents,       │ specific output style,│
  │                      │ enterprise search         │ task specialization   │
  └──────────────────────┴───────────────────────────┴───────────────────────┘

  Decision guide:
    Use RAG when:
      ✔  Knowledge changes frequently (wikis, support docs, policies)
      ✔  You need source citations and auditability
      ✔  You have a large, evolving document corpus
      ✔  Budget is limited (no GPU training)

    Use Fine-Tuning when:
      ✔  You need a specific output style or tone
      ✔  The domain vocabulary is highly specialized
      ✔  Latency is critical (no retrieval overhead)
      ✔  You have high-quality labeled examples

    Use Both (RAG + Fine-Tuning) when:
      ✔  Style consistency AND dynamic knowledge are both needed
""")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
  RAG Pipeline Topics Demonstrated:
  ──────────────────────────────────────────────────────────────────
   1.  What is RAG             Architecture: index → retrieve → generate
   2.  Document Ingestion      Load docs with metadata (id, category, source)
   3.  Chunking Strategies     Paragraph chunks + overlap chunking
   4.  Embedding & Indexing    SentenceTransformer encode → numpy matrix
   5.  Retrieval               Cosine similarity top-K chunk search
   6.  End-to-End Pipeline     retrieve → context → LLM prompt → answer
   7.  Metadata Filtering      Filter chunks by category before retrieval
   8.  Multi-Document RAG      Cross-document context synthesis
   9.  Conversational RAG      Multi-turn Q&A with chat history
  10.  RAG Evaluation          Context precision/recall + LLM-as-judge
  11.  RAG vs Fine-Tuning      Trade-off comparison and decision guide
  ──────────────────────────────────────────────────────────────────
  LLM      : {LLM_MODEL}
  Embedder : {EMBED_MODEL}  ({DIM}-dim)
  Chunks   : {len(ALL_CHUNKS)} chunks from {len(DOCUMENTS)} documents
  Library  : sentence-transformers · anthropic · numpy · scikit-learn
""")
