"""
llm/semantic_search/semantic_search.py
----------------------------------------
Comprehensive Semantic Search guide using sentence-transformers and FAISS.

Topics covered:
  1.  Keyword Search vs Semantic Search  — why semantics beats keywords
  2.  Basic Semantic Search              — encode + cosine similarity
  3.  FAISS Flat Index                   — exact nearest neighbour (brute force)
  4.  FAISS IVF Index                    — approximate nearest neighbour (fast)
  5.  FAISS HNSW Index                   — graph-based ANN (production grade)
  6.  Top-K Retrieval                    — retrieve K most relevant results
  7.  Threshold Filtering                — return only results above a score cutoff
  8.  Re-Ranking                         — coarse retrieve → fine-grained re-rank
  9.  Hybrid Search                      — combine keyword (BM25) + semantic scores
  10. Multi-Query Search                 — expand query with paraphrases
  11. Incremental Index Updates          — add new documents to a live index
  12. Search Analytics                   — query latency and recall measurement

Model  : all-MiniLM-L6-v2 (384-dim)
Library: sentence-transformers · faiss · scikit-learn · numpy

Usage:
    pip install sentence-transformers faiss-cpu scikit-learn numpy
    python semantic_search.py
"""

import time
import numpy as np
import faiss
from sentence_transformers      import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise   import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 66)
    print(f"  {title}")
    print("═" * 66)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def elapsed_ms(start: float) -> str:
    return f"{(time.time() - start) * 1000:.2f} ms"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL & CORPUS
# ══════════════════════════════════════════════════════════════════════════════

section("LOADING MODEL & BUILDING CORPUS")

MODEL_NAME = "all-MiniLM-L6-v2"
print(f"\n  Loading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
DIM   = model.get_sentence_embedding_dimension()
print(f"  Embedding dimensions : {DIM}")

# ── Knowledge base corpus (data engineering & ML domain) ─────────────────────
CORPUS = [
    # Data Engineering
    "Apache Spark is a distributed data processing engine for large-scale analytics.",
    "Apache Kafka is a distributed event streaming platform for real-time data pipelines.",
    "Apache Airflow is a workflow orchestration tool used to schedule and monitor ETL pipelines.",
    "Delta Lake adds ACID transactions and schema enforcement on top of data lakes.",
    "dbt (data build tool) transforms raw data inside the warehouse using SQL models.",
    "Apache Flink provides stateful stream processing with exactly-once semantics.",
    "Apache Hive enables SQL-like querying on top of Hadoop distributed file system.",
    "Presto is a distributed SQL query engine for querying large datasets across sources.",
    "Apache Iceberg is an open table format providing snapshot isolation for large tables.",
    "Apache Parquet is a columnar storage format optimized for analytical queries.",
    # Machine Learning
    "Scikit-learn provides simple tools for classification, regression, and clustering.",
    "TensorFlow is an end-to-end open-source platform for machine learning.",
    "PyTorch is a deep learning framework with dynamic computation graphs.",
    "Hugging Face Transformers provides pre-trained models for NLP tasks.",
    "XGBoost is an optimized gradient boosting library for tabular data.",
    "MLflow tracks machine learning experiments, parameters, and model artifacts.",
    "Kubeflow deploys machine learning workflows on Kubernetes clusters.",
    "Feature stores centralize and serve features for training and inference.",
    "Model drift detection monitors when model performance degrades over time.",
    "AutoML automatically searches for optimal model architectures and hyperparameters.",
    # Cloud & Infrastructure
    "AWS S3 provides scalable object storage for data lakes and backups.",
    "AWS Glue is a serverless ETL service for data discovery and transformation.",
    "Google BigQuery is a serverless data warehouse for analytics at scale.",
    "Azure Synapse Analytics combines big data and data warehousing capabilities.",
    "Snowflake is a cloud-native data warehouse with automatic scaling and concurrency.",
    "Databricks provides a unified analytics platform built on Apache Spark.",
    "Kubernetes orchestrates containerized applications across distributed clusters.",
    "Docker packages applications and their dependencies into portable containers.",
    "Terraform provisions infrastructure as code across cloud providers.",
    "GitHub Actions automates CI/CD workflows for software delivery pipelines.",
]

print(f"  Corpus size          : {len(CORPUS)} documents")

t = time.time()
CORPUS_EMB = model.encode(CORPUS, normalize_embeddings=True, show_progress_bar=False)
print(f"  Encoded corpus       : {CORPUS_EMB.shape}  in {elapsed_ms(t)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. KEYWORD SEARCH vs SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════

section("1. KEYWORD SEARCH vs SEMANTIC SEARCH")
print("""
  Keyword Search (TF-IDF / BM25):
    • Matches exact words or stems
    • Fast and interpretable
    • Fails on synonyms, paraphrases, and intent

  Semantic Search (Embeddings):
    • Matches meaning, not exact words
    • Understands synonyms, context, and intent
    • Handles "how to ingest streaming events" ↔ "Kafka pipeline setup"

  Example: Query = "how to process real-time events at scale"
""")

query = "how to process real-time events at scale"

# ── Keyword search (TF-IDF) ───────────────────────────────────────────────────
subsection("Keyword Search (TF-IDF)")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(CORPUS)
q_vec   = tfidf.transform([query])
kw_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
kw_ranked = sorted(enumerate(kw_scores), key=lambda x: x[1], reverse=True)[:5]

print(f"  Query  : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Document'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (idx, score) in enumerate(kw_ranked, 1):
    print(f"  {rank:<5} {score:>7.4f}  {CORPUS[idx][:60]}")

# ── Semantic search ───────────────────────────────────────────────────────────
subsection("Semantic Search (Embeddings)")
q_emb  = model.encode(query, normalize_embeddings=True)
sem_scores = CORPUS_EMB @ q_emb
sem_ranked = sorted(enumerate(sem_scores), key=lambda x: x[1], reverse=True)[:5]

print(f"  Query  : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Document'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (idx, score) in enumerate(sem_ranked, 1):
    print(f"  {rank:<5} {score:>7.4f}  {CORPUS[idx][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BASIC SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════

section("2. BASIC SEMANTIC SEARCH")

def basic_search(query: str, top_k: int = 5) -> list:
    q_emb  = model.encode(query, normalize_embeddings=True)
    scores = CORPUS_EMB @ q_emb
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], CORPUS[i]) for i in ranked]

queries = [
    "Which tool helps schedule and monitor data workflows?",
    "What is the best cloud data warehouse?",
    "How do I track ML experiment results?",
]

for q in queries:
    results = basic_search(q)
    print(f"\n  Query : \"{q}\"")
    print(f"  {'Rank':<5} {'Score':>7}  {'Result'}")
    print(f"  {'─'*5} {'─'*7}  {'─'*57}")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank:<5} {score:>7.4f}  {doc[:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FAISS FLAT INDEX (Exact Nearest Neighbour)
# ══════════════════════════════════════════════════════════════════════════════

section("3. FAISS FLAT INDEX (Exact Nearest Neighbour)")
print("""
  IndexFlatIP  — Inner product (use with L2-normalized vectors = cosine similarity)
  IndexFlatL2  — L2 / Euclidean distance

  • Brute-force: checks every vector (100% recall)
  • Best for   : small-to-medium corpora (< 100K vectors)
  • No training required
""")

index_flat = faiss.IndexFlatIP(DIM)
index_flat.add(CORPUS_EMB.astype("float32"))
print(f"  Index type       : IndexFlatIP")
print(f"  Vectors indexed  : {index_flat.ntotal}")

query = "ACID transactions for data lakes"
q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")

t = time.time()
scores, indices = index_flat.search(q_emb, k=5)
print(f"  Search latency   : {elapsed_ms(t)}")

print(f"\n  Query : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Result'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<5} {score:>7.4f}  {CORPUS[idx][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FAISS IVF INDEX (Approximate Nearest Neighbour)
# ══════════════════════════════════════════════════════════════════════════════

section("4. FAISS IVF INDEX (Approximate Nearest Neighbour)")
print("""
  IndexIVFFlat — Inverted File Index
    • Divides vector space into nlist Voronoi cells
    • At search time, only nprobe cells are scanned
    • Faster than flat at the cost of slight recall loss
    • Requires training before adding vectors
    • Best for: medium corpora (100K – 10M vectors)
""")

nlist      = 8        # number of Voronoi cells (typically sqrt(N))
nprobe     = 3        # cells to scan at query time (higher = better recall)
quantizer  = faiss.IndexFlatIP(DIM)
index_ivf  = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)

index_ivf.train(CORPUS_EMB.astype("float32"))
index_ivf.add(CORPUS_EMB.astype("float32"))
index_ivf.nprobe = nprobe

print(f"  Index type       : IndexIVFFlat")
print(f"  nlist (cells)    : {nlist}")
print(f"  nprobe (search)  : {nprobe}")
print(f"  Vectors indexed  : {index_ivf.ntotal}")

query = "streaming data pipeline for real-time analytics"
q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")

t = time.time()
scores, indices = index_ivf.search(q_emb, k=5)
print(f"  Search latency   : {elapsed_ms(t)}")

print(f"\n  Query : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Result'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<5} {score:>7.4f}  {CORPUS[idx][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FAISS HNSW INDEX (Graph-Based ANN — Production Grade)
# ══════════════════════════════════════════════════════════════════════════════

section("5. FAISS HNSW INDEX (Graph-Based ANN)")
print("""
  IndexHNSWFlat — Hierarchical Navigable Small World graphs
    • Logarithmic search time O(log N) — extremely fast at scale
    • High recall (~99%) with low latency
    • M        : number of neighbours per node (16–64; higher = better recall, more memory)
    • efSearch : search expansion (higher = better recall, slower)
    • No GPU required, works well on CPU
    • Best for : large corpora (1M+ vectors), production deployments
""")

M          = 16     # HNSW neighbours
ef_search  = 50     # search expansion factor
index_hnsw = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efSearch = ef_search
index_hnsw.add(CORPUS_EMB.astype("float32"))

print(f"  Index type       : IndexHNSWFlat")
print(f"  M (neighbours)   : {M}")
print(f"  efSearch         : {ef_search}")
print(f"  Vectors indexed  : {index_hnsw.ntotal}")

query = "deploying ML models on Kubernetes"
q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")

t = time.time()
scores, indices = index_hnsw.search(q_emb, k=5)
print(f"  Search latency   : {elapsed_ms(t)}")

print(f"\n  Query : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Result'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<5} {score:>7.4f}  {CORPUS[idx][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TOP-K RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

section("6. TOP-K RETRIEVAL")
print("""
  Retrieve exactly K results regardless of score.
  Common pattern for downstream LLM context injection (RAG).
""")

def top_k_search(query: str, index: faiss.Index, k: int = 5) -> list:
    q_emb  = model.encode(query, normalize_embeddings=True).reshape(1,-1).astype("float32")
    scores, indices = index.search(q_emb, k=k)
    return list(zip(scores[0], [CORPUS[i] for i in indices[0]]))

for k in [3, 5, 8]:
    results = top_k_search("cloud data warehousing and analytics", index_flat, k=k)
    print(f"\n  Top-{k} results for \"cloud data warehousing and analytics\":")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"    {rank}. [{score:.4f}] {doc[:65]}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. THRESHOLD FILTERING
# ══════════════════════════════════════════════════════════════════════════════

section("7. THRESHOLD FILTERING")
print("""
  Only return results that exceed a minimum similarity score.
  Prevents low-quality / irrelevant results from reaching the user.

  Common thresholds:
    > 0.8  — very high confidence (near-duplicate or paraphrase)
    > 0.6  — relevant result
    > 0.4  — loosely related
    < 0.4  — likely irrelevant (filter out)
""")

def threshold_search(query: str, threshold: float = 0.5, max_k: int = 10) -> list:
    q_emb  = model.encode(query, normalize_embeddings=True)
    scores = CORPUS_EMB @ q_emb
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:max_k]
    return [(s, CORPUS[i]) for i, s in ranked if s >= threshold]

query = "SQL query engine for big data"
for threshold in [0.7, 0.55, 0.4]:
    results = threshold_search(query, threshold=threshold)
    print(f"\n  Query: \"{query}\"  |  Threshold ≥ {threshold}")
    if results:
        for rank, (score, doc) in enumerate(results, 1):
            print(f"    {rank}. [{score:.4f}] {doc[:65]}")
    else:
        print(f"    (no results above threshold {threshold})")


# ══════════════════════════════════════════════════════════════════════════════
# 8. RE-RANKING
# ══════════════════════════════════════════════════════════════════════════════

section("8. RE-RANKING (Coarse Retrieve → Fine-Grained Re-Rank)")
print("""
  Two-stage retrieval pipeline:
    Stage 1 — Bi-encoder (fast)  : retrieve top-N candidates using FAISS
    Stage 2 — Cross-encoder (slow): re-score top-N with a more powerful model

  Why re-rank?
    • Bi-encoders are fast but less accurate (encode independently)
    • Cross-encoders see both query and document jointly (more accurate)
    • Re-ranking only the top-N keeps latency manageable

  Simulated here with a second scoring pass using dot product on full model.
""")

from sentence_transformers import CrossEncoder

query = "best tool for scheduling data pipeline workflows"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"  Loading cross-encoder: {RERANK_MODEL} ...")
cross_encoder = CrossEncoder(RERANK_MODEL)

# Stage 1: Bi-encoder retrieval (top-15)
q_emb     = model.encode(query, normalize_embeddings=True)
bi_scores = CORPUS_EMB @ q_emb
top15_idx = np.argsort(bi_scores)[::-1][:15]
candidates = [(bi_scores[i], CORPUS[i]) for i in top15_idx]

print(f"\n  Stage 1 — Bi-Encoder top-5 (of 15 candidates):")
print(f"  {'Rank':<5} {'Bi-Score':>9}  {'Document'}")
print(f"  {'─'*5} {'─'*9}  {'─'*55}")
for rank, (score, doc) in enumerate(candidates[:5], 1):
    print(f"  {rank:<5} {score:>9.4f}  {doc[:57]}")

# Stage 2: Cross-encoder re-ranking
pairs    = [[query, doc] for _, doc in candidates]
re_scores = cross_encoder.predict(pairs)
reranked  = sorted(zip(re_scores, [doc for _, doc in candidates]), reverse=True)

print(f"\n  Stage 2 — Cross-Encoder Re-Ranked top-5:")
print(f"  {'Rank':<5} {'CE-Score':>9}  {'Document'}")
print(f"  {'─'*5} {'─'*9}  {'─'*55}")
for rank, (score, doc) in enumerate(reranked[:5], 1):
    print(f"  {rank:<5} {score:>9.4f}  {doc[:57]}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. HYBRID SEARCH (Keyword + Semantic)
# ══════════════════════════════════════════════════════════════════════════════

section("9. HYBRID SEARCH (BM25-style TF-IDF + Semantic)")
print("""
  Hybrid search combines:
    • Sparse retrieval  (TF-IDF / BM25) — exact keyword match
    • Dense retrieval   (embeddings)    — semantic meaning

  Fusion method: Reciprocal Rank Fusion (RRF)
    score = Σ 1 / (k + rank_i)   where k=60 is a smoothing constant

  Benefits:
    ✦ Catches keyword-specific queries that embeddings miss
    ✦ Catches semantic queries that keyword search misses
    ✦ Generally outperforms either method alone
""")

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> dict:
    """Merge multiple ranked lists using RRF."""
    scores = {}
    for ranked_list in rankings:
        for rank, doc_idx in enumerate(ranked_list, 1):
            scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank)
    return scores

query = "SQL engine for querying distributed data"

# Keyword ranking
tfidf_scores = cosine_similarity(tfidf.transform([query]), tfidf_matrix).flatten()
kw_ranking   = list(np.argsort(tfidf_scores)[::-1])

# Semantic ranking
q_emb        = model.encode(query, normalize_embeddings=True)
sem_scores_v = CORPUS_EMB @ q_emb
sem_ranking  = list(np.argsort(sem_scores_v)[::-1])

# RRF fusion
rrf_scores = reciprocal_rank_fusion([kw_ranking, sem_ranking])
hybrid_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"\n  Query : \"{query}\"")
print(f"\n  {'Rank':<5} {'KW Rank':>8} {'Sem Rank':>9} {'RRF Score':>10}  {'Document'}")
print(f"  {'─'*5} {'─'*8} {'─'*9} {'─'*10}  {'─'*50}")
for rank, (idx, rrf_score) in enumerate(hybrid_ranked, 1):
    kw_rank  = kw_ranking.index(idx)  + 1
    sem_rank = sem_ranking.index(idx) + 1
    print(f"  {rank:<5} {kw_rank:>8} {sem_rank:>9} {rrf_score:>10.5f}  {CORPUS[idx][:52]}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MULTI-QUERY SEARCH (Query Expansion)
# ══════════════════════════════════════════════════════════════════════════════

section("10. MULTI-QUERY SEARCH (Query Expansion)")
print("""
  A single query may miss relevant results due to vocabulary gap.
  Multi-query search:
    1. Generate multiple paraphrases / expansions of the original query
    2. Run semantic search for each variant
    3. Merge results using score averaging or max pooling

  Benefits:
    • Better recall — catches documents each variant would miss
    • More robust — reduces sensitivity to exact query phrasing
""")

original_query = "container orchestration for microservices"
query_variants = [
    "container orchestration for microservices",
    "Kubernetes managing containerized applications",
    "deploying Docker containers at scale",
    "microservice deployment and scaling platform",
]

print(f"  Original query: \"{original_query}\"")
print(f"  Variants       : {len(query_variants)} (including original)\n")

# Encode all variants and average their embeddings
variant_embs = model.encode(query_variants, normalize_embeddings=True)
avg_emb      = variant_embs.mean(axis=0)
avg_emb      = avg_emb / np.linalg.norm(avg_emb)   # re-normalize

multi_scores = CORPUS_EMB @ avg_emb
multi_ranked = sorted(enumerate(multi_scores), key=lambda x: x[1], reverse=True)[:5]

# Compare single-query vs multi-query
single_scores = CORPUS_EMB @ model.encode(original_query, normalize_embeddings=True)
single_ranked = sorted(enumerate(single_scores), key=lambda x: x[1], reverse=True)[:5]

print(f"  {'Rank':<5} {'Single':>8} {'Multi':>8}  {'Document'}")
print(f"  {'─'*5} {'─'*8} {'─'*8}  {'─'*52}")
all_idx = list(dict.fromkeys([i for i,_ in single_ranked] + [i for i,_ in multi_ranked]))
for rank, idx in enumerate(all_idx[:6], 1):
    ss = single_scores[idx]
    ms = multi_scores[idx]
    indicator = "← improved" if ms > ss + 0.02 else ""
    print(f"  {rank:<5} {ss:>8.4f} {ms:>8.4f}  {CORPUS[idx][:52]}  {indicator}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. INCREMENTAL INDEX UPDATES
# ══════════════════════════════════════════════════════════════════════════════

section("11. INCREMENTAL INDEX UPDATES")
print("""
  Production search indexes need to accept new documents without
  rebuilding from scratch.

  FAISS IndexFlatIP supports add() incrementally.
  For IVF / HNSW, new vectors can be added post-training.
""")

# Start with first 20 documents
initial_docs = CORPUS[:20]
initial_emb  = CORPUS_EMB[:20].astype("float32")

live_index = faiss.IndexFlatIP(DIM)
live_index.add(initial_emb)
print(f"  Initial index size   : {live_index.ntotal} documents")

# Add 5 new documents incrementally
new_docs = [
    "Apache Arrow provides a columnar memory format for fast analytics.",
    "OpenLineage tracks data lineage across pipelines and transformations.",
    "Apache Nifi automates data flows between systems with a visual interface.",
    "Trino (formerly PrestoSQL) is a fast distributed SQL query engine.",
    "Great Expectations validates, documents, and profiles data quality.",
]
new_emb = model.encode(new_docs, normalize_embeddings=True).astype("float32")
live_index.add(new_emb)

all_docs_live = initial_docs + new_docs
print(f"  After adding 5 docs  : {live_index.ntotal} documents")

query = "data quality validation and profiling"
q_emb = model.encode(query, normalize_embeddings=True).reshape(1,-1).astype("float32")
scores, indices = live_index.search(q_emb, k=5)

print(f"\n  Query : \"{query}\"")
print(f"  {'Rank':<5} {'Score':>7}  {'Document'}")
print(f"  {'─'*5} {'─'*7}  {'─'*57}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    tag = " ← newly added" if idx >= 20 else ""
    print(f"  {rank:<5} {score:>7.4f}  {all_docs_live[idx][:57]}{tag}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. SEARCH ANALYTICS (Latency & Recall)
# ══════════════════════════════════════════════════════════════════════════════

section("12. SEARCH ANALYTICS (Latency & Recall@K)")
print("""
  Key metrics for evaluating a semantic search system:

  Latency   — time from query submission to results returned
  Recall@K  — fraction of relevant docs found in top-K results
              Recall@5 = (relevant docs in top-5) / (total relevant docs)
  MRR       — Mean Reciprocal Rank: 1/rank of first relevant result
""")

# ── Latency benchmark ─────────────────────────────────────────────────────────
subsection("Latency Benchmark — Flat vs IVF vs HNSW")

bench_query = "distributed data processing for big data workloads"
q_emb_bench = model.encode(bench_query, normalize_embeddings=True).reshape(1,-1).astype("float32")

RUNS = 100
results_table = []
for name, idx in [("IndexFlatIP", index_flat), ("IndexIVFFlat", index_ivf), ("IndexHNSWFlat", index_hnsw)]:
    times = []
    for _ in range(RUNS):
        t = time.time()
        idx.search(q_emb_bench, k=5)
        times.append((time.time() - t) * 1000)
    avg_ms = np.mean(times)
    p99_ms = np.percentile(times, 99)
    results_table.append((name, avg_ms, p99_ms))

print(f"\n  {'Index Type':<16} {'Avg (ms)':>10} {'P99 (ms)':>10}  {'Note'}")
print(f"  {'─'*16} {'─'*10} {'─'*10}  {'─'*35}")
for name, avg, p99 in results_table:
    note = {"IndexFlatIP": "Exact, 100% recall",
            "IndexIVFFlat": f"Approx, nprobe={nprobe}",
            "IndexHNSWFlat": f"Graph ANN, M={M}"}.get(name, "")
    print(f"  {name:<16} {avg:>10.3f} {p99:>10.3f}  {note}")

# ── Recall@K evaluation ────────────────────────────────────────────────────────
subsection("Recall@K Evaluation")

# Ground truth: for each query, the first result from flat (exact) is "relevant"
eval_queries = [
    ("workflow orchestration scheduling",    [2]),   # Airflow at index 2
    ("columnar storage format for analytics", [9]),  # Parquet at index 9
    ("deep learning framework Python",       [12]),  # PyTorch at index 12
    ("cloud object storage data lake",       [20]),  # AWS S3 at index 20
    ("infrastructure as code provisioning",  [28]),  # Terraform at index 28
]

print(f"\n  {'Query':<45} {'R@1':>5} {'R@3':>5} {'R@5':>5}")
print(f"  {'─'*45} {'─'*5} {'─'*5} {'─'*5}")
for query_text, relevant_ids in eval_queries:
    q_emb_e = model.encode(query_text, normalize_embeddings=True).reshape(1,-1).astype("float32")
    _, top_indices = index_flat.search(q_emb_e, k=5)
    top_ids = list(top_indices[0])
    r1 = 1.0 if any(r in top_ids[:1] for r in relevant_ids) else 0.0
    r3 = 1.0 if any(r in top_ids[:3] for r in relevant_ids) else 0.0
    r5 = 1.0 if any(r in top_ids[:5] for r in relevant_ids) else 0.0
    print(f"  {query_text:<45} {r1:>5.1f} {r3:>5.1f} {r5:>5.1f}")

r_at_1 = sum(1.0 if any(r in list(index_flat.search(
    model.encode(q, normalize_embeddings=True).reshape(1,-1).astype("float32"), k=1)[1][0])
    for r in rel) else 0.0 for q, rel in eval_queries) / len(eval_queries)
r_at_5 = sum(1.0 if any(r in list(index_flat.search(
    model.encode(q, normalize_embeddings=True).reshape(1,-1).astype("float32"), k=5)[1][0])
    for r in rel) else 0.0 for q, rel in eval_queries) / len(eval_queries)
print(f"\n  Mean Recall@1 : {r_at_1:.2f}")
print(f"  Mean Recall@5 : {r_at_5:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
  Semantic Search Topics Demonstrated:
  ────────────────────────────────────────────────────────────────
   1.  Keyword vs Semantic    TF-IDF loses on synonyms; embeddings win
   2.  Basic Semantic Search  encode + dot product on normalized vectors
   3.  FAISS Flat Index       Exact search, 100% recall, O(N) scan
   4.  FAISS IVF Index        Approx search, Voronoi partitioning
   5.  FAISS HNSW Index       Graph ANN, log(N) lookup, production grade
   6.  Top-K Retrieval        Retrieve exactly K results for LLM context
   7.  Threshold Filtering    Drop results below minimum score cutoff
   8.  Re-Ranking             Bi-encoder retrieve → cross-encoder re-rank
   9.  Hybrid Search          RRF fusion of keyword + semantic rankings
  10.  Multi-Query Search     Query expansion + embedding averaging
  11.  Incremental Updates    Add new documents to live FAISS index
  12.  Search Analytics       Latency benchmarks + Recall@K evaluation
  ────────────────────────────────────────────────────────────────
  Model  : {MODEL_NAME}  ({DIM}-dim)
  Library: sentence-transformers · faiss-cpu · scikit-learn · numpy

  FAISS Index Selection Guide:
  ┌────────────────┬───────────────┬──────────────────────────────────┐
  │ Index          │ Corpus Size   │ Use Case                         │
  ├────────────────┼───────────────┼──────────────────────────────────┤
  │ IndexFlatIP    │ < 100K        │ Dev, testing, exact recall needed │
  │ IndexIVFFlat   │ 100K – 10M   │ Production with training budget   │
  │ IndexHNSWFlat  │ 1M+           │ High-throughput production search │
  └────────────────┴───────────────┴──────────────────────────────────┘
""")
