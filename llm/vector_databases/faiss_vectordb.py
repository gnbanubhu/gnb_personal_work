"""
llm/vector_databases/faiss.py
-------------------------------
Comprehensive FAISS (Facebook AI Similarity Search) guide.

Topics covered:
  1.  What is FAISS                — architecture and use cases
  2.  IndexFlatL2                  — exact search using Euclidean distance
  3.  IndexFlatIP                  — exact search using inner product (cosine)
  4.  IndexIVFFlat                 — approximate search with inverted file index
  5.  IndexHNSWFlat                — graph-based approximate nearest neighbour
  6.  IndexPQ                      — product quantization for memory compression
  7.  IndexIVFPQ                   — IVF + PQ combined (production standard)
  8.  ID Mapping                   — use custom IDs instead of sequential integers
  9.  Index Serialization          — save and load indexes to/from disk
  10. Batch Search                 — search multiple queries at once
  11. Index Benchmarking           — latency, recall, and memory comparison
  12. Vector Store Pattern         — document store + FAISS index (RAG use case)

Library: faiss-cpu · numpy

Usage:
    pip install faiss-cpu numpy
    python faiss.py
"""

import os
import time
import tempfile
import numpy as np
import faiss


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
    return f"{(time.time() - start) * 1000:.3f} ms"

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)

def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors for testing."""
    rng = np.random.default_rng(seed)
    v   = rng.random((n, dim)).astype("float32")
    return l2_normalize(v)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DIM      = 128          # embedding dimension
N_CORPUS = 10_000       # number of corpus vectors
N_QUERY  = 100          # number of query vectors
TOP_K    = 5            # neighbours to retrieve

print(f"  FAISS version     : {faiss.__version__}")
print(f"  Dimension (DIM)   : {DIM}")
print(f"  Corpus size       : {N_CORPUS:,}")
print(f"  Query count       : {N_QUERY}")
print(f"  Top-K             : {TOP_K}")

# Pre-generate corpus and query vectors
CORPUS_VECS = generate_vectors(N_CORPUS, DIM, seed=42)
QUERY_VECS  = generate_vectors(N_QUERY,  DIM, seed=99)


# ══════════════════════════════════════════════════════════════════════════════
# 1. WHAT IS FAISS
# ══════════════════════════════════════════════════════════════════════════════

section("1. WHAT IS FAISS")
print("""
  FAISS (Facebook AI Similarity Search) is a library for efficient
  similarity search and clustering of dense vectors.

  Core idea:
    Given a set of vectors X and a query vector q,
    find the K vectors in X most similar to q.

  Why FAISS?
    • Handles billions of vectors efficiently
    • Multiple index types: exact ↔ approximate trade-off
    • GPU support for massive throughput
    • Written in C++ with Python bindings — very fast

  Distance metrics:
    METRIC_L2          Euclidean distance  (lower = more similar)
    METRIC_INNER_PRODUCT  Dot product / cosine  (higher = more similar)

  Index selection guide:
  ┌──────────────────┬──────────────┬────────────┬────────────────────────┐
  │ Index            │ Recall       │ Speed      │ Best for               │
  ├──────────────────┼──────────────┼────────────┼────────────────────────┤
  │ IndexFlatL2      │ 100% exact   │ Slow (O N) │ Dev / small corpus     │
  │ IndexFlatIP      │ 100% exact   │ Slow (O N) │ Dev / cosine search    │
  │ IndexIVFFlat     │ ~95-99%      │ Fast       │ Medium corpus 100K-10M │
  │ IndexHNSWFlat    │ ~99%         │ Very fast  │ Production, large data │
  │ IndexPQ          │ ~85-95%      │ Very fast  │ Huge corpus, low memory│
  │ IndexIVFPQ       │ ~90-97%      │ Very fast  │ 1B+ vectors production │
  └──────────────────┴──────────────┴────────────┴────────────────────────┘
""")


# ══════════════════════════════════════════════════════════════════════════════
# 2. IndexFlatL2 — Exact Euclidean Distance
# ══════════════════════════════════════════════════════════════════════════════

section("2. IndexFlatL2 — Exact Euclidean Distance")
print("""
  • Brute-force scan of all vectors
  • 100% recall — never misses the true nearest neighbour
  • Distance metric: L2 (lower = more similar)
  • No training required
  • Memory: 4 bytes × N × DIM
""")

index_l2 = faiss.IndexFlatL2(DIM)
index_l2.add(CORPUS_VECS)

print(f"  Index type    : IndexFlatL2")
print(f"  Total vectors : {index_l2.ntotal:,}")
print(f"  Dimension     : {index_l2.d}")
print(f"  Is trained    : {index_l2.is_trained}")
print(f"  Memory (est.) : {index_l2.ntotal * DIM * 4 / 1024:.1f} KB")

q = QUERY_VECS[:1]
t = time.time()
distances, indices = index_l2.search(q, k=TOP_K)
print(f"\n  Single query latency : {elapsed_ms(t)}")
print(f"\n  Query vector (first 6 dims): {q[0][:6].round(4)}")
print(f"\n  {'Rank':<6} {'Distance (L2)':>14}  {'Vector ID':>10}")
print(f"  {'─'*6} {'─'*14}  {'─'*10}")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    print(f"  {rank:<6} {dist:>14.6f}  {idx:>10}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. IndexFlatIP — Exact Inner Product (Cosine)
# ══════════════════════════════════════════════════════════════════════════════

section("3. IndexFlatIP — Exact Inner Product / Cosine Similarity")
print("""
  • Brute-force scan using inner product (dot product)
  • With L2-normalized vectors: inner product = cosine similarity
  • Score range: -1.0 (opposite) to +1.0 (identical)
  • Higher score = more similar
  • Preferred over IndexFlatL2 for semantic search
""")

index_ip = faiss.IndexFlatIP(DIM)
index_ip.add(CORPUS_VECS)   # already L2-normalized

print(f"  Index type    : IndexFlatIP")
print(f"  Total vectors : {index_ip.ntotal:,}")

q = QUERY_VECS[:1]
t = time.time()
scores, indices = index_ip.search(q, k=TOP_K)
print(f"\n  Single query latency : {elapsed_ms(t)}")

print(f"\n  {'Rank':<6} {'Cosine Score':>13}  {'Vector ID':>10}")
print(f"  {'─'*6} {'─'*13}  {'─'*10}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<6} {score:>13.6f}  {idx:>10}")

# Show that adding the same vector returns score ≈ 1.0
index_ip.add(q)  # add query itself
scores2, indices2 = index_ip.search(q, k=1)
print(f"\n  Self-similarity (query vs itself) : {scores2[0][0]:.6f}  (≈ 1.0 ✓)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. IndexIVFFlat — Approximate Nearest Neighbour with Inverted File
# ══════════════════════════════════════════════════════════════════════════════

section("4. IndexIVFFlat — Inverted File Approximate Search")
print("""
  Divides vector space into nlist Voronoi cells using k-means clustering.
  At search time only nprobe cells are scanned — much faster than flat.

  Parameters:
    nlist   — number of Voronoi cells (typical: sqrt(N) to 4*sqrt(N))
    nprobe  — cells scanned per query (higher → better recall, slower)

  Training:
    IVF indexes require a training step to learn the cell centroids.
    Training data: ≥ 39 * nlist vectors recommended.

  Trade-off:
    nprobe=1  → fastest, lowest recall
    nprobe=nlist → same as flat (100% recall, no speedup)
""")

nlist  = int(np.sqrt(N_CORPUS))   # ≈ 100
nprobe = max(1, nlist // 10)      # ≈ 10

quantizer   = faiss.IndexFlatIP(DIM)
index_ivf   = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)

t = time.time()
index_ivf.train(CORPUS_VECS)
print(f"  Training time  : {elapsed_ms(t)}")

index_ivf.add(CORPUS_VECS)
index_ivf.nprobe = nprobe

print(f"  Index type     : IndexIVFFlat")
print(f"  nlist          : {nlist}")
print(f"  nprobe         : {nprobe}")
print(f"  Total vectors  : {index_ivf.ntotal:,}")
print(f"  Is trained     : {index_ivf.is_trained}")

q = QUERY_VECS[:1]
t = time.time()
scores, indices = index_ivf.search(q, k=TOP_K)
print(f"\n  Query latency (nprobe={nprobe}) : {elapsed_ms(t)}")

print(f"\n  {'Rank':<6} {'Score':>9}  {'Vector ID':>10}")
print(f"  {'─'*6} {'─'*9}  {'─'*10}")
for rank, (s, i) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<6} {s:>9.6f}  {i:>10}")

# Recall vs nprobe
subsection("Recall@5 vs nprobe")

# Ground truth from flat index
gt_scores, gt_indices = index_ip.search(QUERY_VECS, k=TOP_K)
gt_sets = [set(row) for row in gt_indices]

print(f"  {'nprobe':<8} {'Recall@5':>10}  {'Avg Latency':>13}")
print(f"  {'─'*8} {'─'*10}  {'─'*13}")
for np_val in [1, 5, 10, 20, nlist]:
    index_ivf.nprobe = np_val
    times = []
    all_correct = 0
    t0 = time.time()
    ivf_scores, ivf_indices = index_ivf.search(QUERY_VECS, k=TOP_K)
    avg_lat = (time.time() - t0) / N_QUERY * 1000
    for gt_set, ivf_row in zip(gt_sets, ivf_indices):
        all_correct += len(gt_set & set(ivf_row))
    recall = all_correct / (N_QUERY * TOP_K)
    print(f"  {np_val:<8} {recall:>10.3f}  {avg_lat:>11.3f} ms")

index_ivf.nprobe = nprobe  # reset


# ══════════════════════════════════════════════════════════════════════════════
# 5. IndexHNSWFlat — Graph-Based ANN (Production Grade)
# ══════════════════════════════════════════════════════════════════════════════

section("5. IndexHNSWFlat — Hierarchical Navigable Small World")
print("""
  HNSW builds a multi-layer proximity graph for fast approximate search.
  Each node is connected to M nearest neighbours at each layer.

  Parameters:
    M         — number of connections per node (16–64; higher = better recall, more memory)
    efSearch  — search expansion at query time (higher = better recall, slower)
    efConstruction — graph build quality (higher = better graph, slower build)

  Advantages:
    • No training required
    • Very fast query time O(log N)
    • High recall (~99%) with low latency
    • Supports incremental add() without rebuild

  Memory: ~(M × 2 × 4 + DIM × 4) bytes per vector
""")

M               = 32
ef_search       = 64
ef_construction = 80

index_hnsw = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efSearch       = ef_search
index_hnsw.hnsw.efConstruction = ef_construction

t = time.time()
index_hnsw.add(CORPUS_VECS)
build_time = elapsed_ms(t)

print(f"  Index type      : IndexHNSWFlat")
print(f"  M               : {M}")
print(f"  efSearch        : {ef_search}")
print(f"  efConstruction  : {ef_construction}")
print(f"  Build time      : {build_time}")
print(f"  Total vectors   : {index_hnsw.ntotal:,}")

q = QUERY_VECS[:1]
t = time.time()
scores, indices = index_hnsw.search(q, k=TOP_K)
print(f"\n  Single query latency : {elapsed_ms(t)}")

print(f"\n  {'Rank':<6} {'Score':>9}  {'Vector ID':>10}")
print(f"  {'─'*6} {'─'*9}  {'─'*10}")
for rank, (s, i) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  {rank:<6} {s:>9.6f}  {i:>10}")

# Recall vs efSearch
subsection("Recall@5 vs efSearch")
print(f"  {'efSearch':<10} {'Recall@5':>10}  {'Avg Latency':>13}")
print(f"  {'─'*10} {'─'*10}  {'─'*13}")
for ef in [8, 16, 32, 64, 128]:
    index_hnsw.hnsw.efSearch = ef
    t0 = time.time()
    h_scores, h_indices = index_hnsw.search(QUERY_VECS, k=TOP_K)
    avg_lat = (time.time() - t0) / N_QUERY * 1000
    correct = sum(len(set(gt) & set(h)) for gt, h in zip(gt_indices, h_indices))
    recall  = correct / (N_QUERY * TOP_K)
    print(f"  {ef:<10} {recall:>10.3f}  {avg_lat:>11.3f} ms")

index_hnsw.hnsw.efSearch = ef_search  # reset


# ══════════════════════════════════════════════════════════════════════════════
# 6. IndexPQ — Product Quantization (Memory Compression)
# ══════════════════════════════════════════════════════════════════════════════

section("6. IndexPQ — Product Quantization")
print("""
  Product Quantization compresses vectors by splitting each into M
  sub-vectors and quantizing each sub-vector independently.

  Parameters:
    M   — number of sub-quantizers (DIM must be divisible by M)
    nbits — bits per sub-quantizer (8 = 256 centroids per sub-space)

  Compression ratio:
    Original : DIM × 4 bytes per vector
    PQ       : M × (nbits/8) bytes per vector
    Example  : 128-dim → 512 bytes  vs  16-byte PQ = 32× compression

  Trade-off:
    More compression → lower recall, faster search, less memory
""")

M_pq   = 16   # DIM=128, so each sub-vector is 128/16=8 dims
nbits  = 8    # 256 centroids per sub-space

index_pq = faiss.IndexPQ(DIM, M_pq, nbits, faiss.METRIC_INNER_PRODUCT)

t = time.time()
index_pq.train(CORPUS_VECS)
index_pq.add(CORPUS_VECS)
print(f"  Training + add time : {elapsed_ms(t)}")

orig_bytes = N_CORPUS * DIM * 4
pq_bytes   = N_CORPUS * M_pq * (nbits // 8)
print(f"\n  Index type        : IndexPQ")
print(f"  Sub-quantizers M  : {M_pq}")
print(f"  Bits per sub-q    : {nbits}")
print(f"  Original memory   : {orig_bytes / 1024:.1f} KB")
print(f"  PQ memory         : {pq_bytes / 1024:.1f} KB")
print(f"  Compression ratio : {orig_bytes / pq_bytes:.1f}×")

q = QUERY_VECS[:1]
t = time.time()
scores, indices = index_pq.search(q, k=TOP_K)
print(f"\n  Query latency     : {elapsed_ms(t)}")

# Recall check
pq_s, pq_i = index_pq.search(QUERY_VECS, k=TOP_K)
correct = sum(len(set(gt) & set(pq)) for gt, pq in zip(gt_indices, pq_i))
recall  = correct / (N_QUERY * TOP_K)
print(f"  Recall@{TOP_K} (vs flat) : {recall:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. IndexIVFPQ — IVF + PQ (Production Standard for 1B+ Vectors)
# ══════════════════════════════════════════════════════════════════════════════

section("7. IndexIVFPQ — IVF + Product Quantization")
print("""
  Combines coarse quantization (IVF) with fine quantization (PQ).
  Industry standard for billion-scale vector search.

  Workflow:
    1. IVF partitions space into nlist cells (coarse quantization)
    2. PQ compresses each vector within its cell (fine quantization)
    3. At search, only nprobe cells are scanned using compressed vectors

  Used by: Facebook, Pinterest, Spotify for recommendation at scale.
""")

nlist_ivfpq = 50
M_ivfpq     = 16
nbits_ivfpq = 8
nprobe_ivfpq = 10

quantizer_ivfpq = faiss.IndexFlatIP(DIM)
index_ivfpq     = faiss.IndexIVFPQ(
    quantizer_ivfpq, DIM, nlist_ivfpq, M_ivfpq, nbits_ivfpq,
    faiss.METRIC_INNER_PRODUCT
)

t = time.time()
index_ivfpq.train(CORPUS_VECS)
index_ivfpq.add(CORPUS_VECS)
index_ivfpq.nprobe = nprobe_ivfpq
print(f"  Training + add time : {elapsed_ms(t)}")

orig_bytes  = N_CORPUS * DIM * 4
ivfpq_bytes = N_CORPUS * M_ivfpq

print(f"\n  Index type        : IndexIVFPQ")
print(f"  nlist             : {nlist_ivfpq}")
print(f"  nprobe            : {nprobe_ivfpq}")
print(f"  PQ sub-quantizers : {M_ivfpq}  (nbits={nbits_ivfpq})")
print(f"  Original memory   : {orig_bytes / 1024:.1f} KB")
print(f"  IVFPQ memory      : {ivfpq_bytes / 1024:.1f} KB")
print(f"  Compression ratio : {orig_bytes / ivfpq_bytes:.1f}×")

q = QUERY_VECS[:1]
t = time.time()
scores, indices = index_ivfpq.search(q, k=TOP_K)
print(f"\n  Query latency     : {elapsed_ms(t)}")

ivfpq_s, ivfpq_i = index_ivfpq.search(QUERY_VECS, k=TOP_K)
correct = sum(len(set(gt) & set(iv)) for gt, iv in zip(gt_indices, ivfpq_i))
recall  = correct / (N_QUERY * TOP_K)
print(f"  Recall@{TOP_K} (vs flat) : {recall:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. ID MAPPING — Custom IDs
# ══════════════════════════════════════════════════════════════════════════════

section("8. ID MAPPING — Custom Document IDs")
print("""
  By default FAISS assigns sequential integer IDs (0, 1, 2...).
  IndexIDMap wraps any index to support arbitrary 64-bit integer IDs.

  Use case: Map FAISS results back to your own document IDs
  (e.g., database primary keys, UUIDs encoded as ints).
""")

base_index  = faiss.IndexFlatIP(DIM)
index_idmap = faiss.IndexIDMap(base_index)

# Custom IDs: 1001, 1002, ... 1010
n_docs      = 10
doc_vecs    = generate_vectors(n_docs, DIM, seed=7)
custom_ids  = np.array([1001, 1002, 1003, 1004, 1005,
                        2001, 2002, 2003, 3001, 3002], dtype="int64")
id_to_title = {
    1001: "Apache Spark Overview",    1002: "PySpark RDD Guide",
    1003: "Spark SQL Reference",      1004: "Spark Streaming Docs",
    1005: "Spark MLlib Tutorial",     2001: "Kafka Producer Guide",
    2002: "Kafka Consumer Groups",    2003: "Kafka Streams API",
    3001: "Airflow DAG Authoring",    3002: "Airflow Operators",
}

index_idmap.add_with_ids(doc_vecs, custom_ids)

print(f"  Indexed {index_idmap.ntotal} documents with custom IDs")
print(f"  Custom IDs: {custom_ids.tolist()}")

q = doc_vecs[:1]
scores, result_ids = index_idmap.search(q, k=5)

print(f"\n  Query: doc_id=1001 ({id_to_title[1001]})")
print(f"\n  {'Rank':<6} {'Score':>9}  {'Doc ID':>8}  {'Title'}")
print(f"  {'─'*6} {'─'*9}  {'─'*8}  {'─'*35}")
for rank, (s, doc_id) in enumerate(zip(scores[0], result_ids[0]), 1):
    title = id_to_title.get(int(doc_id), "unknown")
    print(f"  {rank:<6} {s:>9.6f}  {doc_id:>8}  {title}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. INDEX SERIALIZATION — Save & Load
# ══════════════════════════════════════════════════════════════════════════════

section("9. INDEX SERIALIZATION — Save and Load")
print("""
  FAISS indexes can be persisted to disk and reloaded without rebuilding.
  Essential for production: build once offline, serve from disk.

  Functions:
    faiss.write_index(index, path)   — save to file
    faiss.read_index(path)           — load from file
""")

save_path = os.path.join(tempfile.gettempdir(), "faiss_hnsw.index")

# Save
t = time.time()
faiss.write_index(index_hnsw, save_path)
save_time = elapsed_ms(t)
file_size = os.path.getsize(save_path) / 1024

print(f"  Saved  : {save_path}")
print(f"  Size   : {file_size:.1f} KB")
print(f"  Time   : {save_time}")

# Load
t = time.time()
loaded_index = faiss.read_index(save_path)
load_time = elapsed_ms(t)

print(f"\n  Loaded index type    : {type(loaded_index).__name__}")
print(f"  Loaded index vectors : {loaded_index.ntotal:,}")
print(f"  Load time            : {load_time}")

# Verify results match
q = QUERY_VECS[:1]
orig_s,   orig_i   = index_hnsw.search(q, k=TOP_K)
loaded_s, loaded_i = loaded_index.search(q, k=TOP_K)
match = np.array_equal(orig_i, loaded_i)
print(f"\n  Results match original index : {match} ✓")

os.remove(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 10. BATCH SEARCH
# ══════════════════════════════════════════════════════════════════════════════

section("10. BATCH SEARCH — Multiple Queries at Once")
print("""
  FAISS search() accepts a matrix of query vectors.
  Batching queries is significantly faster than looping one-by-one
  because FAISS parallelises the batch internally.
""")

batch_sizes = [1, 10, 50, 100]
print(f"  {'Batch Size':<12} {'Total Time':>12}  {'Per-Query Avg':>14}")
print(f"  {'─'*12} {'─'*12}  {'─'*14}")
for bs in batch_sizes:
    batch = QUERY_VECS[:bs]
    t = time.time()
    index_hnsw.search(batch, k=TOP_K)
    total = (time.time() - t) * 1000
    per_q = total / bs
    print(f"  {bs:<12} {total:>10.3f} ms  {per_q:>12.4f} ms")

# Batch results shape
all_scores, all_indices = index_hnsw.search(QUERY_VECS, k=TOP_K)
print(f"\n  Batch search({N_QUERY} queries, k={TOP_K}):")
print(f"    scores  shape : {all_scores.shape}")
print(f"    indices shape : {all_indices.shape}")
print(f"    scores[0]     : {all_scores[0].round(4)}")
print(f"    indices[0]    : {all_indices[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. INDEX BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

section("11. INDEX BENCHMARKING — Latency · Recall · Memory")

RUNS = 10

def benchmark(name, index, queries, gt_indices, k=TOP_K, runs=RUNS):
    """Benchmark index: avg latency, recall@k, memory estimate."""
    times = []
    for _ in range(runs):
        t = time.time()
        s, i = index.search(queries, k)
        times.append(time.time() - t)
    avg_ms = np.mean(times) * 1000
    correct = sum(len(set(gt) & set(res)) for gt, res in zip(gt_indices, i))
    recall  = correct / (len(queries) * k)
    return avg_ms, recall

indexes = [
    ("IndexFlatIP",   index_ip),
    ("IndexIVFFlat",  index_ivf),
    ("IndexHNSWFlat", index_hnsw),
    ("IndexPQ",       index_pq),
    ("IndexIVFPQ",    index_ivfpq),
]

print(f"\n  Benchmarking {N_QUERY} queries × {RUNS} runs  (dim={DIM}, corpus={N_CORPUS:,})\n")
print(f"  {'Index':<16} {'Recall@5':>10} {'Avg Lat(ms)':>13}  {'Notes'}")
print(f"  {'─'*16} {'─'*10} {'─'*13}  {'─'*38}")

notes = {
    "IndexFlatIP":   "Exact, baseline",
    "IndexIVFFlat":  f"nprobe={nprobe}",
    "IndexHNSWFlat": f"M={M}, efSearch={ef_search}",
    "IndexPQ":       f"M={M_pq}, nbits={nbits}",
    "IndexIVFPQ":    f"nlist={nlist_ivfpq}, nprobe={nprobe_ivfpq}, M={M_ivfpq}",
}

for name, idx in indexes:
    avg_ms, recall = benchmark(name, idx, QUERY_VECS, gt_indices)
    print(f"  {name:<16} {recall:>10.3f} {avg_ms:>11.3f} ms  {notes[name]}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. VECTOR STORE PATTERN — Document Store + FAISS Index
# ══════════════════════════════════════════════════════════════════════════════

section("12. VECTOR STORE PATTERN (RAG Use Case)")
print("""
  A vector store combines:
    • A FAISS index for fast similarity search
    • A document store (dict) to map vector IDs → text and metadata
  This is the core pattern used in RAG pipelines.
""")

class VectorStore:
    """Simple in-memory vector store backed by FAISS."""

    def __init__(self, dim: int):
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base)
        self.docs  = {}   # id → {text, metadata}
        self._next = 0

    def add(self, texts: list, vectors: np.ndarray, metadata: list = None) -> list:
        """Add texts with their embedding vectors. Returns assigned IDs."""
        vectors = l2_normalize(vectors.astype("float32"))
        ids     = np.arange(self._next, self._next + len(texts), dtype="int64")
        self.index.add_with_ids(vectors, ids)
        for i, (text, idx) in enumerate(zip(texts, ids)):
            self.docs[int(idx)] = {
                "text":     text,
                "metadata": metadata[i] if metadata else {}
            }
        self._next += len(texts)
        return ids.tolist()

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list:
        """Return top_k results as list of (score, doc_id, text, metadata)."""
        q = l2_normalize(query_vec.reshape(1, -1).astype("float32"))
        scores, ids = self.index.search(q, k=top_k)
        results = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id == -1:
                continue
            doc = self.docs[int(doc_id)]
            results.append((float(score), int(doc_id), doc["text"], doc["metadata"]))
        return results

    @property
    def size(self):
        return self.index.ntotal


# Populate the vector store
knowledge_base = [
    ("Apache Spark is a distributed computing engine for big data processing.",
     {"source": "wiki", "category": "data_engineering"}),
    ("Kafka is a distributed event streaming platform for real-time pipelines.",
     {"source": "wiki", "category": "data_engineering"}),
    ("Airflow schedules and monitors data pipeline workflows using DAGs.",
     {"source": "wiki", "category": "data_engineering"}),
    ("Delta Lake provides ACID transactions on top of object storage data lakes.",
     {"source": "docs", "category": "data_engineering"}),
    ("Scikit-learn provides machine learning algorithms for Python.",
     {"source": "wiki", "category": "machine_learning"}),
    ("PyTorch is a deep learning framework with dynamic computation graphs.",
     {"source": "wiki", "category": "machine_learning"}),
    ("Kubernetes orchestrates containerized applications across clusters.",
     {"source": "wiki", "category": "infrastructure"}),
    ("Terraform provisions cloud infrastructure as code across providers.",
     {"source": "docs", "category": "infrastructure"}),
]

texts     = [t for t, _ in knowledge_base]
meta      = [m for _, m in knowledge_base]
rng       = np.random.default_rng(123)
kb_vecs   = l2_normalize(rng.random((len(texts), DIM)).astype("float32"))

vs = VectorStore(dim=DIM)
ids = vs.add(texts, kb_vecs, metadata=meta)

print(f"  Vector store populated : {vs.size} documents")

# Search
query_vec = kb_vecs[1]   # use Kafka vector as query
results   = vs.search(query_vec, top_k=4)

print(f"\n  Query  : \"{texts[1]}\"")
print(f"\n  {'Rank':<5} {'Score':>8}  {'ID':>5}  {'Category':<18} {'Text'}")
print(f"  {'─'*5} {'─'*8}  {'─'*5}  {'─'*18} {'─'*45}")
for rank, (score, doc_id, text, meta) in enumerate(results, 1):
    print(f"  {rank:<5} {score:>8.4f}  {doc_id:>5}  {meta['category']:<18} {text[:45]}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
  FAISS Topics Demonstrated:
  ──────────────────────────────────────────────────────────────────
   1.  What is FAISS        Architecture, metrics, index selection guide
   2.  IndexFlatL2          Brute-force L2 / Euclidean exact search
   3.  IndexFlatIP          Brute-force inner product / cosine exact search
   4.  IndexIVFFlat         Voronoi cells + nprobe recall/speed trade-off
   5.  IndexHNSWFlat        HNSW graph ANN, efSearch recall/speed trade-off
   6.  IndexPQ              Product quantization — 32× memory compression
   7.  IndexIVFPQ           IVF + PQ combined — production at billion scale
   8.  ID Mapping           Custom 64-bit IDs with IndexIDMap
   9.  Serialization        write_index / read_index — persist to disk
  10.  Batch Search         Multi-query batching — speed comparison
  11.  Benchmarking         Recall@5, latency across all index types
  12.  Vector Store         Document store + FAISS = RAG building block
  ──────────────────────────────────────────────────────────────────
  FAISS version : {faiss.__version__}
  Corpus size   : {N_CORPUS:,} vectors  ×  {DIM} dimensions
  Library       : faiss-cpu · numpy

  Index Quick-Pick:
  ┌──────────────────┬────────────────────────────────────────────┐
  │ Corpus size      │ Recommended index                          │
  ├──────────────────┼────────────────────────────────────────────┤
  │ < 10K            │ IndexFlatIP  (exact, simple)               │
  │ 10K – 1M         │ IndexHNSWFlat  (fast, high recall)         │
  │ 1M – 100M        │ IndexIVFFlat or IndexIVFPQ                 │
  │ 100M – 1B+       │ IndexIVFPQ  (memory-efficient, fast)       │
  └──────────────────┴────────────────────────────────────────────┘
""")
