"""
llm/embeddings/embeddings.py
-----------------------------
Comprehensive guide to Text Embeddings using sentence-transformers.

Topics covered:
  1.  What Are Embeddings          — concept, dimensionality, vector space
  2.  Generate Embeddings          — encode sentences into dense vectors
  3.  Cosine Similarity            — measure semantic closeness
  4.  Semantic Textual Similarity  — rank sentence pairs by similarity
  5.  Document Retrieval           — find most relevant docs for a query
  6.  Clustering                   — group similar texts with K-Means
  7.  Nearest Neighbour Search     — find top-K similar items
  8.  Embedding Arithmetic         — vector analogy (king - man + woman)
  9.  Cross-Domain Similarity      — compare code, questions, answers
  10. Embedding Dimensions         — PCA compression and visualization data
  11. Batch Encoding               — efficient large-scale encoding
  12. Model Comparison             — compare two embedding models

Model: all-MiniLM-L6-v2  (384-dim, fast, high quality)
       paraphrase-MiniLM-L3-v2 (384-dim, smaller/faster)

Usage:
    pip install sentence-transformers scikit-learn numpy
    python embeddings.py
"""

import numpy as np
from sentence_transformers        import SentenceTransformer, util
from sklearn.cluster              import KMeans
from sklearn.decomposition        import PCA
from sklearn.metrics.pairwise     import cosine_similarity
from sklearn.preprocessing        import normalize


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

section("LOADING EMBEDDING MODELS")

MODEL_NAME   = "all-MiniLM-L6-v2"
MODEL_NAME_2 = "paraphrase-MiniLM-L3-v2"

print(f"\n  Loading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print(f"  Embedding dimensions : {model.get_sentence_embedding_dimension()}")
print(f"  Max sequence length  : {model.max_seq_length}")

print(f"\n  Loading {MODEL_NAME_2} (for model comparison) ...")
model2 = SentenceTransformer(MODEL_NAME_2)
print(f"  Embedding dimensions : {model2.get_sentence_embedding_dimension()}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. WHAT ARE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

section("1. WHAT ARE EMBEDDINGS")
print("""
  An embedding is a dense, fixed-size numerical vector that represents
  the semantic meaning of text in a high-dimensional space.

  Key properties:
    • Semantically similar texts → vectors that are close together
    • Distance metric: cosine similarity (angle), euclidean (magnitude)
    • Dimensionality: 384 (MiniLM), 768 (BERT), 1536 (OpenAI ada-002)
    • Produced by encoder-only transformer models (BERT, RoBERTa, etc.)

  Applications:
    ✦ Semantic search        — find docs by meaning, not keywords
    ✦ Clustering             — group similar documents
    ✦ Classification         — embed + classify with lightweight model
    ✦ Recommendation         — find similar items
    ✦ RAG                    — retrieve relevant context for LLMs
    ✦ Deduplication          — detect near-duplicate content
    ✦ Anomaly detection      — outlier vectors in embedding space
""")

sentence = "Apache Spark is a distributed data processing framework."
embedding = model.encode(sentence)

print(f"  Input sentence   : {sentence}")
print(f"  Embedding shape  : {embedding.shape}")
print(f"  Embedding dtype  : {embedding.dtype}")
print(f"  First 8 dims     : {embedding[:8].round(4)}")
print(f"  Vector norm      : {np.linalg.norm(embedding):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

section("2. GENERATE EMBEDDINGS")

sentences = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text.",
    "Computer vision allows machines to interpret images.",
    "Reinforcement learning trains agents through rewards and penalties.",
]

embeddings = model.encode(sentences)
print(f"\n  Encoded {len(sentences)} sentences")
print(f"  Matrix shape     : {embeddings.shape}  (sentences × dimensions)")
print(f"  Memory usage     : {embeddings.nbytes / 1024:.1f} KB")

print(f"\n  {'#':<4} {'Sentence':<52} {'Norm':>6}")
print(f"  {'─'*4} {'─'*52} {'─'*6}")
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    print(f"  {i+1:<4} {sent[:50]:<52} {np.linalg.norm(emb):>6.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

section("3. COSINE SIMILARITY")
print("""
  Cosine similarity measures the cosine of the angle between two vectors.
    • Range: -1.0 (opposite) to +1.0 (identical)
    • 0.9+  → Near-duplicate / paraphrase
    • 0.7–0.9 → Highly related
    • 0.5–0.7 → Somewhat related
    • < 0.5  → Unrelated
""")

pairs = [
    ("I love programming in Python.",
     "Python is my favourite coding language."),
    ("The stock market crashed today.",
     "Financial markets experienced a sharp decline."),
    ("I love programming in Python.",
     "The weather is sunny today."),
    ("Apache Kafka is a streaming platform.",
     "Kafka handles real-time event streams at scale."),
    ("My cat is sleeping on the sofa.",
     "The database query returned 500 rows."),
]

print(f"  {'Sentence A':<40} {'Sentence B':<40} {'Similarity':>10}")
print(f"  {'─'*40} {'─'*40} {'─'*10}")
for a, b in pairs:
    ea, eb = model.encode([a, b])
    score  = cosine_score(ea, eb)
    print(f"  {a[:38]:<40} {b[:38]:<40} {score:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEMANTIC TEXTUAL SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

section("4. SEMANTIC TEXTUAL SIMILARITY (STS)")

reference = "How do I connect to a PostgreSQL database in Python?"

candidates = [
    "What is the Python library to access PostgreSQL?",
    "Steps to establish a Postgres connection using psycopg2.",
    "How to query a MySQL database with SQLAlchemy?",
    "Python list comprehension examples.",
    "Configure a PostgreSQL connection pool in Python.",
    "What is the best database for time-series data?",
]

ref_emb  = model.encode(reference)
cand_emb = model.encode(candidates)
scores   = [cosine_score(ref_emb, c) for c in cand_emb]
ranked   = sorted(zip(scores, candidates), reverse=True)

print(f"\n  Reference: \"{reference}\"")
print(f"\n  {'Rank':<6} {'Score':>8}  {'Candidate'}")
print(f"  {'─'*6} {'─'*8}  {'─'*52}")
for rank, (score, cand) in enumerate(ranked, 1):
    print(f"  {rank:<6} {score:>8.4f}  {cand}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DOCUMENT RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

section("5. DOCUMENT RETRIEVAL (Semantic Search)")

corpus = [
    "Apache Spark is an open-source distributed processing system for big data.",
    "Kafka is a distributed event streaming platform for real-time data pipelines.",
    "Airflow is a workflow orchestration tool for scheduling data pipelines.",
    "Delta Lake provides ACID transactions on top of data lakes.",
    "dbt (data build tool) transforms data inside the warehouse using SQL.",
    "Kubernetes orchestrates containerized applications across clusters.",
    "Docker provides OS-level virtualization using containers.",
    "PostgreSQL is a powerful open-source relational database.",
    "Redis is an in-memory data store used for caching and message brokering.",
    "Elasticsearch is a distributed search and analytics engine.",
]

corpus_emb = model.encode(corpus, convert_to_tensor=True)

queries = [
    "Which tool handles real-time streaming data?",
    "How do I schedule and orchestrate data workflows?",
    "What provides ACID guarantees for data lakes?",
]

for query in queries:
    query_emb = model.encode(query, convert_to_tensor=True)
    hits      = util.semantic_search(query_emb, corpus_emb, top_k=3)[0]

    print(f"\n  Query : \"{query}\"")
    print(f"  {'Rank':<6} {'Score':>8}  {'Document'}")
    print(f"  {'─'*6} {'─'*8}  {'─'*55}")
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank:<6} {hit['score']:>8.4f}  {corpus[hit['corpus_id']]}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

section("6. CLUSTERING WITH K-MEANS")

docs = [
    # Data Engineering
    "Build ETL pipelines to move data from source to warehouse.",
    "Apache Spark processes large-scale data in a distributed manner.",
    "Kafka streams real-time events to downstream consumers.",
    "Airflow orchestrates complex data pipeline workflows.",
    # Machine Learning
    "Train a neural network to classify images with deep learning.",
    "Gradient boosting algorithms improve prediction accuracy.",
    "Hyperparameter tuning optimizes model performance.",
    "Cross-validation evaluates model generalization on unseen data.",
    # Cloud Infrastructure
    "AWS S3 stores objects at petabyte scale.",
    "Kubernetes manages containerized workloads across nodes.",
    "Terraform provisions cloud infrastructure as code.",
    "Azure Data Factory orchestrates cloud data integration.",
]

doc_emb   = model.encode(docs)
kmeans    = KMeans(n_clusters=3, random_state=42, n_init=10)
labels    = kmeans.fit_predict(doc_emb)

cluster_names = {0: "Cluster A", 1: "Cluster B", 2: "Cluster C"}
print(f"\n  Clustered {len(docs)} documents into 3 groups:\n")
for cluster_id in sorted(set(labels)):
    print(f"  [{cluster_names[cluster_id]}]")
    for doc, label in zip(docs, labels):
        if label == cluster_id:
            print(f"    • {doc}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. NEAREST NEIGHBOUR SEARCH (Top-K)
# ══════════════════════════════════════════════════════════════════════════════

section("7. NEAREST NEIGHBOUR SEARCH (Top-K Similar Items)")

items = [
    "Python is great for data science and machine learning.",
    "R is widely used for statistical analysis.",
    "SQL is essential for querying relational databases.",
    "Scala is the native language of Apache Spark.",
    "Java is used for building enterprise backend systems.",
    "JavaScript powers interactive web applications.",
    "Go is known for high-performance concurrent programs.",
    "Rust offers memory safety without garbage collection.",
]

item_emb = model.encode(items)
sim_matrix = cosine_similarity(item_emb)

query_item = "Python is great for data science and machine learning."
query_idx  = items.index(query_item)

print(f"\n  Query item: \"{query_item}\"")
print(f"\n  Top-3 nearest neighbours:")
print(f"  {'Rank':<6} {'Score':>8}  {'Item'}")
print(f"  {'─'*6} {'─'*8}  {'─'*52}")

scores = list(enumerate(sim_matrix[query_idx]))
scores.sort(key=lambda x: x[1], reverse=True)

rank = 1
for idx, score in scores:
    if idx == query_idx:
        continue
    print(f"  {rank:<6} {score:>8.4f}  {items[idx]}")
    rank += 1
    if rank > 3:
        break


# ══════════════════════════════════════════════════════════════════════════════
# 8. EMBEDDING ARITHMETIC (VECTOR ANALOGIES)
# ══════════════════════════════════════════════════════════════════════════════

section("8. EMBEDDING ARITHMETIC (Vector Analogies)")
print("""
  Word2Vec famously demonstrated: king - man + woman ≈ queen
  The same principle applies to sentence embeddings.

  Formula: result_vector = vec(A) - vec(B) + vec(C)
  Then find the item in the corpus closest to result_vector.
""")

analogy_corpus = [
    "Python",
    "pandas",
    "NumPy",
    "JavaScript",
    "React",
    "Vue",
    "Java",
    "Spring Boot",
    "Maven",
    "Ruby",
    "Ruby on Rails",
    "Bundler",
]

corpus_vecs = model.encode(analogy_corpus)
corpus_vecs = normalize(corpus_vecs)

def analogy(a, b, c, corpus_texts, corpus_vectors, top_k=3):
    """Find: a is to b as c is to ?"""
    va = normalize(model.encode([a]))[0]
    vb = normalize(model.encode([b]))[0]
    vc = normalize(model.encode([c]))[0]
    result = va - vb + vc
    result = result / np.linalg.norm(result)
    sims   = corpus_vectors @ result
    ranked = sorted(zip(sims, corpus_texts), reverse=True)
    # exclude input words
    exclude = {a.lower(), b.lower(), c.lower()}
    return [(s, t) for s, t in ranked if t.lower() not in exclude][:top_k]

print(f"  Analogy: 'pandas' is to 'Python' as ??? is to 'JavaScript'")
results = analogy("pandas", "Python", "JavaScript",
                  analogy_corpus, corpus_vecs)
for score, term in results:
    print(f"    → {term:<20}  (score: {score:.4f})")

print(f"\n  Analogy: 'Spring Boot' is to 'Java' as ??? is to 'Ruby'")
results2 = analogy("Spring Boot", "Java", "Ruby",
                   analogy_corpus, corpus_vecs)
for score, term in results2:
    print(f"    → {term:<20}  (score: {score:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CROSS-DOMAIN SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

section("9. CROSS-DOMAIN SIMILARITY")
print("""
  Embeddings can bridge natural language and code questions,
  matching user questions to relevant code snippets or documentation.
""")

questions = [
    "How do I read a CSV file into a DataFrame?",
    "How to filter rows where a column value is greater than 100?",
    "How do I group data and calculate the average?",
]

code_snippets = [
    "df = pd.read_csv('data.csv')",
    "df_filtered = df[df['amount'] > 100]",
    "df.groupby('category')['sales'].mean()",
    "df.sort_values('date', ascending=False)",
    "df.dropna(subset=['email'])",
]

q_emb = model.encode(questions, convert_to_tensor=True)
c_emb = model.encode(code_snippets, convert_to_tensor=True)

for question, q_vec in zip(questions, q_emb):
    hits = util.semantic_search(q_vec, c_emb, top_k=2)[0]
    print(f"\n  Question : {question}")
    for hit in hits:
        print(f"    Score {hit['score']:.4f}  →  {code_snippets[hit['corpus_id']]}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PCA DIMENSIONALITY REDUCTION
# ══════════════════════════════════════════════════════════════════════════════

section("10. PCA DIMENSIONALITY REDUCTION")
print("""
  Embeddings are high-dimensional (384-dim). PCA reduces them to 2D/3D
  for visualization and can speed up downstream tasks.

  Use cases:
    • Visualize clusters in 2D scatter plots
    • Compress embeddings to reduce memory in vector stores
    • Feature extraction before training classifiers
""")

topics = {
    "Data Engineering": [
        "ETL pipelines move data from source systems to data warehouses.",
        "Apache Spark processes terabytes of data in parallel.",
        "Data quality checks ensure accuracy and completeness.",
    ],
    "Machine Learning": [
        "Neural networks learn patterns from training data.",
        "Random forests combine multiple decision trees.",
        "Feature engineering improves model accuracy.",
    ],
    "DevOps": [
        "CI/CD pipelines automate code testing and deployment.",
        "Docker containers package applications with dependencies.",
        "Kubernetes scales containerized services automatically.",
    ],
}

all_texts  = []
all_labels = []
for label, texts in topics.items():
    all_texts.extend(texts)
    all_labels.extend([label] * len(texts))

all_emb   = model.encode(all_texts)
pca       = PCA(n_components=2)
reduced   = pca.fit_transform(all_emb)

print(f"\n  Original dimensions  : {all_emb.shape[1]}")
print(f"  Reduced dimensions   : {reduced.shape[1]}")
print(f"  Explained variance   : {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"\n  2D coordinates per document:")
print(f"  {'Label':<20} {'Text':<45} {'PC1':>8} {'PC2':>8}")
print(f"  {'─'*20} {'─'*45} {'─'*8} {'─'*8}")
for text, label, (x, y) in zip(all_texts, all_labels, reduced):
    print(f"  {label:<20} {text[:43]:<45} {x:>8.4f} {y:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. BATCH ENCODING
# ══════════════════════════════════════════════════════════════════════════════

section("11. BATCH ENCODING (Efficient Large-Scale)")

import time

large_corpus = [f"Document {i}: This is a sample text about topic {i % 10}."
                for i in range(500)]

# Single encode (no batching hint)
t = time.time()
emb_single = model.encode(large_corpus, batch_size=32, show_progress_bar=False)
t_single = time.time() - t

# Larger batch size
t = time.time()
emb_batch = model.encode(large_corpus, batch_size=128, show_progress_bar=False)
t_batch = time.time() - t

print(f"\n  Corpus size            : {len(large_corpus)} documents")
print(f"  Embedding shape        : {emb_single.shape}")
print(f"\n  batch_size=32   time   : {t_single*1000:.1f} ms")
print(f"  batch_size=128  time   : {t_batch*1000:.1f} ms")
print(f"  Speedup                : {t_single/t_batch:.2f}x")

print("""
  Batch encoding tips:
    • batch_size=32–128 is typically optimal on CPU
    • Use convert_to_tensor=True for GPU tensor output
    • Use show_progress_bar=True for long-running jobs
    • normalize_embeddings=True pre-normalizes for cosine search
""")


# ══════════════════════════════════════════════════════════════════════════════
# 12. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

section("12. MODEL COMPARISON")
print(f"""
  Comparing:
    Model 1: {MODEL_NAME}   (384-dim, higher quality)
    Model 2: {MODEL_NAME_2} (384-dim, faster/smaller)
""")

test_pairs = [
    ("A data pipeline ingests raw data from multiple sources.",
     "ETL processes extract, transform, and load data into a warehouse."),
    ("Python is a versatile programming language.",
     "JavaScript runs in the browser for frontend development."),
    ("The quick brown fox jumps over the lazy dog.",
     "A fast red fox leaps above a sleeping hound."),
]

print(f"  {'Pair':<4} {'Model 1 Score':>14} {'Model 2 Score':>14}  {'Sentences'}")
print(f"  {'─'*4} {'─'*14} {'─'*14}  {'─'*50}")
for i, (a, b) in enumerate(test_pairs, 1):
    e1a, e1b = model.encode([a, b])
    e2a, e2b = model2.encode([a, b])
    s1 = cosine_score(e1a, e1b)
    s2 = cosine_score(e2a, e2b)
    print(f"  {i:<4} {s1:>14.4f} {s2:>14.4f}  {a[:25]}... ↔ {b[:20]}...")

print(f"""
  Model selection guide:
  ┌──────────────────────────────┬────────────────────────────────────────┐
  │ Model                        │ Best For                               │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ all-MiniLM-L6-v2             │ General semantic search, high quality  │
  │ all-mpnet-base-v2            │ Highest quality, slower                │
  │ paraphrase-MiniLM-L3-v2      │ Speed-critical applications            │
  │ multi-qa-MiniLM-L6-cos-v1    │ Q&A and document retrieval             │
  │ all-distilroberta-v1         │ Balanced speed and quality             │
  │ OpenAI text-embedding-3-small│ Cloud API, multilingual, 1536-dim      │
  └──────────────────────────────┴────────────────────────────────────────┘
""")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print(f"""
  Embeddings Topics Demonstrated:
  ──────────────────────────────────────────────────────────────
   1.  What Are Embeddings     Dense vector representation of text meaning
   2.  Generate Embeddings     SentenceTransformer.encode(), shape, norm
   3.  Cosine Similarity       Angle-based semantic closeness (-1 to +1)
   4.  Semantic Similarity     Rank sentence pairs by meaning proximity
   5.  Document Retrieval      util.semantic_search() for corpus search
   6.  Clustering              K-Means on embedding matrix
   7.  Nearest Neighbour       Top-K most similar items in a collection
   8.  Embedding Arithmetic    Vector analogy: A - B + C ≈ D
   9.  Cross-Domain Similarity Match natural language to code snippets
  10.  PCA Reduction           384-dim → 2-dim for visualization
  11.  Batch Encoding          Throughput optimization with batch_size
  12.  Model Comparison        Quality vs speed trade-offs across models
  ──────────────────────────────────────────────────────────────
  Model  : {MODEL_NAME}
  Library: sentence-transformers · scikit-learn · numpy
""")
