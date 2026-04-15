import faiss
import numpy as np

# ─────────────────────────────────────────────────────────────
# Sample 1: Exact Search with IndexFlatL2
# Use case: Small datasets where accuracy matters most
# ─────────────────────────────────────────────────────────────

def sample_1_exact_search():
    print("=" * 60)
    print("Sample 1: Exact Search with IndexFlatL2")
    print("=" * 60)

    dimension = 128
    num_vectors = 1000

    # Simulate document embeddings
    vectors = np.random.rand(num_vectors, dimension).astype("float32")

    # Build exact L2 index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    print(f"Total vectors in index: {index.ntotal}")

    # Search for top-3 nearest neighbors
    query = np.random.rand(1, dimension).astype("float32")
    distances, indices = index.search(query, k=3)

    print(f"Top-3 nearest indices : {indices[0]}")
    print(f"Top-3 distances       : {distances[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# Sample 2: Approximate Search with IndexIVFFlat
# Use case: Medium/large datasets — faster but approximate
# ─────────────────────────────────────────────────────────────

def sample_2_approximate_search():
    print("=" * 60)
    print("Sample 2: Approximate Search with IndexIVFFlat")
    print("=" * 60)

    dimension = 128
    num_vectors = 50000
    nlist = 100  # Number of Voronoi cells (clusters)

    vectors = np.random.rand(num_vectors, dimension).astype("float32")

    # IVFFlat requires training to partition the vector space
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(vectors)
    index.add(vectors)

    print(f"Total vectors in index: {index.ntotal}")

    # nprobe: how many clusters to search (higher = more accurate, slower)
    index.nprobe = 10

    query = np.random.rand(1, dimension).astype("float32")
    distances, indices = index.search(query, k=5)

    print(f"Top-5 nearest indices : {indices[0]}")
    print(f"Top-5 distances       : {distances[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# Sample 3: Inner Product Search (Cosine Similarity)
# Use case: Semantic search where direction matters, not magnitude
# ─────────────────────────────────────────────────────────────

def sample_3_cosine_similarity_search():
    print("=" * 60)
    print("Sample 3: Inner Product Search (Cosine Similarity)")
    print("=" * 60)

    dimension = 128
    num_vectors = 1000

    vectors = np.random.rand(num_vectors, dimension).astype("float32")

    # Normalize vectors to unit length for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)  # Inner Product index
    index.add(vectors)

    query = np.random.rand(1, dimension).astype("float32")
    faiss.normalize_L2(query)  # Normalize query too

    scores, indices = index.search(query, k=5)

    print(f"Top-5 similar indices : {indices[0]}")
    print(f"Cosine similarity     : {scores[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# Sample 4: Save and Load a FAISS Index
# Use case: Persist index to disk and reload without recomputing
# ─────────────────────────────────────────────────────────────

def sample_4_save_and_load_index():
    print("=" * 60)
    print("Sample 4: Save and Load a FAISS Index")
    print("=" * 60)

    dimension = 64
    num_vectors = 500

    vectors = np.random.rand(num_vectors, dimension).astype("float32")

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save index to disk
    faiss.write_index(index, "/tmp/faiss_sample.index")
    print("Index saved to /tmp/faiss_sample.index")

    # Load index from disk
    loaded_index = faiss.read_index("/tmp/faiss_sample.index")
    print(f"Index loaded. Total vectors: {loaded_index.ntotal}")

    query = np.random.rand(1, dimension).astype("float32")
    distances, indices = loaded_index.search(query, k=3)

    print(f"Top-3 nearest indices : {indices[0]}")
    print(f"Top-3 distances       : {distances[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# Sample 5: ID Mapping with IndexIDMap
# Use case: Map FAISS internal IDs to your own document IDs
# ─────────────────────────────────────────────────────────────

def sample_5_custom_id_mapping():
    print("=" * 60)
    print("Sample 5: Custom ID Mapping with IndexIDMap")
    print("=" * 60)

    dimension = 64
    num_vectors = 10

    vectors = np.random.rand(num_vectors, dimension).astype("float32")

    # Assign custom IDs (e.g., database row IDs or document IDs)
    custom_ids = np.array([101, 202, 303, 404, 505, 606, 707, 808, 909, 1010], dtype="int64")

    base_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(vectors, custom_ids)

    print(f"Total vectors in index: {index.ntotal}")

    query = np.random.rand(1, dimension).astype("float32")
    distances, ids = index.search(query, k=3)

    print(f"Top-3 custom IDs  : {ids[0]}")
    print(f"Top-3 distances   : {distances[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# Sample 6: Batch Query Search
# Use case: Search multiple queries at once efficiently
# ─────────────────────────────────────────────────────────────

def sample_6_batch_query_search():
    print("=" * 60)
    print("Sample 6: Batch Query Search")
    print("=" * 60)

    dimension = 128
    num_vectors = 10000
    num_queries = 5

    vectors = np.random.rand(num_vectors, dimension).astype("float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Multiple queries at once
    queries = np.random.rand(num_queries, dimension).astype("float32")
    distances, indices = index.search(queries, k=3)

    for i in range(num_queries):
        print(f"Query {i + 1} → Top-3 indices: {indices[i]}  Distances: {distances[i]}")
    print()


# ─────────────────────────────────────────────────────────────
# Run all samples
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_1_exact_search()
    sample_2_approximate_search()
    sample_3_cosine_similarity_search()
    sample_4_save_and_load_index()
    sample_5_custom_id_mapping()
    sample_6_batch_query_search()
