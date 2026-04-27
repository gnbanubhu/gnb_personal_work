# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Personal learning workspace covering AI/ML, data engineering, Python, and related technologies. Each top-level folder is an independent learning domain — there is no single application entry point.

## Running Scripts

Every Python file is a self-contained, runnable demonstration. Execute directly:

```bash
python agentic_ai/agent.py
python generative_ai/anthropic.py
python llm/rag_application.py
python pyspark/spark_session.py
```

No build step required. Scripts run top-to-bottom and print structured output to stdout.

## Environment Variables

AI scripts that call external APIs require:

```bash
export ANTHROPIC_API_KEY="your-key"    # used by agentic_ai/, generative_ai/, llm/
```

OpenAI and Google scripts follow the same pattern using `OPENAI_API_KEY` and `GOOGLE_API_KEY`.

## Testing

Tests exist only for the PySpark module. Run them from the repo root:

```bash
# Unit tests
pytest pyspark/test/test_spark_session.py -v

# Integration tests (exercises full Spark execution engine)
pytest pyspark/test/integration_test_spark_session.py -v

# Run a single test
pytest pyspark/test/test_spark_session.py::TestCreateSparkSession::test_app_name -v
```

Test files import from `pyspark/` by inserting the parent directory into `sys.path`.

## Key Architecture Patterns

### Anthropic Agentic Loop (`agentic_ai/agent.py`)
The agent pattern follows: send query + tool definitions → if `stop_reason == "tool_use"`, execute the requested tool and append the result as a `tool_result` message → repeat until `stop_reason == "end_turn"`. Tools are defined as dicts with `name`, `description`, and `input_schema` (JSON Schema). A dispatcher dict maps tool names to handler functions.

### RAG Pipeline (`llm/rag_application.py`)
Eight-stage pipeline: Document loading → text chunking (sliding window with overlap) → embedding via `sentence-transformers` (`all-MiniLM-L6-v2`) → FAISS `IndexFlatIP` for cosine similarity → top-K retrieval with a minimum score threshold → context injection into an Anthropic prompt → conversational multi-turn Q&A → source citation. Scaling guidance: `IndexFlatIP` for < 10K chunks, `IndexHNSWFlat` for medium, `IndexIVFPQ` for large.

### PySpark Tests
`pytest` fixtures use `scope="module"` so a single `SparkSession` is shared across all tests in a file and stopped via `yield`. The session runs in `local[*]` mode with log level set to `ERROR`.

### Anthropic Client Initialization
All scripts that use the Anthropic API follow this pattern:
```python
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(...)
client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-opus-4-6"
```

## Domain Map

| Folder | Contents |
|---|---|
| `agentic_ai/` | Agentic AI implementations, tool use, agentic loop demos, course notes |
| `ai_engineering/` | AI engineering lifecycle concepts (RAG, fine-tuning, evaluation, prompting) |
| `generative_ai/` | GenAI demos using Anthropic, OpenAI, and Google APIs |
| `llm/` | LLM implementations: RAG, embeddings, vector DBs, tokenization, prompt engineering |
| `anthropic_courses/` | Notes from Anthropic learning courses |
| `pyspark/` | PySpark learning: RDDs, DataFrames, Spark SQL, Streaming, MLlib, GraphX |
| `nlp/` | NLP implementations: tokenization, classification, summarization, NER, Q&A |
| `deep_learning/` | Keras, PyTorch, TensorFlow samples |
| `large_language_models/` | LLM concepts: embeddings, fine-tuning, RAG, prompt engineering |
| `generative_deep_learning/` | GANs, VAEs, diffusion models, transformers |
| `data_engineering/` | Data engineering lifecycle concepts and pipeline samples |
| `airflow/` | Airflow DAG samples |
| `kafka/` | Kafka producer/consumer samples |
| `python/` + `python_advanced/` | Python fundamentals through advanced patterns |
| `algorithms/` + `data_structures/` | Algorithm and data structure implementations |
| `machine_learning/` | ML with scikit-learn |
| `sql/` | SQL and MySQL connector samples |
| `cloud/` | AWS, Azure, GCP concept notes |
