"""
generative_ai/generative_ai_google.py
---------------------------------------
A sample Generative AI application using the Google Gemini API.

Demonstrates:
  1. Basic text generation (single-turn)
  2. System prompt customization
  3. Multi-turn conversation (chat session)
  4. Streaming response
  5. Structured output (JSON)
  6. Multimodal input (text + image description)

Setup:
    pip install google-generativeai
    export GOOGLE_API_KEY="your-api-key-here"

Usage:
    python generative_ai_google.py
"""

import os
import json
import google.generativeai as genai

# ── Initialize client ─────────────────────────────────────────────────────────
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your-key'"
    )

genai.configure(api_key=api_key)
MODEL = "gemini-1.5-pro"


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def sub(title):
    print(f"\n  ── {title}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC TEXT GENERATION (Single-turn)
# ══════════════════════════════════════════════════════════════════════════════
section("1. BASIC TEXT GENERATION")

model    = genai.GenerativeModel(MODEL)
response = model.generate_content("Explain what Generative AI is in 3 sentences.")

print(f"\n  Prompt   : Explain what Generative AI is in 3 sentences.")
print(f"\n  Response :\n  {response.text}")
print(f"\n  Usage    : prompt_tokens={response.usage_metadata.prompt_token_count}, "
      f"output_tokens={response.usage_metadata.candidates_token_count}, "
      f"total_tokens={response.usage_metadata.total_token_count}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM PROMPT CUSTOMIZATION
# ══════════════════════════════════════════════════════════════════════════════
section("2. SYSTEM PROMPT CUSTOMIZATION")

model_with_system = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction=(
        "You are an expert PySpark engineer. "
        "Answer concisely with code examples when relevant. "
        "Always mention performance implications."
    )
)

response = model_with_system.generate_content(
    "When should I use RDDs over DataFrames in PySpark?"
)

print(f"\n  System   : Expert PySpark engineer")
print(f"\n  Prompt   : When should I use RDDs over DataFrames in PySpark?")
print(f"\n  Response :\n")
for line in response.text.splitlines():
    print(f"  {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-TURN CONVERSATION (Chat Session)
# ══════════════════════════════════════════════════════════════════════════════
section("3. MULTI-TURN CONVERSATION")

chat_model   = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="You are a helpful AI assistant. Be concise."
)
chat_session = chat_model.start_chat(history=[])

turns = [
    "My name is Naresh and I am learning PySpark.",
    "What topic should I learn after RDDs?",
    "Can you give me one practical project idea using what I've learned so far?"
]

for turn in turns:
    reply = chat_session.send_message(turn)
    print(f"\n  User      : {turn}")
    print(f"  Assistant : {reply.text}")

print(f"\n  Total turns in conversation: {len(chat_session.history) // 2}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
section("4. STREAMING RESPONSE")

print(f"\n  Prompt   : List 5 use cases of Generative AI in data engineering.")
print(f"\n  Response (streaming):\n")

stream = model.generate_content(
    "List 5 use cases of Generative AI in data engineering.",
    stream=True
)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRUCTURED OUTPUT (JSON)
# ══════════════════════════════════════════════════════════════════════════════
section("5. STRUCTURED OUTPUT (JSON)")

json_model = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction=(
        "You are a data extraction assistant. "
        "Always respond with valid JSON only — no explanation, no markdown, no code fences."
    )
)

response = json_model.generate_content(
    "Extract the following information as JSON with keys: "
    "name, role, skills (list), years_experience.\n\n"
    "Text: Naresh is a Senior Data Engineer with 8 years of experience. "
    "He is proficient in PySpark, Python, SQL, Kafka, and Airflow."
)

raw_json = response.text.strip()
parsed   = json.loads(raw_json)

print(f"\n  Input text:")
print(f"    Naresh is a Senior Data Engineer with 8 years of experience.")
print(f"    He is proficient in PySpark, Python, SQL, Kafka, and Airflow.")
print(f"\n  Extracted JSON:")
print(json.dumps(parsed, indent=4))


# ══════════════════════════════════════════════════════════════════════════════
# 6. GENERATION CONFIG (Temperature, Top-K, Top-P, Max Tokens)
# ══════════════════════════════════════════════════════════════════════════════
section("6. GENERATION CONFIG")

generation_config = genai.types.GenerationConfig(
    temperature=0.2,       # lower = more deterministic
    top_p=0.9,             # nucleus sampling
    top_k=40,              # top-k sampling
    max_output_tokens=128, # limit response length
)

model_configured = genai.GenerativeModel(
    model_name=MODEL,
    generation_config=generation_config
)

response = model_configured.generate_content(
    "In one sentence, what is Apache Spark?"
)

print(f"\n  Config   : temperature=0.2, top_p=0.9, top_k=40, max_tokens=128")
print(f"\n  Prompt   : In one sentence, what is Apache Spark?")
print(f"\n  Response : {response.text}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAFETY SETTINGS INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
section("7. SAFETY RATINGS INSPECTION")

response = model.generate_content("Tell me about data pipeline best practices.")

print(f"\n  Prompt   : Tell me about data pipeline best practices.")
print(f"\n  Safety Ratings:")
for rating in response.candidates[0].safety_ratings:
    print(f"    {rating.category.name:<45} : {rating.probability.name}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
print("""
  Feature                    Description
  ────────────────────────────────────────────────────────────
  Basic Generation           Single-turn prompt → response
  System Prompt              Custom persona via system_instruction
  Multi-turn Conversation    Stateful chat session with history
  Streaming                  Real-time chunk-by-chunk output
  Structured Output (JSON)   Extracts structured data from text
  Generation Config          Control temperature, top_p, top_k
  Safety Ratings             Inspect content safety classifications
  ────────────────────────────────────────────────────────────
  Model used : gemini-1.5-pro  (Google)
  SDK        : google-generativeai  (pip install google-generativeai)
""")
