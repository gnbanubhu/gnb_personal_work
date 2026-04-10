"""
generative_ai/genai.py
-----------------------
A sample Generative AI application using the Anthropic Claude API.

Demonstrates:
  1. Basic text generation (single-turn)
  2. Multi-turn conversation (chat history)
  3. System prompt customization
  4. Streaming response
  5. Structured output (JSON)

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-api-key-here"

Usage:
    python genai.py
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


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC TEXT GENERATION (Single-turn)
# ══════════════════════════════════════════════════════════════════════════════
section("1. BASIC TEXT GENERATION")

response = client.messages.create(
    model=MODEL,
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Explain what Generative AI is in 3 sentences."}
    ]
)

print(f"\n  Prompt  : Explain what Generative AI is in 3 sentences.")
print(f"\n  Response:\n  {response.content[0].text}")
print(f"\n  Usage   : input_tokens={response.usage.input_tokens}, "
      f"output_tokens={response.usage.output_tokens}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM PROMPT CUSTOMIZATION
# ══════════════════════════════════════════════════════════════════════════════
section("2. SYSTEM PROMPT CUSTOMIZATION")

response = client.messages.create(
    model=MODEL,
    max_tokens=256,
    system=(
        "You are an expert PySpark engineer. "
        "Answer concisely with code examples when relevant. "
        "Always mention performance implications."
    ),
    messages=[
        {"role": "user", "content": "When should I use RDDs over DataFrames in PySpark?"}
    ]
)

print(f"\n  System  : Expert PySpark engineer")
print(f"\n  Prompt  : When should I use RDDs over DataFrames in PySpark?")
print(f"\n  Response:\n")
for line in response.content[0].text.splitlines():
    print(f"  {line}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-TURN CONVERSATION (Chat History)
# ══════════════════════════════════════════════════════════════════════════════
section("3. MULTI-TURN CONVERSATION")

conversation = []

def chat(user_message):
    """Send a message and maintain conversation history."""
    conversation.append({"role": "user", "content": user_message})
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system="You are a helpful AI assistant. Be concise.",
        messages=conversation
    )
    assistant_reply = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply

turns = [
    "My name is Naresh and I am learning PySpark.",
    "What topic should I learn after RDDs?",
    "Can you give me one practical project idea using what I've learned so far?"
]

for turn in turns:
    reply = chat(turn)
    print(f"\n  User      : {turn}")
    print(f"  Assistant : {reply}")

print(f"\n  Total turns in conversation: {len(conversation) // 2}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
section("4. STREAMING RESPONSE")

print(f"\n  Prompt  : List 5 use cases of Generative AI in data engineering.")
print(f"\n  Response (streaming):\n")

with client.messages.stream(
    model=MODEL,
    max_tokens=300,
    messages=[
        {"role": "user", "content": "List 5 use cases of Generative AI in data engineering."}
    ]
) as stream:
    for text_chunk in stream.text_stream:
        print(text_chunk, end="", flush=True)

print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRUCTURED OUTPUT (JSON)
# ══════════════════════════════════════════════════════════════════════════════
section("5. STRUCTURED OUTPUT (JSON)")

response = client.messages.create(
    model=MODEL,
    max_tokens=512,
    system=(
        "You are a data extraction assistant. "
        "Always respond with valid JSON only — no explanation, no markdown."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                "Extract the following information as JSON with keys: "
                "name, role, skills (list), years_experience.\n\n"
                "Text: Naresh is a Senior Data Engineer with 8 years of experience. "
                "He is proficient in PySpark, Python, SQL, Kafka, and Airflow."
            )
        }
    ]
)

raw_json = response.content[0].text.strip()
parsed   = json.loads(raw_json)

print(f"\n  Input text:")
print(f"    Naresh is a Senior Data Engineer with 8 years of experience.")
print(f"    He is proficient in PySpark, Python, SQL, Kafka, and Airflow.")
print(f"\n  Extracted JSON:")
print(json.dumps(parsed, indent=4))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
print("""
  Feature                  Description
  ──────────────────────────────────────────────────────────
  Basic Generation         Single-turn prompt → response
  System Prompt            Custom persona / instruction set
  Multi-turn Conversation  Maintains chat history across turns
  Streaming                Real-time token-by-token output
  Structured Output (JSON) Extracts structured data from text
  ──────────────────────────────────────────────────────────
  Model used : claude-opus-4-6  (Anthropic)
  SDK        : anthropic (pip install anthropic)
""")
