"""
TOKENIZATION IN LARGE LANGUAGE MODELS
=======================================
Tokenization is the process of converting raw text into tokens (integer IDs)
that a language model can process. This file covers the key tokenization
concepts used in modern LLMs.

Topics Covered:
  1.  Word Tokenization
  2.  Character Tokenization
  3.  Subword Tokenization (BPE)
  4.  WordPiece Tokenization (BERT-style)
  5.  SentencePiece Tokenization
  6.  Hugging Face AutoTokenizer
  7.  Encoding and Decoding Text
  8.  Special Tokens
  9.  Padding and Truncation
  10. Attention Masks
  11. Tokenizing Multiple Sentences (Batching)
  12. Tokenizer Vocabulary and Vocab Size
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import re
from collections import Counter, defaultdict
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    GPT2Tokenizer,
)


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────
# TOPIC 1 — WORD TOKENIZATION
# Split text into words by whitespace and punctuation.
# Simple but struggles with unknown words and morphological variants.
# ─────────────────────────────────────────────────────────────────
def word_tokenization():
    section("TOPIC 1 — WORD TOKENIZATION")

    text = "Natural Language Processing is amazing! Let's explore tokenization."

    # Simple whitespace split
    tokens_whitespace = text.split()
    print(f"Text             : {text}")
    print(f"Whitespace split : {tokens_whitespace}")

    # Better: split on whitespace and punctuation
    tokens_regex = re.findall(r"\b\w+\b", text.lower())
    print(f"Regex split      : {tokens_regex}")

    # Vocabulary from word tokens
    vocab = sorted(set(tokens_regex))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    print(f"\nVocabulary       : {vocab}")
    print(f"Word → Index     : {word2idx}")

    # Encode sentence as integer IDs
    encoded = [word2idx[w] for w in tokens_regex if w in word2idx]
    print(f"Encoded          : {encoded}")

    print("\nLimitation: words like 'running', 'ran', 'runs' are treated "
          "as completely different tokens — large vocabulary, OOV problem.")


# ─────────────────────────────────────────────────────────────────
# TOPIC 2 — CHARACTER TOKENIZATION
# Split text into individual characters.
# No unknown tokens, but sequences become very long.
# ─────────────────────────────────────────────────────────────────
def character_tokenization():
    section("TOPIC 2 — CHARACTER TOKENIZATION")

    text = "Hello, NLP!"

    # Character-level tokens
    tokens = list(text)
    print(f"Text         : {text}")
    print(f"Char tokens  : {tokens}")
    print(f"Sequence len : {len(tokens)}")

    # Build character vocabulary
    vocab = sorted(set(tokens))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}

    encoded = [char2idx[ch] for ch in tokens]
    decoded = "".join(idx2char[idx] for idx in encoded)

    print(f"Vocabulary   : {vocab}")
    print(f"Encoded      : {encoded}")
    print(f"Decoded      : {decoded}")

    print("\nLimitation: very long sequences, model must learn to combine "
          "characters into meaningful units — slow and inefficient.")


# ─────────────────────────────────────────────────────────────────
# TOPIC 3 — SUBWORD TOKENIZATION (BPE — Byte Pair Encoding)
# Used by GPT-2, GPT-4, LLaMA, Mistral.
# Starts with characters, iteratively merges the most frequent pairs.
# Balances vocabulary size and sequence length.
# ─────────────────────────────────────────────────────────────────
def bpe_tokenization():
    section("TOPIC 3 — SUBWORD TOKENIZATION (BPE)")

    print("BPE Algorithm Overview:")
    print("  1. Start with character-level vocabulary")
    print("  2. Count all adjacent symbol pairs in the corpus")
    print("  3. Merge the most frequent pair into a new symbol")
    print("  4. Repeat until desired vocabulary size is reached")
    print()

    # Simple BPE demonstration on a small corpus
    corpus = ["low", "low", "low", "lower", "lower", "newest",
              "newest", "newest", "newest", "widest", "widest"]

    # Initialize: split each word into characters + end-of-word marker
    vocab = Counter()
    for word in corpus:
        vocab[" ".join(list(word)) + " </w>"] += 1

    print("Initial character vocabulary:")
    for word, count in vocab.items():
        print(f"  {word!r:30s} → {count}")

    def get_pairs(vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(pair, vocab):
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in vocab:
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    print("\nBPE Merge Steps (top 5):")
    for i in range(5):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        print(f"  Merge {i+1}: {best_pair} → {''.join(best_pair)!r}  "
              f"(freq={pairs[best_pair]})")

    print("\nFinal vocabulary after BPE merges:")
    for word, count in vocab.items():
        print(f"  {word!r:30s} → {count}")

    print("\nKey insight: 'low', 'lower' → share 'low' subword token. "
          "'newest', 'widest' → share 'est' subword token.")


# ─────────────────────────────────────────────────────────────────
# TOPIC 4 — WORDPIECE TOKENIZATION (BERT-style)
# Similar to BPE but merges based on likelihood, not frequency.
# Prefix '##' indicates continuation of a word.
# Used by: BERT, DistilBERT, RoBERTa, ALBERT
# ─────────────────────────────────────────────────────────────────
def wordpiece_tokenization():
    section("TOPIC 4 — WORDPIECE TOKENIZATION (BERT-style)")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    words = ["tokenization", "unbelievable", "preprocessing",
             "NLP", "transformer", "embeddings"]

    print(f"{'Word':<20} {'WordPiece Tokens'}")
    print("-" * 55)
    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"{word:<20} {tokens}")

    print()
    text = "The tokenizer splits unknown words into subword pieces."
    tokens = tokenizer.tokenize(text)
    print(f"Text   : {text}")
    print(f"Tokens : {tokens}")
    print()
    print("Note: '##' prefix = continuation subword (part of previous word)")
    print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")


# ─────────────────────────────────────────────────────────────────
# TOPIC 5 — SENTENCEPIECE TOKENIZATION
# Language-independent, treats text as raw Unicode.
# Used by: T5, LLaMA, Gemma, PaLM, Mistral
# Does NOT require pre-tokenization (handles spaces as special chars)
# ─────────────────────────────────────────────────────────────────
def sentencepiece_tokenization():
    section("TOPIC 5 — SENTENCEPIECE TOKENIZATION")

    # Use LLaMA tokenizer (SentencePiece-based) via AutoTokenizer
    # Using a locally available SentencePiece-based model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    texts = [
        "Hello, how are you?",
        "Tokenization is fundamental to NLP.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(f"Model : google/flan-t5-small (SentencePiece tokenizer)")
    print(f"Vocab size: {tokenizer.vocab_size:,}\n")

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids    = tokenizer.encode(text)
        print(f"Text   : {text}")
        print(f"Tokens : {tokens}")
        print(f"IDs    : {ids}")
        print()

    print("Note: '▁' (underscore) marks the start of a new word (space prefix).")
    print("SentencePiece works directly on raw text without pre-tokenization.")


# ─────────────────────────────────────────────────────────────────
# TOPIC 6 — HUGGING FACE AUTOTOKENIZER
# Automatically loads the correct tokenizer for any model.
# ─────────────────────────────────────────────────────────────────
def autotokenizer_demo():
    section("TOPIC 6 — HUGGING FACE AUTOTOKENIZER")

    models = [
        ("bert-base-uncased",       "BERT     — WordPiece"),
        ("gpt2",                    "GPT-2    — BPE"),
        ("google/flan-t5-small",    "T5       — SentencePiece"),
    ]

    text = "Hugging Face makes NLP easy and accessible."

    for model_name, description in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens    = tokenizer.tokenize(text)
        ids       = tokenizer.encode(text)
        print(f"\nModel      : {description}")
        print(f"Tokens     : {tokens}")
        print(f"IDs        : {ids}")
        print(f"Num tokens : {len(tokens)}")


# ─────────────────────────────────────────────────────────────────
# TOPIC 7 — ENCODING AND DECODING TEXT
# encode()  → text → token IDs
# decode()  → token IDs → text
# ─────────────────────────────────────────────────────────────────
def encoding_decoding():
    section("TOPIC 7 — ENCODING AND DECODING TEXT")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "NLP transforms how computers understand language."

    # Encode: text → token IDs
    input_ids = tokenizer.encode(text)
    print(f"Original text : {text}")
    print(f"Encoded IDs   : {input_ids}")

    # Decode: token IDs → text
    decoded_text = tokenizer.decode(input_ids)
    print(f"Decoded text  : {decoded_text}")

    # Decode without special tokens
    decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"Clean decoded : {decoded_clean}")

    # Decode individual tokens
    print(f"\nToken ID → Token:")
    for token_id in input_ids:
        token = tokenizer.decode([token_id])
        print(f"  {token_id:6d} → {token!r}")


# ─────────────────────────────────────────────────────────────────
# TOPIC 8 — SPECIAL TOKENS
# Models use special tokens to mark boundaries, padding, unknown words.
# ─────────────────────────────────────────────────────────────────
def special_tokens():
    section("TOPIC 8 — SPECIAL TOKENS")

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("BERT Special Tokens:")
    print(f"  [CLS]  (cls_token)  : {bert_tokenizer.cls_token!r:10s} "
          f"ID={bert_tokenizer.cls_token_id}")
    print(f"  [SEP]  (sep_token)  : {bert_tokenizer.sep_token!r:10s} "
          f"ID={bert_tokenizer.sep_token_id}")
    print(f"  [PAD]  (pad_token)  : {bert_tokenizer.pad_token!r:10s} "
          f"ID={bert_tokenizer.pad_token_id}")
    print(f"  [UNK]  (unk_token)  : {bert_tokenizer.unk_token!r:10s} "
          f"ID={bert_tokenizer.unk_token_id}")
    print(f"  [MASK] (mask_token) : {bert_tokenizer.mask_token!r:10s} "
          f"ID={bert_tokenizer.mask_token_id}")

    print("\nGPT-2 Special Tokens:")
    print(f"  EOS token   : {gpt2_tokenizer.eos_token!r:10s} "
          f"ID={gpt2_tokenizer.eos_token_id}")
    print(f"  BOS token   : {gpt2_tokenizer.bos_token!r:10s} "
          f"ID={gpt2_tokenizer.bos_token_id}")

    # BERT encodes with special tokens automatically
    text = "Hello world"
    encoded = bert_tokenizer.encode(text)
    print(f"\nBERT encodes '{text}' → {encoded}")
    print(f"  [CLS]=101, tokens, [SEP]=102 are added automatically")

    print("\nSpecial Token Purposes:")
    print("  [CLS]  — classification token, represents whole sequence (BERT)")
    print("  [SEP]  — separates two sentences in BERT (sentence pair tasks)")
    print("  [PAD]  — pads shorter sequences to match batch length")
    print("  [UNK]  — replaces tokens not in the vocabulary")
    print("  [MASK] — masked token for Masked Language Modeling (BERT training)")
    print("  <EOS>  — end of sequence marker (GPT-style models)")


# ─────────────────────────────────────────────────────────────────
# TOPIC 9 — PADDING AND TRUNCATION
# Batches require all sequences to be the same length.
# Padding adds [PAD] tokens; truncation cuts long sequences.
# ─────────────────────────────────────────────────────────────────
def padding_and_truncation():
    section("TOPIC 9 — PADDING AND TRUNCATION")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = [
        "Short sentence.",
        "This is a medium length sentence for demonstration.",
        "This is a much longer sentence that contains many more words "
        "and will demonstrate how truncation works when sequences exceed "
        "the maximum allowed length.",
    ]

    print("Without padding/truncation:")
    for s in sentences:
        ids = tokenizer.encode(s)
        print(f"  len={len(ids):3d}  {ids[:8]}...")

    print("\nWith padding='max_length', max_length=20, truncation=True:")
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )
    print(f"  input_ids shape      : {list(encoded['input_ids'].shape)}")
    print(f"  attention_mask shape : {list(encoded['attention_mask'].shape)}")

    for i, s in enumerate(sentences):
        ids  = encoded["input_ids"][i].tolist()
        mask = encoded["attention_mask"][i].tolist()
        print(f"\n  Sentence {i+1}: '{s[:40]}...' " if len(s) > 40 else
              f"\n  Sentence {i+1}: '{s}'")
        print(f"    IDs  : {ids}")
        print(f"    Mask : {mask}")

    print("\nPadding strategies:")
    print("  'longest'    — pad to the longest sequence in the batch")
    print("  'max_length' — pad to the specified max_length")
    print("  True / False — enable/disable padding")


# ─────────────────────────────────────────────────────────────────
# TOPIC 10 — ATTENTION MASKS
# 1 = real token (attend to it), 0 = padding token (ignore it)
# Prevents the model from attending to meaningless pad tokens.
# ─────────────────────────────────────────────────────────────────
def attention_masks():
    section("TOPIC 10 — ATTENTION MASKS")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = ["I love NLP.", "Transformers changed everything."]

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt"
    )

    print(f"input_ids      :\n{encoded['input_ids']}\n")
    print(f"attention_mask :\n{encoded['attention_mask']}\n")

    for i, s in enumerate(sentences):
        ids  = encoded["input_ids"][i].tolist()
        mask = encoded["attention_mask"][i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"Sentence {i+1}: '{s}'")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Mask   : {mask}")
        real_tokens = sum(mask)
        pad_tokens  = len(mask) - real_tokens
        print(f"  Real tokens: {real_tokens}, Padding tokens: {pad_tokens}\n")

    print("Attention mask usage in model:")
    print("  model(input_ids=..., attention_mask=encoded['attention_mask'])")
    print("  The mask tells the model which positions to attend to.")


# ─────────────────────────────────────────────────────────────────
# TOPIC 11 — TOKENIZING MULTIPLE SENTENCES (BATCHING)
# Efficiently tokenize a batch of texts in one call.
# ─────────────────────────────────────────────────────────────────
def batch_tokenization():
    section("TOPIC 11 — BATCH TOKENIZATION")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = [
        "The weather today is sunny.",
        "Machine learning is transforming industries.",
        "Tokenization is the first step in NLP pipelines.",
        "BERT uses WordPiece tokenization.",
        "GPT models use Byte Pair Encoding.",
    ]

    # Tokenize entire batch at once
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    print(f"Number of sentences : {len(texts)}")
    print(f"input_ids shape     : {list(batch['input_ids'].shape)}")
    print(f"  → {len(texts)} sentences × {batch['input_ids'].shape[1]} tokens each\n")

    print(f"{'#':<4} {'Text':<45} {'Tokens':>6} {'Padding':>8}")
    print("-" * 68)
    for i, text in enumerate(texts):
        mask        = batch["attention_mask"][i].tolist()
        real_tokens = sum(mask)
        pad_tokens  = len(mask) - real_tokens
        print(f"{i+1:<4} {text:<45} {real_tokens:>6} {pad_tokens:>8}")

    # Sentence pair tokenization (BERT NSP-style)
    print("\nSentence Pair Tokenization:")
    sent_a = "How are you?"
    sent_b = "I am doing well, thank you!"
    pair   = tokenizer(sent_a, sent_b, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(pair["input_ids"][0])
    print(f"  Sentence A : {sent_a}")
    print(f"  Sentence B : {sent_b}")
    print(f"  Tokens     : {tokens}")
    print(f"  [SEP] separates the two sentences")


# ─────────────────────────────────────────────────────────────────
# TOPIC 12 — VOCABULARY AND VOCAB SIZE
# The set of all tokens the tokenizer knows.
# Larger vocab = fewer tokens per sentence but more memory.
# ─────────────────────────────────────────────────────────────────
def vocabulary_and_vocab_size():
    section("TOPIC 12 — VOCABULARY AND VOCAB SIZE")

    models = [
        ("bert-base-uncased",    "BERT base uncased"),
        ("gpt2",                 "GPT-2"),
        ("google/flan-t5-small", "T5 / Flan-T5"),
    ]

    print(f"{'Model':<25} {'Tokenizer Type':<15} {'Vocab Size':>12}")
    print("-" * 55)
    for model_name, description in models:
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        vocab_size = tokenizer.vocab_size
        tok_type   = type(tokenizer).__name__.replace("Tokenizer", "")
        print(f"{description:<25} {tok_type:<15} {vocab_size:>12,}")

    # Explore vocabulary
    print("\nSample BERT vocabulary entries:")
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab    = bert_tok.get_vocab()

    sample_tokens = ["hello", "world", "nlp", "##ing", "##tion",
                     "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
    for token in sample_tokens:
        idx = vocab.get(token, "NOT FOUND")
        print(f"  {token:<12} → ID {idx}")

    print("\nVocab size trade-offs:")
    print("  Small vocab (8K)   — fewer params, OOV issues, longer sequences")
    print("  Medium vocab (32K) — good balance (LLaMA, Mistral)")
    print("  Large vocab (128K) — shorter sequences, more memory (LLaMA 3)")
    print("  Typical range      — 30K–130K tokens for modern LLMs")

    # Token count comparison
    print("\nToken count for the same text across models:")
    text = "The transformer architecture revolutionized natural language processing."
    for model_name, description in models:
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        tokens     = tokenizer.tokenize(text)
        print(f"  {description:<25} → {len(tokens):3d} tokens : {tokens}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   TOKENIZATION IN LARGE LANGUAGE MODELS")
    print("=" * 60)

    word_tokenization()
    character_tokenization()
    bpe_tokenization()
    wordpiece_tokenization()
    sentencepiece_tokenization()
    autotokenizer_demo()
    encoding_decoding()
    special_tokens()
    padding_and_truncation()
    attention_masks()
    batch_tokenization()
    vocabulary_and_vocab_size()

    print("\n" + "=" * 60)
    print("   TOKENIZATION SUMMARY")
    print("=" * 60)
    print("""
  Method          Used By                  Pros / Cons
  ─────────────────────────────────────────────────────────────
  Word            Early NLP, NLTK          Simple / OOV problem
  Character       Char-RNN                 No OOV / Very long seqs
  BPE             GPT-2, GPT-4, LLaMA     Balanced / Most popular
  WordPiece       BERT, DistilBERT         Efficient / ##prefix
  SentencePiece   T5, LLaMA, Gemma        Language-agnostic / ▁prefix
  Tiktoken        GPT-4, Claude            Fast BPE / Proprietary

  Key concepts:
    encode()        → text  → token IDs
    decode()        → IDs   → text
    Special tokens  → [CLS], [SEP], [PAD], [UNK], [MASK], <EOS>
    Padding         → align sequences to same length in a batch
    Attention mask  → 1=real token, 0=padding (model ignores padding)
    Vocab size      → 30K–130K for modern LLMs
    Subword tokens  → balance between word-level and char-level
    """)
