from transformers import AutoTokenizer


def word_tokenization(text: str) -> list[str]:
    return text.split()


def demonstrate_tokenizer(tokenizer_name: str, text: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoding = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    input_ids = encoding["input_ids"][0].tolist()
    decoded = tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=True)

    print(f"Tokenizer     : {tokenizer_name}")
    print(f"Text          : {text}")
    print(f"Tokens        : {tokens}")
    print(f"Token IDs     : {input_ids}")
    print(f"Token Count   : {len(tokens)}")
    print(f"Decoded       : {decoded}")
    print("-" * 70)
    print()


def demonstrate_special_tokens(tokenizer_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer     : {tokenizer_name}")
    print(f"PAD token     : {tokenizer.pad_token}  (ID: {tokenizer.pad_token_id})")
    print(f"UNK token     : {tokenizer.unk_token}  (ID: {tokenizer.unk_token_id})")
    print(f"CLS token     : {tokenizer.cls_token}  (ID: {tokenizer.cls_token_id})")
    print(f"SEP token     : {tokenizer.sep_token}  (ID: {tokenizer.sep_token_id})")
    print(f"Vocab Size    : {tokenizer.vocab_size}")
    print("-" * 70)
    print()


def demonstrate_padding_truncation(tokenizer_name: str, texts: list[str]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )
    print(f"Tokenizer     : {tokenizer_name}")
    print(f"Texts         : {texts}")
    print(f"Input IDs     :\n{encoding['input_ids']}")
    print(f"Attention Mask:\n{encoding['attention_mask']}")
    print("-" * 70)
    print()


if __name__ == "__main__":

    sample_text = "Machine learning is transforming the field of artificial intelligence."

    print("=" * 70)
    print("                  TEXT TOKENIZATION RESULTS")
    print("=" * 70)
    print()

    # Word-level tokenization
    print("1. WORD-LEVEL TOKENIZATION")
    print("-" * 70)
    words = word_tokenization(sample_text)
    print(f"Text   : {sample_text}")
    print(f"Tokens : {words}")
    print(f"Count  : {len(words)}")
    print("-" * 70)
    print()

    # BERT Tokenizer (WordPiece)
    print("2. BERT TOKENIZER (WordPiece)")
    print("-" * 70)
    demonstrate_tokenizer("bert-base-uncased", sample_text)

    # GPT-2 Tokenizer (Byte-Pair Encoding)
    print("3. GPT-2 TOKENIZER (Byte-Pair Encoding - BPE)")
    print("-" * 70)
    demonstrate_tokenizer("gpt2", sample_text)

    # Special Tokens
    print("4. SPECIAL TOKENS — BERT")
    print("-" * 70)
    demonstrate_special_tokens("bert-base-uncased")

    # Padding and Truncation
    print("5. PADDING AND TRUNCATION")
    print("-" * 70)
    batch_texts = [
        "AI is amazing.",
        "Deep learning models require large amounts of training data.",
        "NLP enables machines to understand human language.",
    ]
    demonstrate_padding_truncation("bert-base-uncased", batch_texts)
