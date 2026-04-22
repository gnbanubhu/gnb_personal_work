from transformers import AutoTokenizer


def load_tokenizer(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer: {model_name}")
    print(f"Vocabulary size : {tokenizer.vocab_size}\n")
    return tokenizer


def tokenize_text(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    print("=" * 60)
    print("TOKENS")
    print("=" * 60)
    print(f"Text   : {text}")
    print(f"Tokens : {tokens}")
    print(f"Count  : {len(tokens)}")
    return tokens


def encode_text(tokenizer, text):
    encoded = tokenizer(text, return_tensors=None)
    print("\n" + "=" * 60)
    print("ENCODED OUTPUT")
    print("=" * 60)
    print(f"Input IDs      : {encoded['input_ids']}")
    print(f"Attention Mask : {encoded['attention_mask']}")
    return encoded


def decode_text(tokenizer, input_ids):
    decoded = tokenizer.decode(input_ids)
    print("\n" + "=" * 60)
    print("DECODED TEXT")
    print("=" * 60)
    print(f"Decoded : {decoded}")
    return decoded


def tokenize_batch(tokenizer, texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors=None)
    print("\n" + "=" * 60)
    print("BATCH TOKENIZATION")
    print("=" * 60)
    for i, text in enumerate(texts):
        print(f"\nText {i + 1}         : {text}")
        print(f"Input IDs      : {encoded['input_ids'][i]}")
        print(f"Attention Mask : {encoded['attention_mask'][i]}")
    return encoded


def print_special_tokens(tokenizer):
    print("\n" + "=" * 60)
    print("SPECIAL TOKENS")
    print("=" * 60)
    print(f"CLS token : {tokenizer.cls_token} (id: {tokenizer.cls_token_id})")
    print(f"SEP token : {tokenizer.sep_token} (id: {tokenizer.sep_token_id})")
    print(f"PAD token : {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"UNK token : {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    print(f"MASK token: {tokenizer.mask_token} (id: {tokenizer.mask_token_id})")


def main():
    tokenizer = load_tokenizer("bert-base-uncased")

    text = "Hugging Face transformers make NLP tasks easy!"
    tokenize_text(tokenizer, text)
    encoded = encode_text(tokenizer, text)
    decode_text(tokenizer, encoded["input_ids"])

    texts = [
        "I love natural language processing.",
        "Tokenizers split text into smaller units."
    ]
    tokenize_batch(tokenizer, texts)

    print_special_tokens(tokenizer)


if __name__ == "__main__":
    main()
