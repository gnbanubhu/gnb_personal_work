import tensorflow as tf


def build_vocab(texts):
    all_chars = sorted(set("".join(texts)))
    token2idx = {char: idx for idx, char in enumerate(all_chars)}
    idx2token = {idx: char for char, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text):
    characters = list(text)
    print("=" * 60)
    print("TOKENIZATION")
    print("=" * 60)
    print(f"Text       : {text}")
    print(f"Characters : {characters}")
    print(f"Char Count : {len(characters)}")
    return characters


def convert_to_numerical(characters, token2idx):
    indices = [token2idx[char] for char in characters]
    tensor = tf.constant(indices, dtype=tf.int32)
    print("\n" + "=" * 60)
    print("CHARACTER TO NUMERICAL CONVERSION")
    print("=" * 60)
    print(f"Characters : {characters}")
    print(f"Indices    : {indices}")
    print(f"Tensor     : {tensor.numpy()}")
    return tensor


def one_hot_encode(tensor, vocab_size, idx2token):
    one_hot = tf.one_hot(tensor, depth=vocab_size)
    print("\n" + "=" * 60)
    print("ONE-HOT ENCODING")
    print("=" * 60)
    print(f"Vocab Size   : {vocab_size}")
    print(f"Tensor Shape : {one_hot.shape}  (chars x vocab_size)")
    print(f"\n{'Char':<6} {'Index':<6} One-Hot Vector")
    print("-" * 60)
    for i, idx in enumerate(tensor.numpy()):
        char = repr(idx2token[idx]) if idx2token[idx] == " " else idx2token[idx]
        print(f"'{char}'  {idx:<6} {one_hot[i].numpy()}")
    return one_hot


def main():
    texts = [
        "Hello World",
        "Natural Language Processing",
        "TensorFlow makes deep learning easy"
    ]

    token2idx, idx2token = build_vocab(texts)
    print("=" * 60)
    print("VOCABULARY")
    print("=" * 60)
    print(f"Vocab Size : {len(token2idx)}")
    for char, idx in token2idx.items():
        display = repr(char) if char == " " else char
        print(f"  '{display}' → {idx}")

    vocab_size = len(token2idx)

    for text in texts:
        print("\n" + "*" * 60)
        characters = tokenize(text)
        tensor = convert_to_numerical(characters, token2idx)
        one_hot_encode(tensor, vocab_size, idx2token)


if __name__ == "__main__":
    main()
