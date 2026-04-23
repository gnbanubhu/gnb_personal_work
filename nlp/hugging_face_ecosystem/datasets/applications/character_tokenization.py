import torch
from datasets import Dataset


def create_dataset():
    data = {
        "text": [
            "Hello World",
            "Natural Language Processing",
            "Hugging Face is awesome",
            "Character tokenization splits text into characters"
        ]
    }
    return Dataset.from_dict(data)


def build_token2idx(dataset):
    all_chars = set()
    for sample in dataset:
        all_chars.update(list(sample["text"]))
    token2idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
    return token2idx


def character_tokenize(sample):
    characters = list(sample["text"])
    return {
        "characters": characters,
        "char_count": len(characters)
    }


def apply_tokenization(dataset):
    tokenized = dataset.map(character_tokenize)
    return tokenized


def convert_to_indices(sample, token2idx):
    indices = [token2idx[char] for char in sample["characters"]]
    return {"char_indices": indices}


def apply_token2idx(dataset, token2idx):
    return dataset.map(lambda sample: convert_to_indices(sample, token2idx))


def to_one_hot(char_indices, vocab_size):
    tensor = torch.zeros(len(char_indices), vocab_size)
    for i, idx in enumerate(char_indices):
        tensor[i][idx] = 1.0
    return tensor


def print_one_hot(dataset, token2idx):
    vocab_size = len(token2idx)
    print("\n" + "=" * 60)
    print("ONE-HOT ENCODED TENSORS")
    print("=" * 60)
    for sample in dataset:
        one_hot = to_one_hot(sample["char_indices"], vocab_size)
        print(f"\nText         : {sample['text']}")
        print(f"Tensor Shape : {list(one_hot.shape)}  (chars x vocab_size)")
        print(f"One-Hot Tensor:\n{one_hot}")


def print_token2idx(token2idx):
    print("=" * 60)
    print("TOKEN TO INDEX VOCABULARY")
    print("=" * 60)
    print(f"Vocabulary size: {len(token2idx)}\n")
    for char, idx in sorted(token2idx.items(), key=lambda x: x[1]):
        display = repr(char) if char == " " else char
        print(f"  '{display}' → {idx}")


def print_results(dataset):
    print("\n" + "=" * 60)
    print("CHARACTER TOKENIZATION RESULTS")
    print("=" * 60)
    for sample in dataset:
        print(f"\nText         : {sample['text']}")
        print(f"Characters   : {sample['characters']}")
        print(f"Char Indices : {sample['char_indices']}")
        print(f"Char Count   : {sample['char_count']}")


def print_overview(dataset):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total samples : {len(dataset)}")
    print(f"Features      : {dataset.features}")


def main():
    dataset = create_dataset()
    print_overview(dataset)

    token2idx = build_token2idx(dataset)
    print_token2idx(token2idx)

    tokenized_dataset = apply_tokenization(dataset)
    tokenized_dataset = apply_token2idx(tokenized_dataset, token2idx)
    print_results(tokenized_dataset)
    print_one_hot(tokenized_dataset, token2idx)


if __name__ == "__main__":
    main()
