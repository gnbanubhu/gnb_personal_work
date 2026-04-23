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


def character_tokenize(sample):
    characters = list(sample["text"])
    return {
        "characters": characters,
        "char_count": len(characters)
    }


def apply_tokenization(dataset):
    tokenized = dataset.map(character_tokenize)
    return tokenized


def print_results(dataset):
    print("=" * 60)
    print("CHARACTER TOKENIZATION RESULTS")
    print("=" * 60)
    for sample in dataset:
        print(f"\nText       : {sample['text']}")
        print(f"Characters : {sample['characters']}")
        print(f"Char Count : {sample['char_count']}")


def print_overview(dataset):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total samples : {len(dataset)}")
    print(f"Features      : {dataset.features}")


def main():
    dataset = create_dataset()
    print_overview(dataset)

    tokenized_dataset = apply_tokenization(dataset)
    print_results(tokenized_dataset)


if __name__ == "__main__":
    main()
