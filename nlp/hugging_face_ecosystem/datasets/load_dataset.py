from datasets import load_dataset


def main():
    dataset = load_dataset("imdb")

    print(f"Dataset: {dataset}")
    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    print("\nFirst training sample:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
