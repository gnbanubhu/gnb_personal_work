from datasets import load_dataset


def load_data():
    dataset = load_dataset("imdb", split="train[:100]")
    return dataset


def set_pandas_format(dataset):
    dataset.set_format(type="pandas")
    return dataset


def get_dataframe(dataset):
    df = dataset[:]
    return df


def print_overview(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")


def print_sample(df, n=5):
    print("\n" + "=" * 60)
    print(f"FIRST {n} ROWS")
    print("=" * 60)
    print(df.head(n).to_string())


def print_label_distribution(df):
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    print(df["label"].value_counts().to_string())


def print_basic_stats(df):
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df.describe())


def reset_format(dataset):
    dataset.reset_format()
    print("\n" + "=" * 60)
    print("FORMAT RESET TO DEFAULT")
    print("=" * 60)
    print(f"First sample after reset:\n{dataset[0]}")


def main():
    dataset = load_data()
    print(f"Loaded dataset: {dataset}\n")

    dataset = set_pandas_format(dataset)
    df = get_dataframe(dataset)

    print_overview(df)
    print_sample(df)
    print_label_distribution(df)
    print_basic_stats(df)
    reset_format(dataset)


if __name__ == "__main__":
    main()
