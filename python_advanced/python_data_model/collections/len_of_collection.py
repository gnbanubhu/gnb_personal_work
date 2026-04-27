def get_length(collection):
    return len(collection)


def iterate_collection(collection):
    for element in collection:
        print(element)


def iterate_with_index(collection):
    for index, value in enumerate(collection):
        print(f"Index: {index}, Value: {value}")


def sort_ascending(collection):
    return sorted(collection)


if __name__ == "__main__":
    sample = [40, 10, 50, 20, 30]
    length = get_length(sample)
    print(f"Length of collection {sample} is: {length}")

    print("\nIterating over collection:")
    iterate_collection(sample)

    print("\nIterating with index:")
    iterate_with_index(sample)

    print("\nSorted collection (ascending):")
    print(sort_ascending(sample))
