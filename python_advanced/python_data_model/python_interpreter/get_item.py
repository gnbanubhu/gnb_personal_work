def get_item(collection, index):
    return collection.__getitem__(index)


if __name__ == "__main__":
    sample = [10, 20, 30, 40, 50]
    item = get_item(sample, 2)
    print(f"Item at index 2: {item}")
