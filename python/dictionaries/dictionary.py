person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}


def get_keys(dictionary):
    for key in dictionary.keys():
        print(key)


def get_values(dictionary):
    for value in dictionary.values():
        print(value)


def get_items(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")


print(person)
print(f"Name: {person['name']}")
print(f"Age: {person['age']}")
print(f"City: {person['city']}")

print("\nKeys:")
get_keys(person)

print("\nValues:")
get_values(person)

print("\nItems:")
get_items(person)
