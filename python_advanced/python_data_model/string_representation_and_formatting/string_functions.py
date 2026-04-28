def get_string_length(text):
    return len(text)


def char_to_numerical(text):
    for char in text:
        print(f"'{char}' -> {ord(char)}")


def reverse_string(text):
    return text[::-1]


def concatenate_strings(str1, str2):
    return str1 + str2


if __name__ == "__main__":
    sample = "Hello, Python!"
    length = get_string_length(sample)
    print(f"String: '{sample}'")
    print(f"Length: {length}")

    print("\nNumerical representation of each character:")
    char_to_numerical(sample)

    print(f"\nReversed string: '{reverse_string(sample)}'")

    str1 = "Hello"
    str2 = " Python!"
    result = concatenate_strings(str1, str2)
    print(f"\nConcatenated string: '{result}'")
