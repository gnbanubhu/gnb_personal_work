def tokenize_by_whitespace(text):
    return text.split()


def tokenize_by_delimiter(text, delimiter):
    return text.split(delimiter)


def tokenize_into_characters(text):
    return list(text)


def tokenize_into_words_and_punctuation(text):
    tokens = []
    current = []
    for char in text:
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if char.strip():
                tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


if __name__ == "__main__":
    sentence = "Hello, Python! Welcome to tokenization."

    print("--- Tokenize by Whitespace ---")
    print(tokenize_by_whitespace(sentence))

    print("\n--- Tokenize by Delimiter (,) ---")
    print(tokenize_by_delimiter(sentence, ","))

    print("\n--- Tokenize into Characters ---")
    print(tokenize_into_characters(sentence))

    print("\n--- Tokenize into Words and Punctuation ---")
    print(tokenize_into_words_and_punctuation(sentence))
