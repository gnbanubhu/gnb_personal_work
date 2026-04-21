from transformers import pipeline


# Load NER pipeline using a pre-trained model
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)


def recognize_entities(text: str) -> list[dict]:
    entities = ner_pipeline(text)
    return entities


def print_entities(text: str, entities: list[dict]) -> None:
    print(f"Text: {text}")
    print("-" * 65)
    if not entities:
        print("No entities found.")
    else:
        print(f"{'Entity':<30} {'Type':<15} {'Score':<10}")
        print(f"{'-'*30} {'-'*15} {'-'*10}")
        for entity in entities:
            print(f"{entity['word']:<30} {entity['entity_group']:<15} {entity['score']:.4f}")
    print()


if __name__ == "__main__":

    samples = [
        "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California.",
        "Elon Musk is the CEO of Tesla and SpaceX, both headquartered in the United States.",
        "Barack Obama served as the 44th President of the United States from Washington D.C.",
        "Amazon was founded by Jeff Bezos in Seattle, Washington in 1994.",
        "The Eiffel Tower is located in Paris, France and was built by Gustave Eiffel.",
    ]

    print("=" * 65)
    print("         NAMED ENTITY RECOGNITION (NER) RESULTS")
    print("=" * 65)
    print()

    for text in samples:
        entities = recognize_entities(text)
        print_entities(text, entities)

    # Entity type reference
    print("Entity Type Reference:")
    print("-" * 35)
    print("  PER  : Person")
    print("  ORG  : Organization")
    print("  LOC  : Location")
    print("  MISC : Miscellaneous")
