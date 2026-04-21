from transformers import pipeline


# Load the text classification pipeline using a pre-trained model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def classify_text(text: str) -> dict:
    result = classifier(text)
    return result[0]


def classify_batch(texts: list[str]) -> list[dict]:
    results = classifier(texts)
    return results


if __name__ == "__main__":
    # Single text classification
    sample_text = "I absolutely love this product! It works perfectly."
    result = classify_text(sample_text)
    print(f"Text    : {sample_text}")
    print(f"Label   : {result['label']}")
    print(f"Score   : {result['score']:.4f}")

    print()

    # Batch text classification
    sample_texts = [
        "This movie was fantastic and very entertaining.",
        "The service was terrible and I am very disappointed.",
        "The product is okay, nothing special.",
        "I would highly recommend this to everyone!",
        "Worst experience I have ever had.",
    ]

    print("Batch Classification Results:")
    print("-" * 55)
    batch_results = classify_batch(sample_texts)
    for text, res in zip(sample_texts, batch_results):
        print(f"Text  : {text}")
        print(f"Label : {res['label']}  |  Score: {res['score']:.4f}")
        print("-" * 55)
