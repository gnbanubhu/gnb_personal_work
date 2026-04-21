from transformers import pipeline


# Load text generation pipeline using GPT-2
generator = pipeline(
    "text-generation",
    model="gpt2"
)


def generate_text(prompt: str, max_length: int = 100, num_sequences: int = 2) -> list[str]:
    results = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_sequences,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return [r["generated_text"] for r in results]


def print_generated(prompt: str, sequences: list[str]) -> None:
    print(f"Prompt: {prompt}")
    print("-" * 70)
    for i, text in enumerate(sequences, 1):
        print(f"Generated {i}:\n{text}")
        print("-" * 70)
    print()


if __name__ == "__main__":

    prompts = [
        "Artificial intelligence is revolutionizing the world by",
        "The future of data engineering depends on",
        "Machine learning models are trained to",
        "Large language models like GPT have changed the way",
    ]

    print("=" * 70)
    print("                TEXT GENERATION RESULTS (GPT-2)")
    print("=" * 70)
    print()

    for prompt in prompts:
        sequences = generate_text(prompt, max_length=80, num_sequences=2)
        print_generated(prompt, sequences)
