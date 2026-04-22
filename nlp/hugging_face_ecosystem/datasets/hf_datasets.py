import os
from huggingface_hub import list_datasets, login


def main():
    # Authenticate using HF_TOKEN env variable or prompt login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("No HF_TOKEN found. Run: huggingface-cli login  OR  set HF_TOKEN env variable.")
        print("Get your token at: https://huggingface.co/settings/tokens\n")

    print("Fetching datasets from Hugging Face Hub (limited to 100)...\n")

    available_datasets = list(list_datasets(limit=100, token=hf_token))

    print(f"Total datasets fetched: {len(available_datasets)}\n")

    print("List of available datasets:")
    for dataset in available_datasets:
        print(f"  - {dataset.id}")


if __name__ == "__main__":
    main()
