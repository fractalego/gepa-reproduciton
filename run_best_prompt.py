import json
from src.llm_client import LLMClient
from src.model import Model


def load_sentences(file_path: str) -> list[str]:
    """Load sentences from JSON file."""
    with open(file_path, 'r') as f:
        sentences = json.load(f)
    return sentences


def main():
    # Load best prompt
    print("Loading best prompt...")
    with open('best_prompt.txt', 'r') as f:
        best_prompt = f.read()

    print(f"\nBest prompt:\n{best_prompt}\n")
    print("=" * 80)

    # Initialize LLM client and model
    print("\nInitializing LLM client...")
    llm_client = LLMClient(model="gpt-4o-mini")
    model = Model(llm_client)

    # Load validation sentences (use first 10 for demo)
    print("Loading validation sentences...")
    val_sentences = load_sentences('data/PII_dev.json')[:10]
    print(f"Running on {len(val_sentences)} validation sentences\n")
    print("=" * 80)

    # Run best prompt on each sentence
    for i, sentence in enumerate(val_sentences):
        print(f"\nSentence {i + 1}:")
        print(f"  Original: {sentence}")

        # Run PII stripper
        sanitized = model.run(best_prompt, sentence)
        print(f"  Sanitized: {sanitized}")


if __name__ == "__main__":
    main()
