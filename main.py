import json
from src.llm_client import LLMClient
from src.model import Model
from src.evaluator import Evaluator
from src.mutator import Mutator
from src.merger import Merger
from src.gepa_optimizer import GepaOptimizer
from src.prompts import original_prompt


def load_sentences(file_path: str) -> list[str]:
    """Load sentences from JSON file."""
    with open(file_path, 'r') as f:
        sentences = json.load(f)
    return sentences


def main():
    # Load datasets
    print("Loading datasets...")
    train_sentences = load_sentences('data/PII_train.json')
    val_sentences = load_sentences('data/PII_dev.json')[:10]  # Use only first 10 for testing

    print(f"Train set: {len(train_sentences)} sentences")
    print(f"Validation set: {len(val_sentences)} sentences")

    # Initialize LLM client (using gpt-4o-mini for all components)
    print("\nInitializing LLM client...")
    llm_client = LLMClient(model="gpt-4o-mini")

    # Initialize components
    print("Initializing components...")
    model = Model(llm_client)
    evaluator = Evaluator(model, llm_client)
    mutator = Mutator(llm_client)
    merger = Merger(llm_client)

    # Initialize optimizer
    optimizer = GepaOptimizer(max_merges=3, minibatch_size=5)

    # Run optimization
    print(f"\nStarting optimization with base prompt...")
    print("="*80)

    best_prompt = optimizer.optimize(
        base_prompt=original_prompt,
        train_sentences=train_sentences,
        val_sentences=val_sentences,
        evaluator=evaluator,
        mutator=mutator,
        merger=merger,
        rollouts_budget=5,
    )

    print("\n" + "="*80)
    print("Optimization complete!")
    print("\nBest prompt found:")
    print("-"*80)
    print(best_prompt)
    print("-"*80)

    # Save best prompt
    with open('best_prompt.txt', 'w') as f:
        f.write(best_prompt)
    print("\nBest prompt saved to best_prompt.txt")


if __name__ == "__main__":
    main()
