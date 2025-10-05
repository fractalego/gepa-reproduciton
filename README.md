# GEPA: Prompt Optimization for PII Stripping

This is an implementation of the GEPA algorithm for optimizing prompts to strip Personally Identifiable Information (PII) from text. GEPA uses evolutionary strategies combined with Pareto-based multi-objective optimization to evolve prompts through mutation and merging operations. The algorithm evaluates prompt performance across multiple validation examples, maintaining a Pareto frontier of non-dominated solutions, and uses LLM-based reflection to generate improved prompt variants. This implementation is based on the original GEPA framework available at [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa/tree/main) and the paper ["Gradient-Free Optimization of Prompt Ensembles via Pareto Fronts"](https://arxiv.org/pdf/2507.19457).
The data in the `data/` folder is synthetic and generated for demonstration purposes only. It does not contain real PII.

## Key Concepts

### 1. Pareto Frontier
The Pareto frontier tracks which prompts perform best on each validation sentence. A prompt is on the Pareto frontier for a sentence if no other prompt achieves a strictly better score on that sentence. Each sentence has its own Pareto front, and prompts appearing in many fronts are considered more robust.

### 2. Dominated Prompts
A prompt is dominated if, for every validation sentence where it appears in the Pareto front, there exists another prompt in that same front. Dominated prompts are filtered out during selection to focus on the most promising candidates.

### 3. Mutation
Mutation creates new prompts by analyzing failures in the current prompt. An LLM evaluates the parent prompt on a minibatch of training examples, identifies errors and missed PII, then generates an improved prompt that addresses these issues. Only mutations that improve on the minibatch are evaluated on the full validation set.

### 4. Merging
Merging combines two successful prompts from the Pareto frontier to create a hybrid that incorporates strengths from both. An LLM analyzes what each prompt does well and synthesizes a new prompt that combines their complementary strategies. Merges are scheduled after successful mutations.

### 5. Train/Validation Split
The training set is used for minibatch-based mutation evaluation (quick filtering), while the validation set is used for Pareto frontier tracking. This separation prevents overfitting and allows efficient exploration of the prompt space.

### 6. Minibatch Evaluation
Instead of evaluating every mutation on the full validation set, GEPA first tests mutations on a small random sample from the training set. Only mutations that improve on this minibatch proceed to full validation evaluation, significantly reducing computational cost.

### 7. Rollout Budget
The rollout budget determines the number of optimization iterations. Each rollout attempts either a merge (if scheduled) or a mutation. The budget controls the exploration-exploitation tradeoff and total computational cost.

### 8. Weighted Selection
Parent prompts are selected randomly from the Pareto frontier, weighted by their frequency across all sentence-level Pareto fronts. Prompts that appear in more fronts have higher selection probability, biasing evolution toward robust solutions.

### 9. LLM-as-Judge
Evaluation uses an LLM to judge how well PII was stripped from each sentence. The judge assigns scores from 0.0 to 1.0, identifies removed and missed PII, and provides feedback. This flexible evaluation adapts to different types of PII without hard-coded rules.

### 10. Acceptance Criteria
A mutated prompt is accepted only if it scores higher than its parent on the training minibatch. A merged prompt is always added to the candidate pool. Both are then evaluated on the full validation set to update the Pareto frontiers.

## Core Algorithm

The optimization process follows this structure:

```python
def optimize(
    base_prompt: str,
    train_sentences: list[str],
    val_sentences: list[str],
    evaluator,
    mutator,
    merger,
    rollouts_budget: int,
) -> str:
    # Initialize Pareto helper with validation set
    pareto_helper = ParetoHelper(base_prompt, val_sentences)

    # Evaluate base prompt
    base_val_subscores = evaluator.evaluate_per_sentence(base_prompt, val_sentences)
    pareto_helper.update_with_new_prompt(base_prompt, base_val_subscores)

    for rollout in range(rollouts_budget):
        # Step 1: Try merge if scheduled and last mutation succeeded
        if merge_is_scheduled():
            prompt1, prompt2 = select_two_from_pareto_front(pareto_helper)
            merged_prompt = merger.merge(prompt1, prompt2)
            merged_subscores = evaluator.evaluate_per_sentence(merged_prompt, val_sentences)
            pareto_helper.update_with_new_prompt(merged_prompt, merged_subscores)
            continue

        # Step 2: Mutation
        parent_prompt = select_from_pareto_front(pareto_helper)

        # Evaluate on training minibatch
        minibatch = random.sample(train_sentences, minibatch_size)
        parent_eval = evaluator.evaluate_with_traces(parent_prompt, minibatch)
        child_prompt = mutator.mutate(parent_prompt, parent_eval)
        child_eval = evaluator.evaluate_with_traces(child_prompt, minibatch)

        # Accept if improved on minibatch
        if sum(child_eval['scores']) > sum(parent_eval['scores']):
            # Evaluate on full validation set
            child_val_subscores = evaluator.evaluate_per_sentence(child_prompt, val_sentences)
            pareto_helper.update_with_new_prompt(child_prompt, child_val_subscores)

            # Schedule merge for next iteration
            schedule_merge()

    return pareto_helper.best_candidate()
```

## Usage

1. Install dependencies:
```bash
uv sync
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

3. Run the optimization:
```bash
uv run python main.py
```

The optimizer will evolve prompts over 5 iterations, logging progress and saving the best prompt to `best_prompt.txt`.
