from src.pareto_helper import ParetoHelper
import random


class GepaOptimizer:
    def __init__(self, max_merges: int = 10, minibatch_size: int = 5):
        """
        Args:
            max_merges: Maximum number of merge operations allowed
            minibatch_size: Number of train examples to use for mutation evaluation
        """
        self.max_merges = max_merges
        self.minibatch_size = minibatch_size
        self.total_merges_tested = 0
        self.merges_scheduled = 0
        self.last_mutation_succeeded = False

    def optimize(
        self,
        base_prompt: str,
        train_sentences: list[str],  # Training set for minibatch mutation
        val_sentences: list[str],    # Validation set for Pareto fronts
        evaluator,
        mutator,
        merger,
        rollouts_budget: int,
    ) -> str:
        """
        Optimize a prompt using GEPA algorithm with mutation and merge.

        Args:
            base_prompt: Starting prompt string
            train_sentences: Training sentences for minibatch-based mutation
            val_sentences: Validation sentences for Pareto front tracking
            evaluator: Evaluator with evaluate_per_sentence() and evaluate_with_traces()
            mutator: Mutator with mutate() method
            merger: Merger with merge() method
            rollouts_budget: Number of optimization iterations

        Returns:
            Best prompt string found
        """
        # Initialize Pareto helper with VALIDATION set
        pareto_helper = ParetoHelper(base_prompt, val_sentences)

        # Evaluate base prompt on validation set and initialize Pareto fronts
        self._log_header("GEPA Prompt Optimization")
        self._log_prompt("Starting with Base Prompt", base_prompt)

        base_val_subscores = evaluator.evaluate_per_sentence(base_prompt, val_sentences)
        base_score = sum(base_val_subscores) / len(base_val_subscores)
        pareto_helper.update_with_new_prompt(base_prompt, base_val_subscores)

        self._log_info(f"Base prompt validation score: {base_score:.3f}")
        self._log_pareto_front(pareto_helper)

        for rollout in range(rollouts_budget):
            self._log_section(f"Iteration {rollout + 1}/{rollouts_budget}")
            # Step 1: Try merge first if scheduled and last mutation succeeded
            if self._merge_prompts_if_relevant(pareto_helper, evaluator, merger, val_sentences):
                continue  # Skip mutation this iteration

            # Reset flag before mutation
            self.last_mutation_succeeded = False

            # Step 2: Mutation
            parent_idx, parent_prompt = pareto_helper.select_pareto_candidate()
            self._log_info(f"Selected parent prompt [{parent_idx}] from Pareto front")

            # Sample MINIBATCH from TRAIN set
            minibatch = random.sample(train_sentences, min(self.minibatch_size, len(train_sentences)))
            self._log_info(f"Sampled {len(minibatch)} training examples for mutation")

            # Evaluate parent on minibatch with traces
            parent_eval = evaluator.evaluate_with_traces(parent_prompt, minibatch)
            parent_minibatch_score = sum(parent_eval['scores'])

            # Mutate based on evaluation results
            child_prompt = mutator.mutate(parent_prompt, parent_eval)
            self._log_info("Generated mutated prompt")

            # Evaluate child on SAME minibatch (quick check)
            child_eval = evaluator.evaluate_with_traces(child_prompt, minibatch)
            child_minibatch_score = sum(child_eval['scores'])

            self._log_info(f"Minibatch scores - Parent: {parent_minibatch_score:.3f}, Child: {child_minibatch_score:.3f}")

            # Check if mutation improved on minibatch
            if child_minibatch_score > parent_minibatch_score:
                # SUCCESS on minibatch! Now do full VALIDATION evaluation
                self._log_info("✨ Mutation improved on minibatch! Evaluating on validation set...")
                child_val_subscores = evaluator.evaluate_per_sentence(child_prompt, val_sentences)
                child_val_score = sum(child_val_subscores) / len(child_val_subscores)

                pareto_helper.update_with_new_prompt(child_prompt, child_val_subscores)
                self._log_prompt("Accepted New Prompt", child_prompt, child_val_score)
                self._log_pareto_front(pareto_helper)

                # Schedule merge for next iteration
                self.last_mutation_succeeded = True
                if self.total_merges_tested < self.max_merges:
                    self.merges_scheduled += 1
                    self._log_info("📅 Merge scheduled for next iteration")
            else:
                self._log_info("❌ Mutation rejected (no improvement on minibatch)")

        # Final summary
        best_prompt = pareto_helper.best_candidate()
        best_score = max(pareto_helper.per_prompt_scores)
        self._log_header("Optimization Complete")
        self._log_prompt("Best Prompt Found", best_prompt, best_score)
        self._log_info(f"Total prompts explored: {len(pareto_helper.prompt_candidates)}")
        self._log_info(f"Total merges performed: {self.total_merges_tested}")

        return best_prompt

    def _log_header(self, text: str):
        """Print a header log."""
        print(f"\n{'='*80}")
        print(f"  {text}")
        print(f"{'='*80}")

    def _log_section(self, text: str):
        """Print a section log."""
        print(f"\n{'─'*80}")
        print(f"  {text}")
        print(f"{'─'*80}")

    def _log_info(self, text: str):
        """Print an info log."""
        print(f"  ✓ {text}")

    def _log_prompt(self, label: str, prompt: str, score: float = None):
        """Print a prompt with optional score."""
        print(f"\n  📝 {label}:")
        if score is not None:
            print(f"     Score: {score:.3f}")
        print(f"     {prompt[:100]}..." if len(prompt) > 100 else f"     {prompt}")

    def _log_pareto_front(self, pareto_helper):
        """Log the current Pareto front."""
        print(f"\n  🏆 Current Pareto Front ({len(pareto_helper.prompt_candidates)} prompts):")
        for idx, score in enumerate(pareto_helper.per_prompt_scores):
            num_sentences = sum(1 for front in pareto_helper.prompt_at_pareto_front_sentences if idx in front)
            print(f"     [{idx}] Score: {score:.3f} | Pareto on {num_sentences} sentences")

    def _merge_prompts_if_relevant(self, pareto_helper, evaluator, merger, val_sentences):
        """Try to merge two prompts from Pareto front if conditions are met."""
        if not (self.merges_scheduled > 0 and
                self.last_mutation_succeeded and
                self.total_merges_tested < self.max_merges):
            return False

        self._log_info("🔀 Attempting merge...")

        # Get two candidates from Pareto front
        prompt1_idx, prompt1 = pareto_helper.select_pareto_candidate()
        prompt2_idx, prompt2 = pareto_helper.select_pareto_candidate()

        if prompt1_idx == prompt2_idx:
            self._log_info("❌ Merge skipped (same prompt selected twice)")
            return False

        self._log_info(f"Merging prompts [{prompt1_idx}] and [{prompt2_idx}]")

        # Merge the two prompts
        merged_prompt = merger.merge(prompt1, prompt2)

        if merged_prompt is None:
            self._log_info("❌ Merge failed (prompts too similar)")
            return False

        # Evaluate merged prompt on VALIDATION set
        merged_subscores = evaluator.evaluate_per_sentence(merged_prompt, val_sentences)
        merged_score = sum(merged_subscores) / len(merged_subscores)

        # Update Pareto fronts with merged prompt
        pareto_helper.update_with_new_prompt(merged_prompt, merged_subscores)
        self.merges_scheduled -= 1
        self.total_merges_tested += 1

        self._log_prompt("✨ Merged Prompt Accepted", merged_prompt, merged_score)
        self._log_pareto_front(pareto_helper)

        return True