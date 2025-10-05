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
        reflector,
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
            reflector: Reflector that generates feedback from evaluation results
            mutator: Mutator with mutate() method
            merger: Merger with merge() method
            rollouts_budget: Number of optimization iterations

        Returns:
            Best prompt string found
        """
        # Initialize Pareto helper with VALIDATION set
        pareto_helper = ParetoHelper(base_prompt, val_sentences)

        # Evaluate base prompt on validation set and initialize Pareto fronts
        base_val_subscores = evaluator.evaluate_per_sentence(base_prompt, val_sentences)
        pareto_helper.update_with_new_prompt(base_prompt, base_val_subscores)

        for rollout in range(rollouts_budget):
            # Step 1: Try merge first if scheduled and last mutation succeeded
            if self._merge_prompts_if_relevant(pareto_helper, evaluator, merger, val_sentences):
                continue  # Skip mutation this iteration

            # Reset flag before mutation
            self.last_mutation_succeeded = False

            # Step 2: Mutation
            parent_idx, parent_prompt = pareto_helper.select_pareto_candidate()

            # Sample MINIBATCH from TRAIN set
            minibatch = random.sample(train_sentences, min(self.minibatch_size, len(train_sentences)))

            # Evaluate parent on minibatch with traces
            parent_eval = evaluator.evaluate_with_traces(parent_prompt, minibatch)
            parent_minibatch_score = sum(parent_eval['scores'])

            # Generate feedback and mutate
            feedback = reflector.reflect(parent_prompt, parent_eval)
            child_prompt = mutator.mutate(parent_prompt, feedback)

            # Evaluate child on SAME minibatch (quick check)
            child_eval = evaluator.evaluate_with_traces(child_prompt, minibatch)
            child_minibatch_score = sum(child_eval['scores'])

            # Check if mutation improved on minibatch
            if child_minibatch_score > parent_minibatch_score:
                # SUCCESS on minibatch! Now do full VALIDATION evaluation
                child_val_subscores = evaluator.evaluate_per_sentence(child_prompt, val_sentences)
                pareto_helper.update_with_new_prompt(child_prompt, child_val_subscores)

                # Schedule merge for next iteration
                self.last_mutation_succeeded = True
                if self.total_merges_tested < self.max_merges:
                    self.merges_scheduled += 1

        return pareto_helper.best_candidate()

    def _merge_prompts_if_relevant(self, pareto_helper, evaluator, merger, val_sentences):
        """Try to merge two prompts from Pareto front if conditions are met."""
        if not (self.merges_scheduled > 0 and
                self.last_mutation_succeeded and
                self.total_merges_tested < self.max_merges):
            return False

        # Get two candidates from Pareto front
        prompt1_idx, prompt1 = pareto_helper.select_pareto_candidate()
        prompt2_idx, prompt2 = pareto_helper.select_pareto_candidate()

        if prompt1_idx == prompt2_idx:
            return False

        # Merge the two prompts
        merged_prompt = merger.merge(prompt1, prompt2)

        if merged_prompt is None:
            return False

        # Evaluate merged prompt on VALIDATION set
        merged_subscores = evaluator.evaluate_per_sentence(merged_prompt, val_sentences)

        # Update Pareto fronts with merged prompt
        pareto_helper.update_with_new_prompt(merged_prompt, merged_subscores)
        self.merges_scheduled -= 1
        self.total_merges_tested += 1

        return True