from src.pareto_helper import ParetoHelper


class GepaOptimizer:
    def __init__(self, max_merges: int = 10):
        """
        Args:
            max_merges: Maximum number of merge operations allowed
        """
        self.max_merges = max_merges
        self.total_merges_tested = 0
        self.merges_scheduled = 0
        self.last_mutation_succeeded = False

    def _merge_prompts_if_relevant(self, pareto_helper, evaluator, merger, sentences):
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

        # Evaluate merged prompt
        merged_subscores = evaluator.evaluate_per_sentence(merged_prompt, sentences)

        # Update Pareto fronts with merged prompt
        pareto_helper.update_with_new_prompt(merged_prompt, merged_subscores)
        self.merges_scheduled -= 1
        self.total_merges_tested += 1

        return True

    def optimize(
        self,
        base_prompt: str,
        sentences: list[str],
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
            sentences: List of validation sentences to evaluate on
            evaluator: Evaluator with evaluate_per_sentence() and evaluate_with_traces()
            reflector: Reflector that generates feedback from evaluation results
            mutator: Mutator with mutate() method
            merger: Merger with merge() method
            rollouts_budget: Number of optimization iterations

        Returns:
            Best prompt string found
        """
        pareto_helper = ParetoHelper(base_prompt, sentences)

        for rollout in range(rollouts_budget):
            # Step 1: Try merge first if scheduled and last mutation succeeded
            if self._merge_prompts_if_relevant(pareto_helper, evaluator, merger, sentences):
                continue  # Skip mutation this iteration

            # Reset flag before mutation
            self.last_mutation_succeeded = False

            # Step 2: Mutation
            parent_idx, parent_prompt = pareto_helper.select_pareto_candidate()

            # Evaluate parent prompt on sentences to get traces
            eval_results = evaluator.evaluate_with_traces(parent_prompt, sentences)

            # Generate reflection/feedback from evaluation traces
            feedback = reflector.reflect(parent_prompt, eval_results)

            # Mutate prompt based on feedback
            child_prompt = mutator.mutate(parent_prompt, feedback)

            # Evaluate child prompt and get per-sentence scores
            child_subscores = evaluator.evaluate_per_sentence(child_prompt, sentences)
            parent_score = sum(eval_results['scores'])
            child_score = sum(child_subscores)

            # Accept if improved
            if child_score > parent_score:
                pareto_helper.update_with_new_prompt(child_prompt, child_subscores)

                # Schedule merge for next iteration
                self.last_mutation_succeeded = True
                if self.total_merges_tested < self.max_merges:
                    self.merges_scheduled += 1

        return pareto_helper.best_candidate()
