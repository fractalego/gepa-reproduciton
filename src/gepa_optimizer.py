from src.pareto_helper import ParetoHelper


class GepaOptimizer:
    def optimize(
        self,
        base_prompt: str,  # Initial prompt string
        sentences: list[str],  # List of validation sentences
        evaluator,  # Evaluator object with evaluate_per_sentence() method
        reflector,  # Reflector object with reflect() method
        mutator,  # Mutator object with mutate() method
        rollouts_budget: int,
    ) -> str:
        """
        Optimize a prompt using GEPA algorithm.

        Args:
            base_prompt: Starting prompt string
            sentences: List of validation sentences to evaluate on
            evaluator: Object that evaluates prompts and returns per-sentence scores
            reflector: Object that generates feedback from evaluation results
            mutator: Object that mutates prompts based on feedback
            rollouts_budget: Number of optimization iterations

        Returns:
            Best prompt string found
        """
        pareto_helper = ParetoHelper(base_prompt, sentences)

        for rollout in range(rollouts_budget):
            # Select parent prompt from Pareto front
            parent_idx, parent_prompt = pareto_helper.select_pareto_candidate()

            # Evaluate parent prompt on sentences to get traces
            eval_results = evaluator.evaluate_with_traces(parent_prompt, sentences)

            # Generate reflection/feedback from evaluation traces
            feedback = reflector.reflect(parent_prompt, eval_results)

            # Mutate prompt based on feedback
            child_prompt = mutator.mutate(parent_prompt, feedback)

            # Evaluate child prompt and get per-sentence scores
            subscores = evaluator.evaluate_per_sentence(child_prompt, sentences)

            # Update Pareto fronts with new prompt
            pareto_helper.update_with_new_prompt(child_prompt, subscores)

        return pareto_helper.best_candidate()
