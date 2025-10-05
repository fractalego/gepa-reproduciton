import random


class ParetoHelper:
    def __init__(self, base_prompt, sentences, base_subscores):
        """
        Args:
            base_prompt: Initial prompt
            sentences: Validation sentences
            base_subscores: Scores for base prompt on each sentence
        """
        self.prompt_candidates = [base_prompt]
        self.per_prompt_scores = [sum(base_subscores) / len(base_subscores)]

        # Track which prompts are Pareto-optimal for each sentence
        # List of sets: prompt_at_pareto_front_sentences[sentence_idx] = {prompt_idx, ...}
        self.prompt_at_pareto_front_sentences = [{0} for _ in range(len(sentences))]

        # Track best score achieved on each sentence
        self.pareto_front_sentences = base_subscores.copy()

        # Set random seed for reproducibility
        random.seed(42)

    def update_with_new_prompt(self, new_prompt, subscores):
        new_prompt_idx = len(self.prompt_candidates)
        self.prompt_candidates.append(new_prompt)

        # Calculate overall score
        overall_score = sum(subscores) / len(subscores)
        self.per_prompt_scores.append(overall_score)

        # Update per-sentence Pareto fronts
        for sentence_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_sentences, subscores)):
            if new_score > old_score:
                # New prompt beats all previous ones on this sentence
                self.pareto_front_sentences[sentence_idx] = new_score
                self.prompt_at_pareto_front_sentences[sentence_idx] = {new_prompt_idx}
            elif new_score == old_score:
                # Tie - add to the front
                self.prompt_at_pareto_front_sentences[sentence_idx].add(new_prompt_idx)

    def select_pareto_candidate(self):
        """Select a parent prompt from the Pareto fronts using weighted random sampling"""
        # Count frequency of each prompt in sentence Pareto fronts
        prompt_frequency = {}
        for sentence_pareto_front in self.prompt_at_pareto_front_sentences:
            for prompt_idx in sentence_pareto_front:
                if prompt_idx not in prompt_frequency:
                    prompt_frequency[prompt_idx] = 0
                prompt_frequency[prompt_idx] += 1

        # Create weighted sampling list (each prompt appears frequency times)
        sampling_list = [prompt_idx for prompt_idx, freq in prompt_frequency.items() for _ in range(freq)]

        # Randomly select from weighted list
        parent_idx = random.choice(sampling_list)
        parent = self.prompt_candidates[parent_idx]
        return parent_idx, parent

    def best_candidate(self):
        best_idx = self.per_prompt_scores.index(max(self.per_prompt_scores))
        return self.prompt_candidates[best_idx]
