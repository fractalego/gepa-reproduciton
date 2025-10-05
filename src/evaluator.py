import json
from typing import Any
from json_repair import repair_json


EVALUATION_PROMPT = """
You are an expert evaluator for PII (Personally Identifiable Information) stripping tasks.

You will be given:
1. An original sentence
2. A sanitized version of that sentence

Your task is to evaluate how well the PII was removed from the sentence.

Original sentence: {original}
Sanitized sentence: {sanitized}

Evaluate the sanitization and provide:
1. A score from 0.0 to 1.0 (where 1.0 means all PII was perfectly removed)
2. List of PII that was successfully removed
3. List of PII that was missed (still present in sanitized version)
4. Brief feedback on the quality

Respond ONLY with a JSON object in this format:
{{
  "score": 0.0-1.0,
  "removed_pii": ["list", "of", "removed", "pii"],
  "missed_pii": ["list", "of", "missed", "pii"],
  "feedback": "brief explanation"
}}
"""


class Evaluator:
    """Evaluates PII stripping prompts using an LLM as the judge."""

    def __init__(self, model, llm_client):
        """
        Args:
            model: Model instance with run(prompt, sentence) method
            llm_client: LLM client with a generate(prompt: str) -> str method for evaluation
        """
        self.model = model
        self.llm_client = llm_client

    def _evaluate_with_llm(self, original: str, sanitized: str) -> dict:
        """Use LLM to evaluate the sanitization quality."""
        eval_prompt = EVALUATION_PROMPT.format(original=original, sanitized=sanitized)
        response = self.llm_client.generate(eval_prompt)

        # Parse evaluation response
        try:
            repaired = repair_json(response)
            eval_result = json.loads(repaired)
            return {
                "score": float(eval_result.get("score", 0.0)),
                "removed_pii": eval_result.get("removed_pii", []),
                "missed_pii": eval_result.get("missed_pii", []),
                "feedback": eval_result.get("feedback", "")
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            # If evaluation fails, return 0 score
            return {
                "score": 0.0,
                "removed_pii": [],
                "missed_pii": [],
                "feedback": "Failed to parse evaluation response"
            }

    def evaluate_per_sentence(self, prompt: str, sentences: list[str]) -> list[float]:
        """
        Evaluate a prompt on sentences and return per-sentence scores.

        Args:
            prompt: The PII stripping prompt to evaluate
            sentences: List of sentences containing PII

        Returns:
            List of scores (0.0 to 1.0) for each sentence
        """
        scores = []

        for sentence in sentences:
            # Run PII stripper model
            sanitized = self.model.run(prompt, sentence)

            # Evaluate with LLM
            eval_result = self._evaluate_with_llm(sentence, sanitized)
            scores.append(eval_result["score"])

        return scores

    def evaluate_with_traces(self, prompt: str, sentences: list[str]) -> dict[str, Any]:
        """
        Evaluate a prompt and capture detailed traces for reflection.

        Args:
            prompt: The PII stripping prompt to evaluate
            sentences: List of sentences containing PII

        Returns:
            Dict with 'scores', 'traces' containing detailed execution info
        """
        scores = []
        traces = []

        for sentence in sentences:
            # Run PII stripper model
            sanitized = self.model.run(prompt, sentence)

            # Evaluate with LLM
            eval_result = self._evaluate_with_llm(sentence, sanitized)

            score = eval_result["score"]
            scores.append(score)

            # Build trace with all information for reflection
            trace = {
                "input": sentence,
                "sanitized_output": sanitized,
                "score": score,
                "removed_pii": eval_result["removed_pii"],
                "missed_pii": eval_result["missed_pii"],
                "feedback": eval_result["feedback"]
            }
            traces.append(trace)

        return {"scores": scores, "traces": traces}
