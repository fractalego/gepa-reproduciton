import json
from json_repair import repair_json


class Model:
    """PII stripping model that runs prompts on sentences."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client with a generate(prompt: str) -> str method
        """
        self.llm_client = llm_client

    def run(self, prompt: str, sentence: str) -> str:
        """
        Run the PII stripping prompt on a sentence.

        Args:
            prompt: The PII stripping prompt
            sentence: Input sentence containing PII

        Returns:
            Sanitized sentence with PII removed
        """
        full_prompt = f"{prompt}\n\nInput sentence: {sentence}"
        response = self.llm_client.generate(full_prompt)

        try:
            repaired = repair_json(response)
            result = json.loads(repaired)
            sanitized = result.get("text", "")
            return sanitized
        except (json.JSONDecodeError, KeyError):
            return response
