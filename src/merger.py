from src.prompts import MERGE_PROMPT


class Merger:
    """Merges two prompts by combining their best aspects using an LLM."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client with a generate(prompt: str) -> str method
        """
        self.llm_client = llm_client

    def merge(self, prompt1: str, prompt2: str) -> str:
        """
        Merge two prompts into a single improved prompt.

        Args:
            prompt1: First prompt string
            prompt2: Second prompt string

        Returns:
            Merged prompt string, or None if merge fails
        """
        # Don't merge if prompts are identical
        if prompt1 == prompt2:
            return None

        # Create merge prompt
        merge_prompt = MERGE_PROMPT.format(prompt1=prompt1, prompt2=prompt2)

        # Get LLM response
        response = self.llm_client.generate(merge_prompt)

        # Extract merged prompt from response
        merged = self._extract_prompt(response)

        return merged

    def _extract_prompt(self, response: str) -> str:
        """Extract prompt from LLM response (within ``` blocks)."""
        # Find content between ``` blocks
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                # Get the content between first pair of ```
                merged_prompt = parts[1].strip()
                # Remove language identifier if present
                if merged_prompt.startswith(("json", "python", "text")):
                    merged_prompt = "\n".join(merged_prompt.split("\n")[1:]).strip()
                return merged_prompt

        # If no ``` blocks found, return the whole response
        return response.strip()
