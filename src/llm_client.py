import os
from openai import OpenAI
from dotenv import load_dotenv


class LLMClient:
    """Client for connecting to OpenAI LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.

        Args:
            model: OpenAI model name to use (default: gpt-4o-mini)
        """
        # Load environment variables from .env file
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Input prompt string

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
