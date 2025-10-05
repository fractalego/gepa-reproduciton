MUTATION_PROMPT = """I provided an assistant with the following instructions to perform a task for me:
```
{current_instruction}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
{inputs_outputs_feedback}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""


class Mutator:
    """Mutates prompts based on feedback from evaluation traces."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client with a generate(prompt: str) -> str method
        """
        self.llm_client = llm_client

    def _format_feedback_examples(self, traces: list[dict]) -> str:
        """Format traces into feedback examples for the mutation prompt."""
        examples = []

        for i, trace in enumerate(traces):
            example = f"""# Example {i + 1}

## Input
{trace['input']}

## Assistant's Output
{trace['sanitized_output']}

## Score
{trace['score']}

## Feedback
{trace['feedback']}

## Removed PII
{', '.join(trace['removed_pii']) if trace['removed_pii'] else 'None'}

## Missed PII
{', '.join(trace['missed_pii']) if trace['missed_pii'] else 'None'}
"""
            examples.append(example)

        return "\n\n".join(examples)

    def mutate(self, current_prompt: str, eval_results: dict) -> str:
        """
        Mutate a prompt based on evaluation feedback.

        Args:
            current_prompt: The current prompt to mutate
            eval_results: Dict with 'scores' and 'traces' from evaluate_with_traces()

        Returns:
            New mutated prompt string
        """
        # Format feedback examples from traces
        feedback_text = self._format_feedback_examples(eval_results['traces'])

        # Create mutation prompt
        mutation_prompt = MUTATION_PROMPT.format(
            current_instruction=current_prompt,
            inputs_outputs_feedback=feedback_text
        )

        # Get LLM response - the response itself is the new instruction
        new_instruction = self.llm_client.generate(mutation_prompt)

        return new_instruction.strip()
