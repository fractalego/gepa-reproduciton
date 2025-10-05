original_prompt = """
You are an algorithm that strip PII from texts. Your input is a sentence and your output is the same sentence without PII.

The output should be in JSON format with a field "text" containing the sanitized text like this:
{
  "text": "Sanitized text here"
}

Output only the JSON object and nothing else.
"""


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


MERGE_PROMPT = """I have two different prompt instructions that both perform well at stripping PII (Personally Identifiable Information) from text. Each prompt has learned different strategies and strengths.

Prompt 1:
```
{prompt1}
```

Prompt 2:
```
{prompt2}
```

Your task is to create a new, improved prompt that combines the best aspects of both prompts.

Analyze what each prompt does well:
- What techniques or strategies does each use?
- What specific instructions or guidance does each provide?
- Are there complementary strengths that can be combined?

Create a merged prompt that:
- Incorporates the best strategies from both prompts
- Maintains clarity and coherence
- Is not simply a concatenation, but a thoughtful synthesis

Provide the merged prompt within ``` blocks."""
