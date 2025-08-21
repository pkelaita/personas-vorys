import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
from pydantic import BaseModel, ValidationError

import asyncio
import json
import os
import backoff

from dotenv import load_dotenv
from l2m2.client import AsyncLLMClient
from l2m2.exceptions import LLMRateLimitError

load_dotenv()


EVALUATION_MODEL = "gpt-4.1"

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator assessing the *specificity and foresight* of a
strategic legal analysis against the actual subsequent events in a deposition transcript.

You will be given:
1.  The 'Actual Next Transcript Chunk': Text representing what actually happened immediately following the
point where the analysis was made.
2.  'Analysis': A strategic suggestion made prior to the chunk.

Your task is to evaluate if the 'Analysis' provided **specific, personalized strategic guidance** that directly
addressed the potential dynamics, issues, or lines of inquiry that subsequently unfolded in the 'Actual Next
Transcript Chunk'. The focus is on whether the advice was **tailored and insightful for *this specific
context***, rather than being generic legal counsel that could apply in many situations.

Does the 'Analysis' demonstrate a **specific anticipation** of the content, tone, or tactics seen in the
'Actual Next Transcript Chunk'? Was it **personalized advice** addressing the unique circumstances at that
moment, or a more general strategic point?

Respond *only* with a JSON object containing the following fields:

{
"explanation": "A short explanation detailing *why* the analysis was or was not specifically tailored and
predictive, avoiding generic justifications.",
"score": "YES" or "NO"  // YES if the analysis was specific, personalized, AND reflected in the subsequent
                        // chunk; NO otherwise.
}
"""


class ResponseFormat(BaseModel):
    score: Literal["YES", "NO"]
    explanation: str


def load_json_file(filepath: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Loads a JSON file with error handling."""
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, (dict, list)):
                return cast(Union[Dict[str, Any], List[Any]], data)
            else:
                print(f"Error: Unexpected data type {type(data)} in {filepath}.")
                return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None
    except IOError as e:
        print(f"Error reading {filepath}: {e}")
        return None


async def process_results(
    results: List[Union[str, Exception]],
    task_metadata: List[Tuple[str, str]],
) -> Tuple[int, int, int, int]:
    base_yes = 0
    persona_yes = 0
    assessments_run = 0
    skipped_assessments = 0

    for i, result in enumerate(results):
        chunk_key, analysis_type = task_metadata[i]

        if isinstance(result, Exception):
            print(
                f"  Error during LLM assessment for chunk {chunk_key} ({analysis_type}): {result}"
            )
            skipped_assessments += 1
            continue
        elif not isinstance(result, str):
            print(
                f"  Unexpected result type for chunk {chunk_key} ({analysis_type}): {type(result)}. Skipping."
            )
            skipped_assessments += 1
            continue

        print(f"  Result for {chunk_key} ({analysis_type}): {result}")

        try:
            parsed_result = ResponseFormat.model_validate_json(result)
        except ValidationError as e:
            print(f"  Error parsing JSON for {chunk_key} ({analysis_type}): {e}")
            skipped_assessments += 1
            continue

        if parsed_result.score == "YES":
            if analysis_type == "base":
                base_yes += 1
                print(f"  Outcome for {chunk_key}: Base rated YES.")
            elif analysis_type == "persona":
                persona_yes += 1
                print(f"  Outcome for {chunk_key}: Persona rated YES.")
            assessments_run += 1
        else:
            print(f"  Outcome for {chunk_key} ({analysis_type}): Rated NO.")
            assessments_run += 1

    return base_yes, persona_yes, assessments_run, skipped_assessments


@backoff.on_exception(
    backoff.expo,
    LLMRateLimitError,
    max_tries=60,
    on_backoff=lambda details: print(
        f"Rate limit hit, retrying in {details['wait']:.1f} seconds... "
        f"(attempt {details['tries']}/{details['max_tries']})",
        end="",
    ),
)
async def call_with_backoff(
    client: AsyncLLMClient,
    prompt: str,
) -> str:
    """Calls the LLM with exponential backoff for rate limit errors."""
    result = await client.call(
        model=EVALUATION_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        prompt=prompt,
        timeout=300,
        json_mode=True,
        temperature=0.1,
    )
    return str(result)


async def compare_single_analysis(
    client: AsyncLLMClient,
    chunk_key: str,
    base_prediction: str,
    persona_prediction: str,
    next_chunk_text: str,
) -> Tuple[
    Optional[
        Tuple[asyncio.Task[str], Tuple[str, str], asyncio.Task[str], Tuple[str, str]]
    ],
    Optional[str],
]:
    """
    Compares a single chunk's base and persona predictions against the next chunk.
    Returns a tuple of (base_task, base_metadata, persona_task, persona_metadata) if successful,
    or (None, error_message) if failed.
    """
    # Validate inputs
    if (
        not base_prediction
        or not isinstance(base_prediction, str)
        or base_prediction.startswith("Error:")
        or base_prediction.startswith("Skipped")
    ):
        return None, "Invalid or missing base prediction"
    if (
        not persona_prediction
        or not isinstance(persona_prediction, str)
        or persona_prediction.startswith("Error:")
        or persona_prediction.startswith("Skipped")
    ):
        return None, "Invalid or missing persona prediction"

    if (
        not next_chunk_text
        or not isinstance(next_chunk_text, str)
        or not next_chunk_text.strip()
    ):
        return None, "Next chunk is empty or invalid"

    print(f"  Base Analysis:    {base_prediction[:80]}...")
    print(f"  Persona Analysis: {persona_prediction[:80]}...")
    print(f"  Actual Next Chunk: {next_chunk_text[:80]}...")

    prompt_base = (
        f"Actual Next Transcript Chunk:\n---\n{next_chunk_text}\n---\n\n"
        f"Analysis:\n---\n{base_prediction}\n---"
    )
    prompt_persona = (
        f"Actual Next Transcript Chunk:\n---\n{next_chunk_text}\n---\n\n"
        f"Analysis:\n---\n{persona_prediction}\n---"
    )

    base_metadata = (chunk_key, "base")
    persona_metadata = (chunk_key, "persona")

    task_base = call_with_backoff(
        client=client,
        prompt=prompt_base,
    )
    task_persona = call_with_backoff(
        client=client,
        prompt=prompt_persona,
    )

    return (task_base, base_metadata, task_persona, persona_metadata), None


async def compare_analyses(persona_name: str, transcript_key: str, model: str) -> None:
    persona_dir = os.path.join("data", "personas", persona_name, transcript_key)
    base_analysis_file = os.path.join(persona_dir, f"{model}_base_analysis.json")
    persona_analysis_file = os.path.join(persona_dir, f"{model}_persona_analysis.json")
    chunk_file = os.path.join(persona_dir, "strategy_chunks.json")

    base_analysis = load_json_file(base_analysis_file)
    persona_analysis = load_json_file(persona_analysis_file)
    strategy_chunks = load_json_file(chunk_file)

    if base_analysis is None or persona_analysis is None or strategy_chunks is None:
        print("Error loading necessary files. Exiting.")
        return

    if not isinstance(base_analysis, dict) or not isinstance(persona_analysis, dict):
        print("Error: Analysis files should contain JSON objects (dictionaries).")
        return
    if not isinstance(strategy_chunks, list):
        print("Error: Chunk file should contain a JSON list.")
        return

    base_score = 0
    persona_score = 0
    tasks_run = 0
    skips = 0

    # Get all chunk keys that exist in both analyses
    common_keys = sorted(
        k
        for k in base_analysis.keys() & persona_analysis.keys()
        if k.startswith("chunk_0-")
    )
    total_chunks = len(common_keys)
    print(f"\nFound {total_chunks} common analysis chunks to process.")

    async with AsyncLLMClient() as client:
        tasks: List[asyncio.Task[str]] = []
        task_metadata: List[Tuple[str, str]] = []

        for i, chunk_key in enumerate(common_keys):
            current_chunk_index = i + 1
            print(
                f"\n--- Processing chunk: {chunk_key} ({current_chunk_index}/{total_chunks}) ---"
            )

            base_prediction = base_analysis.get(chunk_key)
            persona_prediction = persona_analysis.get(chunk_key)

            # Get next chunk text
            chunk_index = int(
                chunk_key.split("-")[1]
            )  # use the legacy chunk_0-n format
            next_chunk_index = chunk_index + 1
            if next_chunk_index >= len(strategy_chunks):
                print(
                    f"Skipping: Next chunk index ({next_chunk_index}) is out of bounds "
                    f"(total chunks: {len(strategy_chunks)})."
                )
                skips += 1
                continue

            next_chunk_text = strategy_chunks[next_chunk_index]

            result, error = await compare_single_analysis(
                client=client,
                chunk_key=chunk_key,
                base_prediction=base_prediction,
                persona_prediction=persona_prediction,
                next_chunk_text=next_chunk_text,
            )

            if error:
                print(f"Skipping: {error}")
                skips += 1
                continue

            task_base, base_metadata, task_persona, persona_metadata = result
            tasks.append(task_base)
            task_metadata.append(base_metadata)
            tasks.append(task_persona)
            task_metadata.append(persona_metadata)

        print(f"Analyzing {len(tasks)} chunks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        b_score, p_score, t_run, s_run = await process_results(
            results,
            task_metadata,
        )

        base_score += b_score
        persona_score += p_score
        tasks_run += t_run
        skips += s_run

    print("\n--- Evaluation Complete ---")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Successful LLM tasks completed: {tasks_run}")
    print(f"Chunks skipped (prep phase + LLM errors/invalid results): {skips}")
    print("\nFinal Scores:")
    print(f"  Base Analysis Wins:    {base_score}")
    print(f"  Persona Analysis Wins: {persona_score}")
    print("---------------------------")


if __name__ == "__main__":
    persona_name = sys.argv[1]
    transcript_key = sys.argv[2]
    model = sys.argv[3]
    asyncio.run(compare_analyses(persona_name, transcript_key, model))
