import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Protocol, Optional, Union, Awaitable

import backoff
from dotenv import load_dotenv
from l2m2.client import AsyncLLMClient
from l2m2.exceptions import LLMRateLimitError
from llm_router import is_azure_model, normalize_model_label, azure_chat


load_dotenv()


BATCH_SIZE = 40

class _SupportsLLMCall(Protocol):
    def call(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = ...,
        temperature: Optional[float] = ...,
        max_tokens: Optional[int] = ...,
        prefer_provider: Optional[str] = ...,
        json_mode: bool = ...,
        json_mode_strategy: Optional[Any] = ...,
        timeout: Optional[int] = ...,
        bypass_memory: bool = ...,
        alt_memory: Optional[Any] = ...,
        extra_params: Optional[Dict[str, Union[str, int, float]]] = ...,
        extra_headers: Optional[Dict[str, str]] = ...,
    ) -> Awaitable[str]: ...


def build_system_prompt(use_persona: bool, persona_name: str) -> str:
    persona_prompt = ""
    if use_persona:
        persona_file = os.path.join("data", "personas", persona_name, "persona.txt")
        try:
            with open(persona_file, "r") as f:
                persona_contents = f.read()
            persona_prompt = f"""
You are a specific lawyer. Below is a detailed analysis of who you are, including quotes.
Use this analysis to inform your strategic thinking about how you handle depositions.

{persona_contents}
"""
        except FileNotFoundError:
            print(f"Warning: {persona_file} not found. Persona prompt will be empty.")
        except IOError as e:
            print(
                f"Warning: Error reading {persona_file}: {e}. Persona prompt will be empty."
            )

    return f"""You are a senior litigator analyzing excerpts from a deposition transcript.
You will be presented with segments of the transcript, potentially representing points where opposing
counsel employed specific legal strategies.

{persona_prompt}

Based *only* on the provided transcript excerpts, analyze the situation and state the single most
important strategic move, line of questioning, or point you would focus on next if you were
representing the deponent. Be concise and specific in your response. Respond only with your recommended
next step.
"""


async def process_analysis_batch(
    tasks: List[asyncio.Task[str]],
    task_metadata: List[str],
) -> Tuple[Dict[str, str], int, int]:
    """Runs a batch of analysis tasks concurrently and returns results."""

    batch_results: Dict[str, str] = {}
    tasks_run_batch = 0
    skips_batch = 0

    run_type = task_metadata[0].split("_")[0] if task_metadata else "Unknown"
    print(f"  [{run_type.upper()}] Running analysis batch of {len(tasks)} tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        chunk_label = task_metadata[i]

        if isinstance(result, Exception):
            error_message = f"Error calling LLM for chunk {chunk_label}: {result}"
            print(f"  [{run_type.upper()}] {error_message}")
            batch_results[chunk_label] = f"Error: {result}"
            skips_batch += 1
        elif isinstance(result, str):
            print(f"  [{run_type.upper()}] LLM Suggestion received for {chunk_label}:")
            print(f"    {result[:100]}...")
            batch_results[chunk_label] = result
            tasks_run_batch += 1
        else:
            error_message = f"Unexpected result type for {chunk_label}: {type(result)}"
            print(f"  [{run_type.upper()}] {error_message}")
            batch_results[chunk_label] = f"Error: {error_message}"
            skips_batch += 1

    print(
        f"  [{run_type.upper()}] Batch complete. Tasks run: {tasks_run_batch}, Skipped/Errors: {skips_batch}"
    )
    return batch_results, tasks_run_batch, skips_batch


@backoff.on_exception(
    backoff.expo,
    LLMRateLimitError,
    max_tries=60,
    on_backoff=lambda details: print(
        f"\nRate limit hit, retrying in {float(details.get('wait', 0.0)):.1f} seconds... "
        f"(attempt {details.get('tries', '?')}/{details.get('max_tries', '?')})"
    ),
)
async def call_with_backoff(
    client: _SupportsLLMCall,
    model: str,
    system_prompt: str,
    prompt: str,
    timeout: int,
) -> str:
    """Calls the LLM with exponential backoff for rate limit errors."""
    result = await client.call(
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        timeout=timeout,
    )
    return str(result)


# Azure backoff exception tuple and call wrapper
AZURE_BACKOFF_EXC: tuple[type[BaseException], ...]
try:
    from openai import RateLimitError as _AzureRateLimitError
    from openai import APIConnectionError as _AzureAPIConnectionError
    from openai import APITimeoutError as _AzureAPITimeoutError
    from openai import APIError as _AzureAPIError
    AZURE_BACKOFF_EXC = (_AzureRateLimitError, _AzureAPIConnectionError, _AzureAPITimeoutError, _AzureAPIError)
except Exception:  # openai not installed yet or different version
    AZURE_BACKOFF_EXC = (Exception,)


@backoff.on_exception(
    backoff.expo,
    LLMRateLimitError,
    max_tries=60,
    on_backoff=lambda details: print(
        f"\nRate limit hit, retrying in {float(details.get('wait', 0.0)):.1f} seconds... "
        f"(attempt {details.get('tries', '?')}/{details.get('max_tries', '?')})"
    ),
)
@backoff.on_exception(
    backoff.expo,
    AZURE_BACKOFF_EXC,
    max_tries=60,
    on_backoff=lambda details: print(
        f"\nRate limit/API error (Azure), retrying in {float(details.get('wait', 0.0)):.1f} seconds... "
        f"(attempt {details.get('tries', '?')}/{details.get('max_tries', '?')})"
    ),
)
async def azure_call_with_backoff(system_prompt: str, prompt: str, timeout: int, deployment: str) -> str:
    result = await azure_chat(
        system_prompt=system_prompt,
        prompt=prompt,
        deployment=deployment,
        timeout=timeout,
        temperature=0.2,
    )
    return str(result)


async def analyze_strategy_chunks(
    use_persona: bool,
    persona_name: str,
    transcript_key: str,
    model: str,
) -> None:
    """
    Loads transcript chunks, analyzes them using an LLM (batched),
    and saves the suggested next strategies for either base or persona run.
    """

    run_type = "persona" if use_persona else "base"
    system_prompt = build_system_prompt(use_persona, persona_name)
    is_az, deployment = is_azure_model(model)
    norm_model = normalize_model_label(model)

    chunk_file = os.path.join(
        "data", "personas", persona_name, transcript_key, "strategy_chunks.json"
    )
    output_file = os.path.join(
        "data",
        "personas",
        persona_name,
        transcript_key,
        f"{norm_model}_{run_type}_analysis.json",
    )

    print(f"--- Starting analysis run: {run_type.upper()} --- Output: {output_file}")

    # --- Load Chunks ---

    try:
        if not os.path.exists(os.path.dirname(chunk_file)):
            print(
                f"[{run_type.upper()}] Error: Data directory '{os.path.dirname(chunk_file)}' not found."
            )
            return
        if not os.path.exists(chunk_file):
            print(
                f"[{run_type.upper()}] Error: {chunk_file} not found. Please run the main script first."
            )
            return
        with open(chunk_file, "r") as f:
            strategy_chunks = json.load(f)

    except json.JSONDecodeError:
        print(f"[{run_type.upper()}] Error: Could not decode JSON from {chunk_file}.")
        return
    except IOError as e:
        print(f"[{run_type.upper()}] Error reading {chunk_file}: {e}")
        return

    if not isinstance(strategy_chunks, list):
        print(
            f"[{run_type.upper()}] Error: Expected a list of chunks in {chunk_file}, but got {type(strategy_chunks)}."
        )
        return

    if not strategy_chunks:
        print(f"[{run_type.upper()}] {chunk_file} is empty or contains no chunks.")
        return

    num_chunks = len(strategy_chunks)
    analysis_results = {}
    total_tasks_run = 0
    total_skips = 0

    print(
        f"[{run_type.upper()}] Analyzing {num_chunks} chunks from {chunk_file} "
        f"(Batch Size: {BATCH_SIZE})..."
    )

    async with AsyncLLMClient() as client:
        tasks: List[asyncio.Task[str]] = []
        task_metadata: List[str] = []

        for i, chunk in enumerate(strategy_chunks):
            if not chunk or not str(chunk).strip():
                print(f"[{run_type.upper()}] Skipping empty chunk {i}")
                analysis_results[f"chunk_0-{i}"] = "Skipped (empty content)"
                total_skips += 1
                continue

            print(f"\n[{run_type.upper()}] Processing chunk {i+1}/{num_chunks}")
            print(f"  [{run_type.upper()}] Input text length: {len(str(chunk))} chars")

            if is_az:
                task = asyncio.create_task(
                    azure_call_with_backoff(
                        system_prompt=system_prompt,
                        prompt=str(chunk),
                        timeout=600,
                        deployment=deployment,
                    )
                )
            else:
                task = asyncio.create_task(
                    call_with_backoff(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        prompt=str(chunk),
                        timeout=600,
                    )
                )
            tasks.append(task)
            task_metadata.append(f"chunk_0-{i}")

            if len(tasks) >= BATCH_SIZE or i == num_chunks - 1:
                print(
                    f"\n[{run_type.upper()}] Processing batch of {len(tasks)} tasks..."
                )
                batch_res, batch_run, batch_skip = await process_analysis_batch(
                    tasks, task_metadata
                )
                analysis_results.update(batch_res)
                total_tasks_run += batch_run
                total_skips += batch_skip
                tasks.clear()
                task_metadata.clear()

    # --- Save results ---

    print(f"\n[{run_type.upper()}] --- Analysis Complete ---")
    print(f"[{run_type.upper()}] Total chunks processed: {num_chunks}")
    print(f"[{run_type.upper()}] Successful LLM tasks completed: {total_tasks_run}")
    print(f"[{run_type.upper()}] Total skipped tasks: {total_skips}")

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[{run_type.upper()}] Created output directory: {output_dir}")

        print(
            f"[{run_type.upper()}] Saving {len(analysis_results)} results to {output_file}..."
        )
        with open(output_file, "w") as f:
            json.dump(analysis_results, f, indent=2)
        print(
            f"[{run_type.upper()}] Analysis results successfully saved to {output_file}."
        )

    except IOError as e:
        print(f"[{run_type.upper()}] Error saving results to {output_file}: {e}")
    except TypeError as e:
        print(f"[{run_type.upper()}] Error serializing results to JSON: {e}")


async def batch_run_analysis(
    persona_name: str,
    transcript_key: str,
    model: str,
) -> None:
    """Runs the base and persona analysis concurrently."""

    print("Starting concurrent analysis runs (Base and Persona)...")
    task_base = analyze_strategy_chunks(
        use_persona=False,
        persona_name=persona_name,
        transcript_key=transcript_key,
        model=model,
    )
    task_persona = analyze_strategy_chunks(
        use_persona=True,
        persona_name=persona_name,
        transcript_key=transcript_key,
        model=model,
    )

    await asyncio.gather(task_base, task_persona)
    print("\nBoth analysis runs completed.")


if __name__ == "__main__":
    persona_name = sys.argv[1]
    transcript_key = sys.argv[2]
    model = sys.argv[3]
    asyncio.run(batch_run_analysis(persona_name, transcript_key, model))
