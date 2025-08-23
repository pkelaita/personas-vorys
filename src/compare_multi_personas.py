import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, cast

import backoff
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from l2m2.client import AsyncLLMClient
from l2m2.exceptions import LLMRateLimitError

load_dotenv()

ANALYST_PERSONAS = ["mark", "bill", "alycia"]

JUDGE_SYSTEM_PROMPT = '''
You are an impartial evaluator assessing the *specificity and foresight* of a
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
'''

class ResponseFormat(BaseModel):
    score: Literal["YES", "NO"]
    explanation: str


# -------------------------------------------------
# Helper utilities
# -------------------------------------------------

def load_json_file(filepath: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    if not os.path.exists(filepath):
        print(f"Error: file not found -> {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, (dict, list)):
                return cast(Union[Dict[str, Any], List[Any]], data)
            print(f"Error: Unexpected type {type(data)} in {filepath}")
            return None
    except Exception as e:  # JSON / IO errors
        print(f"Error loading {filepath}: {e}")
        return None


# -------------------------------------------------
# LLM call with retry
# -------------------------------------------------

@backoff.on_exception(
    backoff.expo,
    LLMRateLimitError,
    max_tries=60,
    on_backoff=lambda details: print(
        f"Rate-limit hit, retrying in {details['wait']:.1f}s… (attempt {details['tries']})",
        end="",
    ),
)
async def call_with_backoff(client: Any, prompt: str, model: str) -> str:
    resp = await client.call(
        model=model,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        prompt=prompt,
        timeout=300,
        json_mode=True,
        temperature=0.1,
    )
    return str(resp)


# -------------------------------------------------
# Main comparison routine
# -------------------------------------------------

async def compare_personae(transcript_owner: str, transcript_key: str, model: str) -> None:
    base_dir = os.path.join("data", "personas", transcript_owner, transcript_key)
    chunk_path = os.path.join(base_dir, "strategy_chunks.json")

    # load strategy chunks once
    strategy_chunks = load_json_file(chunk_path)
    if strategy_chunks is None or not isinstance(strategy_chunks, list):
        print("Unable to load transcript chunks – aborting.")
        return

    # load each persona analysis
    analyses: Dict[str, Dict[str, str]] = {}
    for persona in ANALYST_PERSONAS:
        path = os.path.join(base_dir, f"{model}_{persona}_analysis.json")
        data = load_json_file(path)
        if data is None or not isinstance(data, dict):
            print(f"Warning: could not load analysis for {persona} – skipping whole run.")
            return
        analyses[persona] = cast(Dict[str, str], data)

    # ensure we only evaluate chunks that every persona produced
    common_keys = sorted(
        set.intersection(*(set(a.keys()) for a in analyses.values()))
        if analyses else []
    )
    total_chunks = len(common_keys)
    print(f"Found {total_chunks} common chunks across all persona analyses.")

    scores = {p: 0 for p in ANALYST_PERSONAS}
    tasks: List[asyncio.Task[str]] = []
    metadata: List[Tuple[str, str]] = []  # (persona, chunk_key)

    async with AsyncLLMClient() as client:
        for chunk_key in common_keys:
            # derive next-chunk text
            idx = int(chunk_key.split("-")[1])
            if idx + 1 >= len(strategy_chunks):
                continue
            next_chunk_text = strategy_chunks[idx + 1]
            if not isinstance(next_chunk_text, str) or not next_chunk_text.strip():
                continue

            for persona in ANALYST_PERSONAS:
                analysis_text = analyses[persona].get(chunk_key, "")
                if not analysis_text or analysis_text.startswith("Error"):
                    continue
                prompt = (
                    f"Actual Next Transcript Chunk:\n---\n{next_chunk_text}\n---\n\n"
                    f"Analysis:\n---\n{analysis_text}\n---"
                )
                task = asyncio.create_task(call_with_backoff(client=client, prompt=prompt, model=model))
                tasks.append(task)
                metadata.append((persona, chunk_key))

        print(f"Submitting {len(tasks)} LLM judge calls…")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # process results
    for res, (persona, _chunk_key) in zip(results, metadata, strict=True):
        if isinstance(res, Exception):
            continue
        try:
            parsed = ResponseFormat.model_validate_json(cast(str, res))
            if parsed.score == "YES":
                scores[persona] += 1
        except ValidationError:
            continue

    # summary
    print("\n--- Cross-Persona Evaluation Complete ---")
    print(f"Chunks evaluated : {total_chunks}")
    for persona in ANALYST_PERSONAS:
        print(f"{persona.capitalize():>6}: {scores[persona]} YES votes")


if __name__ == "__main__":
    owner = sys.argv[1]
    t_key = sys.argv[2]
    mdl = sys.argv[3]
    asyncio.run(compare_personae(owner, t_key, mdl))
