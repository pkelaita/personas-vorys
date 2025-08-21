from typing import List
import os
import asyncio
import json
import re
import sys

from dotenv import load_dotenv
from l2m2.client import AsyncLLMClient


load_dotenv()


MODEL = "gpt-4.1"

PROMPT = """ You are a stellar lawyer reviewing a transcript of a deposition.

Each line in the transcript is marked with a line number at the beginning,
enclosed in square brackets (e.g., "[256]").

Please review the transcript and return a list of each line number where the
deposing attorney invokes a specific legal strategy to gain an advantage in the
case.

Please return a comma-separated list of line numbers, and absolutely nothing
else.

Example output:

123, 456, 789, etc.
"""


def clean_lines(text: str) -> str:
    lines = text.split("\n")

    # remove non-alpha lines
    lines = [line for line in lines if re.search(r"[a-zA-Z]", line)]

    # remove lines with the name of the court reporting company
    lines = [
        line for line in lines if not any(company in line for company in ["Versagi"])
    ]

    # remove all leading non-alpha characters in each line
    lines = [re.sub(r"^[^a-zA-Z]+", "", line) for line in lines]

    return "\n".join(lines)


def add_indicies(text: str) -> str:
    return "\n".join([f"[{i}] {line}" for i, line in enumerate(text.split("\n"))])


async def create_chunks(persona_name: str, transcript_key: str) -> None:
    transcript_dir = os.path.join("data", "personas", persona_name, transcript_key)
    if not os.path.exists(transcript_dir):
        raise FileNotFoundError(
            f"Transcript {transcript_key} not found in {transcript_dir}"
        )

    transcript = open(os.path.join(transcript_dir, "transcript.txt")).read()
    transcript = clean_lines(transcript)
    marked_transcript = add_indicies(transcript)

    print(f"Analyzing transcript {transcript_key} ({len(marked_transcript)} chars)...")

    async with AsyncLLMClient() as client:
        result = await client.call(
            model=MODEL,
            system_prompt=PROMPT,
            prompt=marked_transcript,
            timeout=None,
        )

    # Clean result
    result = re.sub(r"[^0-9,]", "", result)
    result = result.strip(",")

    indices = [int(i) for i in result.split(",")]
    lines = transcript.split("\n")

    strategy_chunks = []
    start = 0
    for end in indices:
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)
        strategy_chunks.append(chunk_text)
        start = end

    with open(os.path.join(transcript_dir, "strategy_chunks.json"), "w") as f:
        json.dump(strategy_chunks, f)

    print(f"Done analyzing transcript {transcript_key}")


async def batch_create_chunks(persona_name: str, transcript_keys: List[str]) -> None:
    tasks = [
        create_chunks(persona_name, transcript_key)
        for transcript_key in transcript_keys
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    persona_name = sys.argv[1]
    transcript_keys = sys.argv[2:]
    asyncio.run(batch_create_chunks(persona_name, transcript_keys))
