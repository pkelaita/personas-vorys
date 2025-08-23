import asyncio
import os
import sys
import argparse
from typing import Optional

from dotenv import load_dotenv

# Ensure 'src' is on sys.path so scripts outside src/ can import project modules
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_router import is_azure_model, azure_chat  # noqa: E402

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL") or "gpt-5"

PROMPT_TEMPLATE = """
You are tasked with creating a persona summary for a lawyer based on the
provided interview transcript. The persona should be written as if you are
speaking as the lawyer (e.g., "You are [Lawyer's Name]...")

Objective: You are creating a persona of {name}.
Create a concise yet comprehensive persona summary that captures
the essence of {name}'s professional style, with particular
emphasis on:

-   Their core professional philosophy and overarching approach, especially
those principles they express with strong conviction, emphatic language, or
through vivid metaphors/analogies (e.g., "like the back of my hand,"
"universal key").
-   Specific, actionable methods, techniques, and processes they use in case
preparation (e.g., document review workflows, fact-finding steps) and any
self-described personality traits that influence this.
-   Their actionable strategies and self-described style in depositions
(e.g., specific questioning approaches, use of outlines vs. flexibility,
how they listen and react, interaction with witnesses/opposing counsel).
-   How they approach uncovering, analyzing, and leveraging critical facts,
evidence, or inconsistencies, focusing on their actionable methodology.
-   Their decision-making process regarding when and how to reveal crucial
information or "smoking guns," highlighting the strategic and actionable
considerations.
-   Any unique analogies, metaphors, emphatic statements, or self-described
personality traits that define their professional conduct and strongly held
beliefs.

Output Requirements:

-   Persona Voice: The summary must begin with "You are {name}..."
and maintain this second-person perspective addressing the lawyer.
-   Key Trait Emphasis: Identify 2-4 defining characteristics/skills and
self-described personality aspects in the opening sentence and bold them.
Ensure this highlights their actionable strategies and strongly held
beliefs/convictions.
-   Detailed Analysis: Provide a few paragraphs detailing their approach,
drawing directly from the transcript content. Ensure this section
particularly highlights actionable strategies, strongly held beliefs (often
indicated by emphatic language, metaphors, or absolutes), and
self-described personality traits. Bold these key concepts, methods, or
philosophies.
-   Key Quotes: Include a section with 3-5 direct, verbatim quotes from the
interview. Select quotes that powerfully illustrate their actionable
strategies, strongly held beliefs/convictions, and self-described
personality style, especially those using emphatic language, vivid imagery,
or absolute terms.
-   Conciseness and Impact: The summary should be impactful and distill the
most important aspects of their persona, reflecting the intensity of their
convictions where appropriate.

Input: The following is an interview transcript with {name}:
--- START OF TRANSCRIPT ---
{transcript}
--- END OF TRANSCRIPT ---

Please generate the persona summary for {name} based on these
instructions.
"""


def read_docx_text(docx_path: str) -> str:
    # Lazy import to avoid import cost if unused elsewhere
    from docx import Document  # type: ignore

    doc = Document(docx_path)
    parts = []
    for p in doc.paragraphs:
        text = p.text or ""
        parts.append(text)
    # Preserve paragraph breaks
    return "\n".join(parts).strip()


async def create_persona(name: str, model: str, reasoning_effort: Optional[str] = None) -> str:
    """
    Reads data/personas/{name}/{name}_interview.docx, generates persona text via LLM,
    and writes data/personas/{name}/persona.txt. Returns output path.
    """
    folder = os.path.join(BASE_DIR, "data", "personas", name)
    interview_docx = os.path.join(folder, f"{name}_interview.docx")
    if not os.path.exists(interview_docx):
        raise FileNotFoundError(
            f"Interview file not found: {interview_docx}. Expected DOCX named '{name}_interview.docx' in {folder}"
        )

    interview = read_docx_text(interview_docx)
    prompt = PROMPT_TEMPLATE.format(transcript=interview, name=name)

    is_az, deployment = is_azure_model(model)
    deployment_name = deployment if is_az else model
    response = await azure_chat(
        system_prompt="",
        prompt=prompt,
        deployment=deployment_name,
        timeout=600,
        reasoning_effort=reasoning_effort,
    )

    out_path = os.path.join(folder, "persona.txt")
    with open(out_path, "w") as f:
        f.write(response)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a persona from a DOCX interview in data/personas/<name>/<name>_interview.docx"
    )
    parser.add_argument(
        "name",
        help="Folder name under data/personas containing <name>_interview.docx (e.g., 'saramirez')",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help="Model label, e.g., 'azure:<deployment>' or a non-Azure model",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        choices=["low", "medium", "high", "minimal"],
        default=os.getenv("REASONING_EFFORT"),
        help="Optional reasoning effort setting for Azure reasoning models",
    )
    args = parser.parse_args()
    out_path = asyncio.run(create_persona(args.name, args.model, args.reasoning_effort))
    print(f"Wrote persona to: {out_path}")


if __name__ == "__main__":
    main()
