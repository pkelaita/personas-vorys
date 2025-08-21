import asyncio
import os

from dotenv import load_dotenv
from l2m2.client import AsyncLLMClient

load_dotenv()

MODEL = "gemini-2.5-pro"
DIR = "sample_persona"

PROMPT_TEMPLATE = """
You are tasked with creating a persona summary for a lawyer based on the
provided interview transcript. The persona should be written as if you are
speaking as the lawyer (e.g., "You are [Lawyer's Name]...")

Objective: You are creating a persona of {name}.
Create a concise yet comprehensive persona summary that captures
the essence of {name}'s professional style, with particular
emphasis on:

-	Their core professional philosophy and overarching approach, especially
those principles they express with strong conviction, emphatic language, or
through vivid metaphors/analogies (e.g., "like the back of my hand,"
"universal key").
-	Specific, actionable methods, techniques, and processes they use in case
preparation (e.g., document review workflows, fact-finding steps) and any
self-described personality traits that influence this.
-	Their actionable strategies and self-described style in depositions
(e.g., specific questioning approaches, use of outlines vs. flexibility,
how they listen and react, interaction with witnesses/opposing counsel).
-	How they approach uncovering, analyzing, and leveraging critical facts,
evidence, or inconsistencies, focusing on their actionable methodology.
-	Their decision-making process regarding when and how to reveal crucial
information or "smoking guns," highlighting the strategic and actionable
considerations.
-	Any unique analogies, metaphors, emphatic statements, or self-described
personality traits that define their professional conduct and strongly held
beliefs.

Output Requirements:

-	Persona Voice: The summary must begin with "You are {name}..."
and maintain this second-person perspective addressing the lawyer.
-	Key Trait Emphasis: Identify 2-4 defining characteristics/skills and
self-described personality aspects in the opening sentence and bold them.
Ensure this highlights their actionable strategies and strongly held
beliefs/convictions.
-	Detailed Analysis: Provide a few paragraphs detailing their approach,
drawing directly from the transcript content. Ensure this section
particularly highlights actionable strategies, strongly held beliefs (often
indicated by emphatic language, metaphors, or absolutes), and
self-described personality traits. Bold these key concepts, methods, or
philosophies.
-	Key Quotes: Include a section with 3-5 direct, verbatim quotes from the
interview. Select quotes that powerfully illustrate their actionable
strategies, strongly held beliefs/convictions, and self-described
personality style, especially those using emphatic language, vivid imagery,
or absolute terms.
-	Conciseness and Impact: The summary should be impactful and distill the
most important aspects of their persona, reflecting the intensity of their
convictions where appropriate.

Input: The following is an interview transcript with {name}:
--- START OF TRANSCRIPT ---
{transcript}
--- END OF TRANSCRIPT ---

Please generate the persona summary for {name} based on these
instructions.
"""


async def create_sample_persona():
    interview_filepath = os.path.join(DIR, "sample_interview.txt")
    with open(interview_filepath, "r") as f:
        interview = f.read()

    prompt = PROMPT_TEMPLATE.format(transcript=interview, name="Partner A")

    async with AsyncLLMClient() as client:
        response = await client.call(
            model=MODEL,
            prompt=prompt,
            timeout=None,
        )

    with open(os.path.join(DIR, "sample_persona.txt"), "w") as f:
        f.write(response)


if __name__ == "__main__":
    asyncio.run(create_sample_persona())
