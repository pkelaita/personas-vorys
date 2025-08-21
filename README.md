# Persona Evaluations

Supplimental material for the paper "Personas Unlock Legal Intuition – Aligning LLMs to
Subjective Expert Strategies."

## Requirements

- GNU Make
- Python >= <!-- python-v -->3.13.3<!-- /python-v -->
- uv >= <!-- uv-v -->0.7.2<!-- /uv-v -->

## Setup

Ensure the following environment variables are set.

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

This can be done either at the system level or by creating a `.env` file in the root directory.

To install the dependencies, run `make init`.

## Usage

Run the below commands from the root directory.

- Load transcripts concurrently: `make load-transcript <persona> <transcript1> [transcript2...]`
  - Example: `make load-transcript mark t1 t2 t3 t4`
- Create strategy chunks concurrently: `make create-chunks <persona> <transcript1> [transcript2...]`
  - Example: `make create-chunks mark t1 t2 t3 t4`
- Run base and persona analyses: `make run-analysis <persona> <transcript> <model>`
  - Example: `make run-analysis mark t1 gpt-4.1`
- Compare analyses and see results: `make compare-analyses <persona> <transcript> <model>`
  - Example: `make compare-analyses mark t1 gpt-4.1 gpt-4o`

Possible personas are `mark`, `bill`, or `alycia`. Transcripts are numbered `t1` through `t[n]`. The models tested in the paper are `gpt-4.1`, `claude-3.7-sonnet`, and `gemini-2.5-pro`, although these scripts support any model that is supported by the [l2m2](https://github.com/pkelaita/l2m2) library.

## Sample Persona Creation

A redacted persona interview transcript is provided to demonstrate our persona creation process. You can find the transcript files in the [sample_persona](./sample_persona/) directory, and the prompt used to generate the sample persona in [sample_persona/create_sample_persona.py](./sample_persona/create_sample_persona.py).

Running the command `make create-sample-persona` will run the above interview and prompt through an LLM and save the output to [sample_persona/sample_persona.txt](./sample_persona/sample_persona.txt). We used Gemini 2.5 Pro in the paper, but you can change the model used by modifying the `MODEL` variable in the script.

## License

This project is licensed under the MIT License.
