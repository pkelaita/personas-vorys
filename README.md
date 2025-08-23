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
- `AZURE_OPENAI_API_KEY`        # For Azure OpenAI (optional, if using Azure)
- `AZURE_OPENAI_ENDPOINT`       # e.g. https://YOUR_RESOURCE_NAME.openai.azure.com
- `AZURE_OPENAI_API_VERSION`    # e.g. 2024-06-01

This can be done either at the system level or by creating a `.env` file in the root directory.

To install the dependencies, run `make init`.

## Usage

Run the below commands from the root directory.

- Load transcripts concurrently: `make load-transcript <persona> <transcript1> [transcript2...]`
  - Example: `make load-transcript mark t1 t2 t3 t4`
- Create strategy chunks concurrently: `make create-chunks <persona> <transcript1> [transcript2...]`
  - Example: `make create-chunks mark t1 t2 t3 t4`
- Run base and persona analyses: `make run-analysis <persona> <transcript> <model>`
  - Example (l2m2 providers): `make run-analysis mark t1 gpt-4.1`
  - Example (Azure OpenAI via SDK): `make run-analysis mark t1 azure:YOUR_DEPLOYMENT_NAME`
    - Note: When using Azure, pass your deployment name prefixed with `azure:`. Filenames will normalize `:` to `-` (e.g., `azure:foo-deploy` -> files start with `azure-foo-deploy_...`).
- Compare analyses and see results: `make compare-analyses <persona> <transcript> <model>`
  - Example: `make compare-analyses mark t1 gpt-4.1`
  - Example (Azure): `make compare-analyses mark t1 azure:YOUR_DEPLOYMENT_NAME`
- Create persona from interview DOCX: `make create-persona <persona> [model]`
  - Input: `data/personas/<persona>/<persona>_interview.docx`
  - Output: `data/personas/<persona>/persona.txt`
  - Example (Azure): `make create-persona saramirez azure:YOUR_DEPLOYMENT_NAME`
  - Example (env vars like sample-persona): `MODEL="azure:YOUR_DEPLOYMENT_NAME" REASONING_EFFORT=minimal make create-persona saramirez`

Possible personas are `mark`, `bill`, or `alycia`. Transcripts are numbered `t1` through `t[n]`. The models tested in the paper are `gpt-4.1`, `claude-3.7-sonnet`, and `gemini-2.5-pro`, although these scripts support any model that is supported by the [l2m2](https://github.com/pkelaita/l2m2) library.

Azure OpenAI usage notes:
- Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION`.
- Use the model string form `azure:DEPLOYMENT_NAME` to route calls to Azure using the official SDK.
- Output filenames replace `:` with `-` for cross-platform compatibility (e.g., `azure:my-deploy` -> `azure-my-deploy_*_analysis.json`).
- The evaluation model for judging in `src/compare_analyses.py` defaults to `gpt-4.1`. You can change it if desired.

## Create Persona from Interview DOCX

Create an LLM persona from your own interview transcript stored as a DOCX file.

- Driver script: `create_persona/create_persona.py`
- Input file naming: the script looks for `data/personas/<name>/<name>_interview.docx`
- Output file: `data/personas/<name>/persona.txt` (overwritten if it exists)
- Invocation (run from repo root):
  - Positional Azure deployment: `make create-persona <name> azure:YOUR_DEPLOYMENT_NAME`
  - Or via env vars (mirrors sample-persona): `MODEL="azure:YOUR_DEPLOYMENT_NAME" REASONING_EFFORT=minimal make create-persona <name>`
- Azure notes:
  - This command uses the Azure OpenAI Chat Completions path. Ensure `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION` are set (see Setup).
  - Pass your Azure deployment name prefixed with `azure:`. If your deployment is named `gpt-5_dzs`, use `azure:gpt-5_dzs`. Passing a raw model string (e.g., `gpt-5`) will only work if your Azure deployment is literally named `gpt-5`.

## Sample Persona Creation

A redacted persona interview transcript is provided to demonstrate our persona creation process. You can find the transcript files in the [sample_persona](./sample_persona/) directory, and the prompt used to generate the sample persona in [sample_persona/create_sample_persona.py](./sample_persona/create_sample_persona.py).

Running the command `make create-sample-persona` will run the above interview and prompt through an LLM and save the output to [sample_persona/sample_persona.txt](./sample_persona/sample_persona.txt). We used Gemini 2.5 Pro in the paper, but you can change the model used by modifying the `MODEL` variable in the script.

## License

This project is licensed under the MIT License.
