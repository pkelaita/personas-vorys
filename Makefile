.PHONY: *

export PYTHONPATH := src

default: lint type

init:
	uv sync

lint:
	uv run ruff check .

type:
	uv run mypy src

create-sample-persona:
	uv run sample_persona/create_sample_persona.py

load-transcript:
	uv run src/load_transcript.py $(filter-out $@,$(MAKECMDGOALS))

create-chunks:
	uv run src/create_chunks.py $(filter-out $@,$(MAKECMDGOALS))

run-analysis:
	uv run src/run_analysis.py $(filter-out $@,$(MAKECMDGOALS))

compare-analyses:
	uv run src/compare_analyses.py $(filter-out $@,$(MAKECMDGOALS))

%:
	@:
