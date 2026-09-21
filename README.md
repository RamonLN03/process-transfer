# ProcessTransfer

Physics-aware transfer of hybrid process models from a data-rich source plant to a related, data-scarce target plant.

**Status:** early research software. The active milestone is M0, a virtual plant and data infrastructure trustworthy enough that later machine-learning results can be believed. No modelling or transfer code exists yet.

## What the project asks

Related plants share physics: conservation laws, stoichiometry, the form of the kinetics. They differ in geometry, heat transfer, catalyst state and instrumentation. ProcessTransfer investigates whether that shared structure can be used to decide what a model carries over from one plant to the next, what has to be recalibrated, and what has to be relearned from a small amount of target data. The scope, research questions, architecture and roadmap live under `docs/`.

## Working on the repository

Read `AGENTS.md` first. It is the operating contract for every agent and person committing here. `CLAUDE.md` adds the Claude Code specifics. All repository content is written in English.

### Installation

```
python -m venv <path-to-venv>
<activate the venv>
pip install -e ".[dev]"
```

Keep the virtual environment outside synced folders such as OneDrive.

### Tests and lint

```
pytest
ruff check .
```

Continuous integration runs both on every push.

### Generated data

Generated data never goes into git. The `PT_DATA_DIR` environment variable sets where Parquet files and the DuckDB database are written (default `data/`, resolved against the repository root). See `.env.example`.

## Layout

```
src/process_transfer/   reusable code, organised by concern (see docs/architecture.md)
tests/                  pytest suite, including tests of physical behaviour
docs/                   scope, research questions, architecture, roadmap, decisions, assumptions
configs/                plant and sensor configurations (engineering units, converted to SI on load)
sql/                    schema and readable data-quality queries
experiments/            scripts that generate data and run experiments
```

Directories appear as the milestones that need them are implemented.
