# ProcessTransfer

Physics-aware transfer of hybrid process models from a data-rich source plant to a related, data-scarce target plant.

**Status:** early research software. The first milestone, M0, a virtual plant and data infrastructure trustworthy enough that later machine-learning results can be believed, was closed on 2026-09-22. The active milestone is M1, black-box and hybrid models fitted on target-plant data only; its plan is `docs/m1_plan.md`. Its first iteration, the evaluation contract and the mechanistic models fitted on the target, is accepted and closed. The second, a diagnostic of what the data determine about the parameters of the mechanistic model, has been run and awaits review. The third, the training framework and the learned models, is closed; no benchmark has been run. No transfer code exists yet.

## What the project asks

Related plants share physics: conservation laws, stoichiometry, the form of the kinetics. They differ in geometry, heat transfer, catalyst state and instrumentation. ProcessTransfer investigates whether that shared structure can be used to decide what a model carries over from one plant to the next, what has to be recalibrated, and what has to be relearned from a small amount of target data. The scope, research questions, architecture and roadmap live under `docs/`.

## Working on the repository

Read `AGENTS.md` first. It is the operating contract for every agent and person committing here. `CLAUDE.md` adds the Claude Code specifics. All repository content is written in English.

### Installation

```
python -m venv <path-to-venv>
<activate the venv>
pip install -e ".[dev,learning]"
```

The extra `learning` adds JAX, which trains the learned models of M1 (D-035). The data path and its container do not need it.

Keep the virtual environment outside synced folders such as OneDrive.

To reconstruct the exact environment the registered M0 results were generated and
checked in, rather than the current dependency bounds of `pyproject.toml`, see
`docs/reference_environment.md` and `requirements-reference.lock.txt`.

The data path can also be built and run in a Linux container, with Docker, without a
local Python: `docs/docker.md` has the commands and what they do.

### Tests and lint

```
pytest
ruff check .
```

Continuous integration runs both on every push.

### Generating a data set

One command goes from the configuration files to a verified data set, a DuckDB database and an export of aligned series, and returns a non-zero exit code if any mandatory check fails:

```
python -m process_transfer.generation configs/datasets/m0_e05.yaml
```

The definitions under `configs/datasets/` are the three operating runs of D-010: `m0_e05.yaml` (protocol P3), `m0_e06.yaml` (steady operation with noise only) and `m0_e07.yaml` (single-input step tests); the experiments `05`, `06` and `07` under `experiments/` add the checks proper to each, and `08_oracle_conductance.py` measures the effect of the temperature-dependent conductance as an oracle diagnostic whose outputs never reach the available branch.

From PyCharm, create a Python run configuration with *module name* `process_transfer.generation` and the path of the definition as its parameter. The working directory does not matter: a relative path is looked for under the repository root, and generated files go under `PT_DATA_DIR`. Experiments are plain scripts, for example `experiments/05_full_data_path.py`, and can be run the same way as *script path*.

The data contract is `docs/data_contract.md`, the SQL is described in `sql/README.md`, and `docs/m0_audit.md` says what the first milestone has and has not delivered.

### Generated data

Generated data never goes into git. The `PT_DATA_DIR` environment variable sets where Parquet files and the DuckDB database are written (default `data/`, resolved against the repository root). See `.env.example`. If the repository lives in a synchronised folder, point `PT_DATA_DIR` outside it: a database file can be damaged by a synchronisation client while it is open.

Under it, `available/` holds everything a model may read, data sets, databases and exports, and `private/` holds what is needed to regenerate and diagnose them, seeds and full configurations included. This is a separation of content and code paths, not a permission of the operating system.

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
