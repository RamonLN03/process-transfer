# Architecture

## Separation of concerns

The package is organised so that these remain logically distinct (AGENTS.md):

| Concern | Package | Milestone |
|---|---|---|
| Simulation truth: true plants, hidden physics, noise-free states | `process_transfer.simulation` | M0 |
| Modeller physics: the simplified equations an engineer would write | `process_transfer.modeller` | M0 (equations only) |
| Data infrastructure: Parquet, DuckDB, SQL, paths | `process_transfer.data` | M0 |
| Configuration and units | `process_transfer.config`, `process_transfer.units` | M0 |
| Models (black box, hybrid) | `process_transfer.models` | M1 |
| Transfer logic (baselines, heuristic, policies) | `process_transfer.transfer` | M2 onwards |
| Evaluation (metrics, physics metrics, plots) | `process_transfer.evaluation` | M1 onwards |

Ground-truth information (true parameters, hidden constitutive relations, noise-free trajectories, exact rates) is stored separately from observable information and never enters training or evaluation inputs, except in an explicit oracle experiment.

## Repository layout

    AGENTS.md, CLAUDE.md        operating rules
    README.md
    pyproject.toml
    .github/workflows/ci.yml    ruff and pytest on every push
    src/process_transfer/       reusable code
    tests/                      pytest, including physical-behaviour tests
    configs/                    plant, modeller and excitation configurations (YAML)
    sql/                        schema and readable data-quality queries
    experiments/                numbered scripts that generate data and run experiments
    docs/                       this documentation
    notebooks/                  exploration only, never primary implementation

Notebooks may analyse results but call package functions; reusable logic lives under `src/`.

## Units

Configuration files use engineering units, each quantity written as a value and a unit string. Values are converted to SI when loaded, through a small explicit conversion table that rejects unknown units. Inside the simulator everything is a plain SI float; no unit-wrapped objects pass through the ODE right-hand side. Units are stored as metadata in the `sensors` table so that data are self-describing.

## Data architecture

    simulation / external data
              |
          raw storage (Parquet)
              |
        SQL database (DuckDB)
              |
       data-quality layer (SQL)
              |
     sensor and time alignment (SQL)
              |
     operating-window selection (SQL)
              |
        ML-ready datasets
              |
           modelling

DuckDB is the initial database: lightweight, serverless, real SQL, strong Parquet integration, reproducible locally and suited to analytical workloads. A production system may later move to PostgreSQL. Raw numerical data are stored as Parquet, not only CSV.

Measurements are stored in long format. The M0 schema has exactly five tables:

    plants               plant_id, name, process_type, created_at
    process_parameters   plant_id, parameter, value, unit   (known parameters only)
    sensors              sensor_id, plant_id, variable_name, unit, sampling_period
    measurements         plant_id, sensor_id, run_id, timestamp, value, quality_flag
    operating_runs       run_id, plant_id, start_time, end_time, operating_mode, description

Experiment and model tables (`model_runs`, `metrics`, `transfer_actions`) arrive with the milestones that need them. Important transformations (quality filtering, time alignment, resampling, window operations, lagged variables, source-target comparisons) are written as readable `.sql` files rather than hidden behind an ORM or replaced by Pandas.

Generated data paths respect the `PT_DATA_DIR` environment variable and are never committed.

## Reproducibility

Every generated dataset and experiment is reproducible from the code version (git hash), the configuration, the random seed, the simulator version and the process parameters. Generated files carry that metadata. Explicit seeds are used wherever randomness is involved, and generated datasets are never edited by hand.

Reproducibility is tested at the level that matters: identical numerical trajectories for the same seed, identical schema, and identical canonical content where appropriate. Tests do not depend on byte-for-byte identity of serialised files or on the serialisation details of a particular PyArrow version.

## Technology stack

* Python 3.12+, NumPy and SciPy (`solve_ivp`) for simulation. Differentiable dynamics later, with PyTorch and possibly `torchdiffeq`.
* Pydantic for configuration validation, YAML for configuration files.
* DuckDB, Parquet, SQL; Pandas where a dataframe is the right tool.
* pytest and ruff.
* Experiment tracking initially in structured files and DuckDB tables; MLflow or similar evaluated later.
* No GUI now. A possible progression is research CLI, internal prototype, FastAPI backend, web frontend.

## Future: process representation

A long-term challenge is a generic representation of a process, moving from a parameter vector to a structured process or physics graph whose nodes and edges represent equipment, streams, states, parameters, mechanisms and learned components, each tagged as known or unknown, shared or plant-specific, learnable or fixed. Initial handling would use NetworkX; graph neural networks are not added until the simpler representation has shown value.

## Future: declarative process definition

A small declarative format (YAML is a candidate) may eventually describe states, inputs, parameters with transferability flags, and unknown terms. It is not designed during M0; models are explicit Python until the abstractions are stable.
