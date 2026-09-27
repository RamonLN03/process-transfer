# Architecture

## Separation of concerns

The package is organised so that these remain logically distinct (AGENTS.md):

| Concern | Package | Milestone |
|---|---|---|
| Simulation truth: true plants, hidden physics, noise-free states | `process_transfer.simulation` | M0 |
| Modeller physics: the simplified equations an engineer would write | `process_transfer.modeller` | M0 (equations only) |
| Measurement: sensors, and the observations that modelling is given | `process_transfer.measurement` | M0 |
| Generation: from configuration files to a verified data set; the truth side of the data path | `process_transfer.generation` | M0 |
| Data infrastructure: Parquet, DuckDB, SQL, paths | `process_transfer.data` | M0 |
| Configuration and units | `process_transfer.config`, `process_transfer.units` | M0 |
| Models: the mechanistic MN, MR and MR_F since I1; black box and hybrid later | `process_transfer.models` | M1 |
| Transfer logic (baselines, heuristic, policies) | `process_transfer.transfer` | M2 onwards |
| Evaluation: windows, budgets, metrics, failures and physical checks since I1 | `process_transfer.evaluation` | M1 onwards |

Ground-truth information (true parameters, hidden constitutive relations, noise-free trajectories, exact rates) is stored separately from observable information and never enters training or evaluation inputs, except in an explicit oracle experiment.

## From the truth to observations

    true trajectory, every 0.1 s          simulation.integration
              |
    acceptance checks of the truth        simulation.checks
              |
    exact values at the sensor instants   simulation.operating_run
              |
    readings                              measurement.sensors (knows no plant)
              |
    Observations        RunTruth
    for modelling       diagnostics only

The boundary is in the interfaces, not only in the documentation. `measurement` never imports `simulation` or `modeller`, and a test on the import graph enforces it. A sensor is handed the values to measure and an instrument specification; it has no access to a rate law, a parameter or the simulator. `Observations` has a closed list of fields, pinned by a test: instants, readings, known inputs, names, units, the sampling period and the noise level of the instruments. `RunTruth` holds what no model may see: the true trajectory, its acceptance checks, the exact values, the measurement errors and the seed of the noise. The seed is on that side on purpose, since it would allow the noise to be regenerated and subtracted.

The criteria for true states, physical bounds and closed balances, apply to the truth and are checked before anything is observed. They are never applied to readings, and readings are never clipped or corrected.

A row of observations holds an instant, the readings of the state at that instant, and the inputs applied from that instant on (zero-order hold, right-continuous). The sensors read stored samples of the truth, never interpolated ones, and every change of the inputs must fall on a row.

## Repository layout

    AGENTS.md, CLAUDE.md        operating rules
    README.md
    pyproject.toml
    .github/workflows/ci.yml    ruff and pytest on every push, and a build and run of the container
    Dockerfile, .dockerignore   the Linux container of the data path (docs/docker.md, D-031)
    docker/                     its entry point and its lock of Python packages
    src/process_transfer/       reusable code
    tests/                      pytest, including physical-behaviour tests
    configs/                    plant, modeller and sensor configurations, and data set definitions (YAML)
    sql/                        schema, data-quality queries, views and analyses (sql/README.md)
    experiments/                numbered scripts that generate data and run experiments
    docs/                       this documentation
    notebooks/                  exploration only, never primary implementation

Notebooks may analyse results but call package functions; reusable logic lives under `src/`.

## Units

Configuration files use engineering units, each quantity written as a value and a unit string. Values are converted to SI when loaded, through a small explicit conversion table that rejects unknown units. Each field declares the physical dimension it requires, and the SI value itself is checked, since a finite value can overflow or underflow in the conversion. Inside the simulator everything is a plain SI float; no unit-wrapped objects pass through the ODE right-hand side. Units are stored as metadata in the `sensors` table so that data are self-describing.

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

Measurements are stored in long format. The M0 schema has exactly five tables, whose columns, identities, time convention and quality flag are fixed in `docs/data_contract.md` (D-021 to D-024):

    plants               plant_id, name, process_type
    process_parameters   plant_id, parameter, value, unit   (known parameters only)
    sensors              sensor_id, plant_id, variable_name, channel_kind, channel_index,
                         unit, sampling_period_s, noise_model, noise_std
    operating_runs       run_id, plant_id, dataset_id, operating_mode, description,
                         start_time_s, end_time_s, sampling_period_s, n_samples,
                         content_sha256
    measurements         plant_id, run_id, sensor_id, sample_index, time_s, value,
                         quality_flag

`sensors` describes channels: measured variables and the four known inputs, told apart by `channel_kind`. No table holds a wall-clock time; process time is relative seconds and an integer tick.

Experiment and model tables (`model_runs`, `metrics`, `transfer_actions`) arrive with the milestones that need them. Important transformations (quality filtering, time alignment, resampling, window operations, lagged variables, source-target comparisons) are written as readable `.sql` files rather than hidden behind an ORM or replaced by Pandas.

Generated data paths respect the `PT_DATA_DIR` environment variable and are never committed.

How the layer is built (M0-E05). `generation` builds the plants, verifies their starting points, simulates, validates the truth, observes, and hands `Observations` and known plant records to `data`. The runs it generates are the three operating runs of D-010, P3, steady operation and single-input steps, defined in `simulation/protocols.py` and told apart by their identities (D-026); a data set definition names one protocol and its settings. `data/parquet_store.py` writes an immutable data set by staging and rename and verifies it on reading, identifiers, relations and quality flag included; `data/database.py` ingests a run in one transaction behind a staging schema and the queries of `sql/quality/`, and checks the database as a whole before committing; `data/export.py` writes aligned series taken from the SQL view `aligned_series`, verified against the stored content, and verifies an export again whenever one is opened; `generation/leak_scan.py` reads the result the way a model would and looks for anything hidden. The writers are never handed the truth: their signatures take `Observations` and a `PlantSpec`, and tests on the import graph keep `data` and `measurement` from importing `simulation`.

    PT_DATA_DIR/available/datasets/<id>/    Parquet and manifest.json, written once
    PT_DATA_DIR/available/databases/        DuckDB, derived, can be rebuilt
    PT_DATA_DIR/available/exports/<id>/     aligned series for models
    PT_DATA_DIR/private/datasets/<id>/      seeds, streams, full configurations, truth checks

One command runs the whole path and returns a non-zero code if a mandatory check fails:

    python -m process_transfer.generation configs/datasets/m0_e05.yaml

The SQL is described in `sql/README.md`, and what M0 has and has not delivered in `docs/m0_audit.md`.

Experiments write to `PT_DATA_DIR/experiments/<experiment>/<run id>/`, one directory per run, never reused: figures, a `summary.json` with a provenance block, and copies of the configuration files. These are diagnostic artefacts of the simulator. They may contain hidden parameters, since the true-plant configurations are copied in full, and they stay apart from the data that will later be made available for training or adaptation, which never carries ground truth.

## Reproducibility

Every generated dataset and experiment is reproducible from the code version (git hash), the configuration, the random seed, the simulator version and the process parameters. Generated files carry that metadata. `data/provenance.py` records the commit and the state of the working tree, fingerprints of the configurations, the environment and the settings of the run. When git is missing, or the working tree has local changes, the run says so and is not presented as identified by its commit. Explicit seeds are used wherever randomness is involved, and generated datasets are never edited by hand.

Randomness is always named. The excitation draws from `numpy.random.default_rng(excitation seed)` and the sensor noise from `numpy.random.SeedSequence(entropy=sensor seed, spawn_key=(plant index, run index, channel))`. The two generators share nothing and no global generator is used, so neither can shift the other: another sensor seed changes the readings and nothing else. Two runs with the same seed and stream replay the same noise, which is what makes a run reproducible and a mistake when the runs are meant to differ; every run needs its own stream.

Reproducibility is tested at the level that matters: identical numerical trajectories for the same seed, identical schema, and identical canonical content where appropriate. Tests do not depend on byte-for-byte identity of serialised files or on the serialisation details of a particular PyArrow version.

## Technology stack

* Python 3.12+, NumPy and SciPy (`solve_ivp`) for simulation. Differentiable dynamics for the learned models of M1: JAX, chosen in I3 from a comparison with PyTorch on development data (D-035), as the optional extra `learning`. PyTorch with `torchdiffeq` was the charter's first guess, not a decision.
* Pydantic for configuration validation, YAML for configuration files.
* DuckDB, Parquet, SQL; Pandas where a dataframe is the right tool.
* pytest and ruff.
* Docker, for a Linux container that runs the data path on the CPU (D-031, `docs/docker.md`). It is a way to run the existing code, not a deployment.
* Experiment tracking initially in structured files and DuckDB tables; MLflow or similar evaluated later.
* No GUI now. A possible progression is research CLI, internal prototype, FastAPI backend, web frontend.

## Future: process representation

A long-term challenge is a generic representation of a process, moving from a parameter vector to a structured process or physics graph whose nodes and edges represent equipment, streams, states, parameters, mechanisms and learned components, each tagged as known or unknown, shared or plant-specific, learnable or fixed. Initial handling would use NetworkX; graph neural networks are not added until the simpler representation has shown value.

## Future: declarative process definition

A small declarative format (YAML is a candidate) may eventually describe states, inputs, parameters with transferability flags, and unknown terms. It is not designed during M0; models are explicit Python until the abstractions are stable.
