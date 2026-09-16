# AGENTS.md

## Project

ProcessTransfer is a long-term research and software project investigating physics-aware transfer of hybrid process models from data-rich source systems to data-scarce target systems.

The project may eventually become an engineering-facing software product, but current development must remain driven by the research questions.

## Language

The user may communicate with coding agents in Spanish.

Everything committed to the repository must be written in English, including:

* source code;
* variable and function names;
* comments;
* docstrings;
* tests;
* documentation;
* configuration files;
* commit messages;
* issue text generated for the repository.

## Start of Every Session

Before modifying the repository:

1. Run `git status`.
2. Read the recent history with `git log --oneline`.
3. Read this file.
4. Read `CLAUDE.md` when using Claude Code.
5. Inspect relevant documentation under `docs/`.
6. Inspect existing code before proposing replacement functionality.

Never assume that an earlier conversational description reflects the current repository state.

The repository is the implementation source of truth.

## Active Milestone

Do not implement future milestones unless explicitly requested.

The initial active milestone is:

M0 - Virtual Plant and Data Infrastructure

M0 is about trustworthy simulation, data generation, validation, SQL infrastructure and reproducibility.

It is not an ML milestone.

## Scientific Discipline

Always distinguish:

* hypothesis;
* method;
* result;
* interpretation.

Do not write conclusions before results exist.

Do not assume a sophisticated method will outperform a simple baseline.

Negative results are valid and useful.

Every major method should eventually be compared against the simplest credible alternative.

## Ground Truth vs Available Knowledge

The synthetic simulator intentionally knows more physics than the modeller.

Maintain a strict boundary between:

### Simulation truth

May contain:

* true parameters;
* hidden constitutive relationships;
* true kinetics;
* hidden heat-transfer behaviour;
* noise-free state trajectories;
* exact derivatives.

### Observable/model-available information

May contain only information that would realistically be available to an engineer or measured from the plant.

Ground-truth information must never leak into training or evaluation inputs unless a specific oracle experiment explicitly requires it.

## Initial Hidden Physics

The first model-mismatch study uses:

### Shared hidden physics

A kinetic relationship shared by source and target that is more complex than the simplified kinetic equation available to the modeller.

This provides a genuinely transferable unknown mechanism.

### Plant-specific hidden physics

Heat-transfer behaviour that differs between source and target and is not fully captured by the modeller's simplified constant-parameter representation.

### Deferred complexity

Do not initially add:

* hidden side reactions;
* sensor bias;
* structural reaction-network mismatch.

A hidden secondary reaction is a later difficulty level.

## Modelling Direction

The primary future model architecture is a continuous-time hybrid ODE.

Prefer meaningful learned physical components, such as:

* kinetic corrections;
* heat-transfer corrections;

over arbitrary opaque residual networks whenever possible.

A discrete one-step predictor may later be implemented as a baseline, but it is not the main conceptual architecture.

## Baselines

Future work must include at least:

1. Oracle model using complete true physics.
2. Mechanistic model with target parameter re-estimation.
3. Target-only black-box model.
4. Target-only hybrid model.
5. Standard transfer model.
6. Hybrid transfer model.
7. Physics-aware manual transfer heuristic.

Any later learned transfer policy must beat or meaningfully complement the manual heuristic.

## Coding Rules

Prefer clarity over cleverness.

Avoid premature abstraction.

Do not create infrastructure for hypothetical future features unless needed by the active milestone.

Keep:

* simulator code;
* modeller physics;
* data infrastructure;
* ML models;
* transfer logic;
* evaluation;

as logically distinct concerns.

Reusable logic belongs under `src/`.

Notebooks are for exploration and analysis, not primary implementation.

Use type hints for important interfaces.

Document scientifically meaningful assumptions.

## Physics

Equations must remain easy to inspect.

Numerical output alone is not evidence of physical correctness.

Add tests for physical behaviour wherever practical.

Examples include:

* steady-state residual closure;
* state positivity where required;
* expected response direction;
* integrated mass balance;
* integrated energy balance where applicable;
* stable operating-point verification;
* deterministic reproduction with fixed seeds.

Do not silently clip unstable or physically invalid behaviour merely to make tests pass.

## Stability

Initial data generation should occur around a verified stable operating point.

Verify stability from the local Jacobian eigenvalues rather than assuming it.

Open-loop excitation should remain inside a documented safe operating envelope.

## Units

Use explicit units at configuration and data boundaries.

Internal numerical simulation should use plain SI floating-point values.

Do not pass unit-wrapped objects through the ODE right-hand side.

## Data

Initial storage:

* Parquet for raw/generated numerical data;
* DuckDB for analytical storage and SQL access.

Use long-format measurement storage.

Initial database entities:

* `plants`
* `process_parameters`
* `sensors`
* `measurements`
* `operating_runs`

Do not create future experiment/model tables before they are needed.

Generated data paths must respect the `PT_DATA_DIR` environment variable.

Keep generated data out of Git.

## SQL

SQL is a genuine part of the project architecture.

Do not replace every SQL transformation with Pandas.

Store important analytical/data-quality transformations as readable SQL where appropriate.

SQL should eventually support:

* quality filtering;
* time alignment;
* resampling;
* window operations;
* lagged measurements;
* source-target comparisons.

## Reproducibility

Experiments and generated data should be reproducible from:

* code version;
* configuration;
* random seed;
* simulator version;
* process parameters.

Use explicit random seeds.

Do not manually edit generated datasets.

## Testing and CI

pytest must be used from M0.

CI must run automatically on repository pushes.

A change is not complete merely because it works interactively.

Relevant tests must pass.

## Git

Read `git status` and recent history before every working session.

Make small, meaningful commits.

Do not bundle unrelated changes in a single commit.

Do not overwrite other work without inspecting the repository state.

When multiple agents or humans may modify the repository, minimise collision risk by:

* working on narrow tasks;
* checking recent changes first;
* avoiding broad rewrites without discussion.

## Multi-Agent Coordination

Claude Code is the primary repository writer and implementation agent.

Codex acts by default as a read-only research reviewer, software reviewer, and architectural collaborator.

Codex must not modify, stage, commit, revert, merge, rebase, or otherwise change repository files unless the user explicitly transfers implementation ownership to Codex.

Only one agent may have repository write ownership at a time.

When Codex identifies an issue, it should report:

* the relevant commit or file;
* the problem;
* why it matters;
* the proposed correction.

The user decides whether the correction should be implemented. Implementation normally returns to Claude Code.

Both agents must inspect `git status` and recent git history before evaluating the repository.

Agents must never silently repair or overwrite each other's work.

## Research Complexity

Do not add a technique because it sounds advanced.

In particular, do not add the following until their milestone justifies them:

* reinforcement learning;
* LLM agents;
* graph neural networks;
* meta-learning;
* GUI frameworks;
* cloud deployment;
* simulator integrations.

A simpler method should be preferred when it answers the same research question more cleanly.

## Future RL Rule

If reinforcement learning is eventually considered, first determine whether the transfer decision is genuinely sequential.

Compare RL against simpler alternatives such as:

* deterministic heuristics;
* supervised policy learning;
* contextual bandits;
* Bayesian strategy optimisation.

RL must earn its complexity.

## Future LLM Rule

LLMs may eventually interpret engineering information, documentation and variable mappings.

LLM-generated physical content must always pass deterministic validation before entering a process model.

Principle:

LLMs interpret; mathematics validates.

## Documentation

Important scientific and architectural decisions must be recoverable later.

Keep documentation updated when assumptions change.

Prefer documenting why a decision was made, not only what the code does.

## Current Priority

Build a reliable experimental foundation before building intelligent automation.

The success criterion for M0 is not an impressive demo.

It is a simulation and data environment trustworthy enough that later ML results can be believed.
