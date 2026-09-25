# Roadmap

Only the active milestone is implemented. Later milestones are recorded so that decisions today do not block them, not as licence to build ahead.

## M0: Virtual plant and data infrastructure (closed)

Goal: reliable source and target CSTR simulation and SQL-backed data infrastructure. No ML.

Deliverables: configurable CSTR with a true plant and a modeller's simplified equations; source and target configurations; verified stable operating point; dynamic input excitation; sensor noise; Parquet output; DuckDB ingestion; SQL data-quality queries; unit tests including tests of physical behaviour; source-versus-target diagnostic plots.

Definition of done: a single reproducible command, or a small documented set, can generate the source and target plants, simulate both, validate the simulations, write Parquet, load DuckDB, run SQL checks, produce ML-ready datasets and basic diagnostic plots, with all tests passing. Complex ML does not start until this foundation is reliable.

**Closed 2026-09-22.** Audited technical reference: commit `91206b2`. Codex's independent audit of that exact commit found no blocking functional defect: the 725-test suite passes, ruff is clean, dependencies match the reference lock file, M0-E05/E06/E07 pass every criterion (the 16 step trajectories of E07 all accepted), M0-E08's anchoring, numerical resolution and physical validity (H1/H2/H3) are correct, and the content hashes of E05, E06 and E07 reproduce the registered ones exactly. CI on that commit is green on both matrix jobs, Python 3.12 and 3.13 (run `35746894043`). The project owner accepted the audit and authorised closure the same day. `docs/m0_audit.md` records the full deliverable-by-deliverable evidence and the closure note.

Left to M1 on purpose, not a missing M0 deliverable: the identifiability of alpha, whose values (0.005 1/K source, 0.002 1/K target) are kept as registered; the data M1 will start from; sensitivity analysis; the benchmark definition.

One non-blocking documented limitation carried into M1's backlog: `tests/test_checks.py`'s exit-code propagation test (around line 232) asserts against the same `all(...)` aggregation logic `experiments/08_oracle_conductance.py` uses, rather than invoking that script's `main()` directly. Codex's audit exercised `main()` directly with controlled in-memory faults and confirmed the real script behaves correctly (invalid balances on A or B, and A outside the envelope, both fail; B outside the envelope alone still permits the diagnostic); the test as written is a fainter regression guard against that specific connection silently breaking in the future. Not fixed here, since this closure iteration changes no code.

## M1: Black-box and hybrid modelling (active)

Target-only baseline, black-box model, hybrid physics-plus-ML model. Does hybrid modelling improve data efficiency or extrapolation?

Started 2026-09-22. Question, comparators and exclusions: D-029. The benchmark is planned in `docs/m1_plan.md`; the owner answered its questions on 2026-09-24 (D-030). I1, the evaluation contract and the mechanistic models MN, MR and MR_F, was implemented on 2026-09-25 (D-032), audited by Codex and corrected the same day, and awaits Codex's review of the last correction; the later iterations have not started.

## M2: Manual transfer learning

Source pretraining, target fine-tuning under adaptation budgets, full and partial fine-tuning, parameter recalibration. First target-data-efficiency curves.

## M3: Domain-shift study

Vary geometry, heat transfer, kinetics, operating ranges and noise systematically; transfer gain against source-target difference; negative-transfer regions.

## M4: Physics-aware transfer heuristic

Explicit rules (preserve, replace, recalibrate, fine-tune, relearn) using sensitivity and identifiability where appropriate.

## M5: Structured process representation

A reusable physics or process graph distinguishing shared structure, target-specific structure, physical parameters and learned mechanisms.

## M6: Learned transfer policy

Whether a learned policy improves on the heuristic. Compare supervised policy learning, contextual bandits, reinforcement learning, meta-learning and Bayesian strategy optimisation before choosing.

## M7: Multiple process families

Beyond CSTRs: batch and fed-batch reactors, heat exchangers, tanks, more complex flowsheets.

## M8: External and public process data

IndPenSim, Tennessee Eastman, DWSIM-generated processes and other public benchmarks, complementing controlled synthetic experiments.

## M9: LLM engineering interface

Document and natural-language interpretation into structured candidate process information, always passing deterministic validation.

## M10: Software package

A clean API and CLI.

## M11: Graphical interface

An engineering workflow from project creation to model export, exposing assumptions rather than hiding them.

## M12: Productisation

PostgreSQL, authentication, project management, deployment, audit logging, model versioning, historian and OPC-UA integration, simulator integrations. Long-term concerns that do not shape early research code.
