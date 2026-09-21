# Roadmap

Only the active milestone is implemented. Later milestones are recorded so that decisions today do not block them, not as licence to build ahead.

## M0: Virtual plant and data infrastructure (active)

Goal: reliable source and target CSTR simulation and SQL-backed data infrastructure. No ML.

Deliverables: configurable CSTR with a true plant and a modeller's simplified equations; source and target configurations; verified stable operating point; dynamic input excitation; sensor noise; Parquet output; DuckDB ingestion; SQL data-quality queries; unit tests including tests of physical behaviour; source-versus-target diagnostic plots.

Definition of done: a single reproducible command, or a small documented set, can generate the source and target plants, simulate both, validate the simulations, write Parquet, load DuckDB, run SQL checks, produce ML-ready datasets and basic diagnostic plots, with all tests passing. Complex ML does not start until this foundation is reliable.

Status, 2026-09-22. The software path is complete and M0 is a candidate for closure; it is not closed. `docs/m0_audit.md` sets every deliverable against its evidence and lists what waits for the project owner: two kinds of run promised in D-010 and never generated, the provisional alpha of D-007, a remote repository so that CI can run, and the data that M1 will start from.

## M1: Black-box and hybrid modelling

Target-only baseline, black-box model, hybrid physics-plus-ML model. Does hybrid modelling improve data efficiency or extrapolation?

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
