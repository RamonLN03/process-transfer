# Audit of M0 against its definition (2026-09-22)

What M0 promised, in `docs/roadmap.md`, `docs/architecture.md` and the decision log, set against what exists at the commit of this file. Three states are kept apart: demonstrated, with where the evidence is; superseded explicitly, by a recorded decision; and pending. An item is not called demonstrated because a later piece of work happens to pass: P3 working and the storage working do not discharge a commitment that neither of them is.

Verdict, stated first. The software path of M0 is complete and has evidence for every part of its definition of done. M0 is a candidate for closure. It is not closed here, because three items need the project owner: two kinds of run promised in D-010 that were never generated and never withdrawn, the provisional value of alpha in D-007, and what data M1 will start from. None of them is a defect of what exists. M1 is not started.

## The deliverables of the roadmap

| Deliverable | State | Evidence, or what is missing |
|---|---|---|
| Configurable CSTR with a true plant and the modeller's simplified equations | demonstrated | `simulation/cstr_true.py`, `modeller/cstr_first_order.py`, strict configuration models; a test on the import graph keeps the two apart |
| Source and target configurations | demonstrated | `configs/source_cstr.yaml`, `configs/target_cstr.yaml`; D-005 to D-008 |
| Verified stable operating point | demonstrated | M0-E01; `tests/test_operating_points.py`; and verified again on every generation by `generation/plants.py`, which refuses a plant whose nominal point is not unique in the scanned range or not stable with the margin of D-017 |
| Dynamic input excitation | demonstrated, within a stated scope | Protocol P3, D-019, `simulation/protocols.py`; M0-E03, M0-E03b and the ten-excursion regression. The scope is the present two plants, A10, 120 s and 600 s; another plant, amplitude, hold or rest needs the same verification |
| Sensor noise | demonstrated | D-020, `configs/sensors_cstr.yaml`, `measurement/`; M0-E04 and H9 of M0-E05 |
| Parquet output | demonstrated | `data/parquet_store.py`; M0-E05 H2, H5 |
| DuckDB ingestion | demonstrated | `data/database.py`, `sql/schema/`; M0-E05 H2, H6 |
| SQL data-quality queries | demonstrated | `sql/quality/`, seven queries; 26 planted defects in the tests, 10 in M0-E05 |
| Unit tests, including tests of physical behaviour | demonstrated | 646 tests, 42 s: steady-state closure, positivity, response direction, integrated balances, stability, reproduction from seeds |
| Source-versus-target diagnostic plots | demonstrated in a basic form | Figure 1 of M0-E04, inputs and true response of both plants; the readings of both plants side by side in M0-E05; `sql/analysis/source_target_comparison.sql` gives the observed differences as numbers. There is no figure of those differences |

## The definition of done

"A single reproducible command, or a small documented set, can generate the source and target plants, simulate both, validate the simulations, write Parquet, load DuckDB, run SQL checks, produce ML-ready datasets and basic diagnostic plots, with all tests passing."

`python -m process_transfer.generation configs/datasets/m0_e05.yaml` does each of these and returns a non-zero code if a mandatory check fails; `python experiments/05_full_data_path.py` adds the second generation, the reader without the private branch, the planted defects and the noise diagnostics. Both were run from a clean commit (M0-E05). "ML-ready" is met in one precise sense: an export of aligned series at the original sampling, with identifiers, units, the time convention, the instrument specification and the known plant parameters, verified bit for bit against what the sensors gave. It has no split into training and evaluation data and no adaptation budget, which D-011 assigns to the experiment code of later milestones.

The criterion of `AGENTS.md`, a simulation and data environment trustworthy enough that later ML results can be believed, is what the reviews and `docs/numerical_robustness.md` have been about. That is a judgement for the owner and the reviewer, not something this file can certify.

## Earlier commitments, one by one

D-010 promised three operating runs per plant. Their state today:

| Commitment of D-010 | State | Detail |
|---|---|---|
| Amplitudes of q and C_Af of 20 % | superseded explicitly | D-018: 10 % (A10) |
| Random binary sequences on a 2 min clock | superseded explicitly | D-019: they take the target to 396 K and must not be used; P3 replaces them |
| Sampling every 0.1 min | kept | `configs/sensors_cstr.yaml` |
| Noise of C_A as 2 % of the nominal concentration | superseded explicitly | D-020: 5 mol/m^3 on both plants |
| Steady operation with noise only | pending | never generated, never withdrawn. The pipeline generates P3 only. D-010 gives no duration |
| Single-input step tests with 10 min holds | pending | never generated as data. M0-E01 and M0-E02 simulated single-input steps of the truth, 40 min from the nominal steady state, to judge the envelope; no observation was made of them. D-019 lists single-input excursions as not decided |

Neither pending run was generated to tick a box. A steady run needs a duration and a purpose: the noise level is an instrument specification that the modeller is given (D-020), so a run for estimating it from data may no longer be needed, or may be wanted as a check. Single-input tests with 10 min holds chain their changes, and M0-E03 showed that chained changes must be verified as sequences, not as steps; a hold of 600 s equals the rest of P3, which suggests they would be safe, and that is a conjecture until it is simulated.

Characterisation of the physical mismatch between truth and model (D-007). D-007 asked M0 to report the size of the mismatch under the planned excitation, so that alpha could be confirmed or raised before M1.

* Demonstrated, at the level of the hidden functions. M0-E03 reports, under P3, over the central 90 % of the samples of 20 sequences of 2 h: r_true / r_model from 1.156 to 0.910 on the source and from 1.326 to 0.992 on the target; UA(T) / UA_ref from 0.972 to 1.032 on the source and from 0.997 to 1.023 on the target. The kinetic mismatch reaches a third; the conductance moves by about 3 % on the source and 2 % on the target.
* Pending. Whether a change of conductance of 2 to 3 % can be seen in readings with sigma_T = 0.5 K has not been examined, and alpha stays provisional. A comparison of trajectories of the modeller's nominal model with the truth would not answer it cleanly: the nominal UA of the modeller is 1.0e5 J/(min K) against a true 0.8e5 on the target, so the difference would be dominated by a parameter that any modeller re-estimates first. Separating the two needs estimation, which is M1, and the owner has placed sensitivity and identifiability at the start of M1.

Other items:

| Item | State | Detail |
|---|---|---|
| Target data comparable in amount to the source (D-011) | demonstrated for the path, not decided for the study | M0-E05 generates the same three runs on both plants. The data that M1 will use, its amount, its seeds and the fixed evaluation set that D-011 mentions, are not defined. M0-E05 says of itself that it is not that data set |
| Reproducibility by content, not by file bytes (D-012) | demonstrated | versioned content hashes; M0-E04 H4, M0-E05 H5 |
| Units at the boundaries (D-013) | demonstrated | `units.py`, `config.py`; SI enforced again where sensors meet states and where data are stored |
| `PT_DATA_DIR`, private repository (D-014) | demonstrated, with a caution | every generated file goes under it. Its default is inside the repository, which is in a synchronised folder; a DuckDB file there is at risk while open. The variable exists to point elsewhere; the default was not changed |
| CI on every push (`AGENTS.md`) | demonstrated | `.github/workflows/ci.yml` runs Ruff and pytest on Ubuntu with Python 3.12 and 3.13, and the suite holds a small storage and SQL round trip. It runs on every push since the history reached the private remote on 2026-09-22. Its first two runs failed on one test, which exposed a stale read from DuckDB's file cache on Linux (`docs/numerical_robustness.md`, eighth review); since `9a16e01` both jobs pass, 691 tests each, seen at `ae5b06a`. Locally the suite also passes on Windows 11 and Python 3.13.7, in the project's environment and in the system's |
| Whether E/R is known or estimable (`docs/assumptions.md`) | pending, and rightly not M0 | it matters from M1; E/R is not exported as a known parameter |
| Steady-state search (D-009) | demonstrated, with stated limits | a count of one is a statement about the scanned range, not a proof of uniqueness |

## What is not verified

* Linux and Python 3.12 are verified by CI since 2026-09-22, on the runners of GitHub: the same 691 tests pass there as on Windows. What CI does not run is the experiments, so the registered content hashes of M0-E04 and M0-E05 are those of the development machine and of no other.
* Agreement of content hashes across library versions: shown not to hold. On the development machine, at one commit, the six runs of M0-E05 have other hashes under numpy 2.3.5 and scipy 1.16.3 than under 2.5.3 and 1.18.1 (experiment log, re-run of 2026-09-22). A data set is reproduced bit for bit by its code, configuration, seeds and environment together, and the environment is recorded with every attempt.
* Behaviour under a power cut during the rename that publishes a data set, and two writers on several machines.

## Questions for the project owner

1. Steady operation with noise only, from D-010: keep it, with what duration and for what use; withdraw it, now that the noise level is a given specification; or defer it.
2. Single-input step tests with 10 min holds, from D-010: keep them, in which case they need the verification that P3 had, as sequences; withdraw them; or defer them. D-019 left single-input excursions undecided.
3. Alpha, provisional in D-007: confirm 0.005 and 0.002 1/K on the evidence that the conductance moves by 2 to 3 % under P3, raise it as D-007 contemplated, or decide it at the start of M1 together with the identifiability study.
4. Answered on 2026-09-22: the private remote `https://github.com/RamonLN03/process-transfer.git` was connected as `origin` and the history pushed at `afd1c51`, without force and onto an empty repository. CI runs on every push and passes on Linux with Python 3.12 and 3.13 since `9a16e01`.
5. The data that M1 starts from: how many runs, which seeds, and the fixed evaluation set of D-011. This can be the first task of M1 and need not hold M0 open.
