# Audit of M0 against its definition (2026-09-22, closed the same day)

*Historical record.* This audit describes the project as it stood at the closure of M0 on 2026-09-22, and its statements about M1 are those of that day. The current state of M1 is in `docs/m1_plan.md` (header and section 3) and in `docs/roadmap.md`. Notes added after the closure are dated.

What M0 promised, in `docs/roadmap.md`, `docs/architecture.md` and the decision log, set against what exists at the commit of this file. Three states are kept apart: demonstrated, with where the evidence is; superseded explicitly, by a recorded decision; and pending. An item is not called demonstrated because a later piece of work happens to pass: P3 working and the storage working do not discharge a commitment that neither of them is.

Verdict, stated first. The software path of M0 is complete and has evidence for every part of its definition of done. The three operating runs that D-010 promised are generated and verified: P3 (M0-E05), steady operation with noise only (M0-E06) and single-input step tests with ten-minute holds (M0-E07). The size of the effect of the temperature-dependent conductance under P3 is characterised (M0-E08), whose verdict gates on the physical validity of the compared pair (D-027), not only on anchoring and numerical resolution. Continuous integration runs on every push and passes on Linux with Python 3.12 and 3.13, and a pinned reference environment (D-028, `docs/reference_environment.md`) lets the registered results be reconstructed separately from CI's rolling dependencies. **M0 is closed.** Two matters are left to M1 on purpose, and neither was a missing deliverable of M0: the identifiability of alpha, whose values are kept for this version, and the data that M1 will start from. M1 is not started.

## Closure

**Closed 2026-09-22. Audited technical reference: commit `91206b2`.** Tag `m0-v1.0` marks the documentary commit that records this closure, one commit later; the reference is `91206b2` because that is the exact commit Codex audited and CI ran green on, and no code, configuration or data changed between it and the closure commit.

Codex's independent audit of `91206b2`, run locally without modifying the repository, found no blocking functional defect. Its own summary, reproduced here rather than paraphrased:

| Check | Result |
|---|---|
| Full suite | 725 tests pass |
| Ruff | clean |
| Dependencies | match the reference lock file, no incompatibility found |
| M0-E05, full data path | passes every criterion |
| M0-E06, steady operation | passes every criterion |
| M0-E07, single-input steps | all 16 trajectories accepted |
| M0-E08, effect of UA(T) | anchoring, numerical resolution and physical validity correct |
| Reproducibility | content hashes of E05, E06 and E07 identical to the registered ones |
| Repository | clean tree |

Two of pytest's first run failed in Codex's isolated environment from a Git ownership protection setting local to that environment; once excepted for the review only, all 725 passed. Codex additionally exercised `experiments/08_oracle_conductance.py`'s `main()` directly with controlled in-memory faults, beyond reading the code: invalid balances on A or B fail the run, A outside the operating envelope fails the run, and B outside the envelope alone still permits the diagnostic and does not fail it. The scientific results are unchanged: maximum differences of about 4.40 K on the source and 1.79 K on the target, with no change to alpha, noise or the physics.

CI for `91206b2` is green on both matrix jobs: [run `35746894043`](https://github.com/RamonLN03/process-transfer/actions/runs/35746894043), `test (3.12)` and `test (3.13)`, status Success, 2m 25s.

**One non-blocking limitation recorded, not fixed here.** `tests/test_checks.py`'s exit-code propagation test (around line 232) asserts against the same `all(...)` aggregation `experiments/08_oracle_conductance.py`'s `main()` uses, rather than invoking `main()` itself; Codex's direct exercise of `main()` (above) confirms the real script behaves correctly, so this is a fainter regression guard against that specific connection silently breaking later, not a present defect. Left for whoever next touches that test, in M1 or later; it does not gate this closure, per the owner's instruction that no code changes with it.

The project owner reviewed Codex's audit and the green CI run and authorised closure and the `m0-v1.0` tag on 2026-09-22.

Post-closure check, 2026-09-24. In an independent audit of `01c6422`, Codex rebuilt `91206b2`, generated the data sets of M0-E05, M0-E06 and M0-E07 in the reference environment on Windows, and found all 24 content hashes equal to those of the code at `01c6422`. The only change to the production code of M0 since `91206b2` is the choice of font of the figures. The audit modified nothing and did not reopen M0; its record is in `docs/docker.md`.

## The deliverables of the roadmap

| Deliverable | State | Evidence, or what is missing |
|---|---|---|
| Configurable CSTR with a true plant and the modeller's simplified equations | demonstrated | `simulation/cstr_true.py`, `modeller/cstr_first_order.py`, strict configuration models; a test on the import graph keeps the two apart |
| Source and target configurations | demonstrated | `configs/source_cstr.yaml`, `configs/target_cstr.yaml`; D-005 to D-008 |
| Verified stable operating point | demonstrated | M0-E01; `tests/test_operating_points.py`; verified again on every generation by `generation/plants.py`; M0-E06 shows the true states staying at it for two hours to within 4.3e-8 mol/m^3 and 2.1e-9 K |
| Dynamic input excitation | demonstrated, within a stated scope | Protocol P3, D-019, `simulation/protocols.py`; M0-E03, M0-E03b and the ten-excursion regression; the single-input steps of M0-E07 with their return, verified as sixteen independent runs. The scope is the present two plants, A10, these holds and rests; another plant, amplitude, hold or rest, or chained step tests, needs the same verification |
| Sensor noise | demonstrated | D-020, `configs/sensors_cstr.yaml`, `measurement/`; M0-E04; H9 of M0-E05, H4 of M0-E06 on constant states, H4 of M0-E07 |
| Parquet output | demonstrated | `data/parquet_store.py`; M0-E05 H2, H5; M0-E06 and M0-E07 H2 |
| DuckDB ingestion | demonstrated | `data/database.py`, `sql/schema/`; M0-E05 H2, H6 |
| SQL data-quality queries | demonstrated | `sql/quality/`, seven queries; 26 planted defects in the tests, 10 in M0-E05 |
| Unit tests, including tests of physical behaviour | demonstrated | the suite, in `tests/`: steady-state closure, positivity, response direction, integrated balances, stability, reproduction from seeds, the protocols and their identities |
| Source-versus-target diagnostic plots | demonstrated in a basic form | Figure 1 of M0-E04, inputs and true response of both plants; the readings of both plants side by side in M0-E05, M0-E06 and M0-E07; the true responses to each input of M0-E07; `sql/analysis/source_target_comparison.sql` gives the observed differences as numbers |

## The definition of done

"A single reproducible command, or a small documented set, can generate the source and target plants, simulate both, validate the simulations, write Parquet, load DuckDB, run SQL checks, produce ML-ready datasets and basic diagnostic plots, with all tests passing."

`python -m process_transfer.generation <definition>` does each of these for any of the three definitions under `configs/datasets/` and returns a non-zero code if a mandatory check fails; `experiments/05_full_data_path.py`, `06_steady_operation.py` and `07_single_input_steps.py` add the checks proper to each protocol. All were run from clean commits. "ML-ready" is met in one precise sense: an export of aligned series at the original sampling, with identifiers, units, the time convention, the instrument specification and the known plant parameters, verified bit for bit against what the sensors gave. It has no split into training and evaluation data and no adaptation budget, which D-011 assigns to the experiment code of later milestones.

The criterion of `AGENTS.md`, a simulation and data environment trustworthy enough that later ML results can be believed, is what the reviews and `docs/numerical_robustness.md` have been about. That is a judgement for the owner and the reviewer, not something this file can certify.

## Earlier commitments, one by one

D-010 promised three operating runs per plant. Their state today:

| Commitment of D-010 | State | Detail |
|---|---|---|
| Amplitudes of q and C_Af of 20 % | superseded explicitly | D-018: 10 % (A10) |
| Random binary sequences on a 2 min clock | superseded explicitly | D-019: they take the target to 396 K and must not be used; P3 replaces them |
| Sampling every 0.1 min | kept | `configs/sensors_cstr.yaml` |
| Noise of C_A as 2 % of the nominal concentration | superseded explicitly | D-020: 5 mol/m^3 on both plants |
| Steady operation with noise only | demonstrated | M0-E06, data set `m0-e06`: two hours per plant at the nominal inputs from the verified steady state, drift of the true states five orders of magnitude inside a tolerance derived from the integrator's precision, cadence and constant inputs, 26 noise scores within 4, nothing hidden. D-026 defines the run |
| Single-input step tests with 10 min holds | demonstrated | M0-E07, data set `m0-e07`: eight independent tests per plant, lead, hold and recovery of 600 s, all sixteen true trajectories accepted with margins of at least 7.7 K to the envelope, the return from the hold included; non-monotone temperature responses on both plants and two-signed ones under flow changes on the target; 240 noise scores within 4. Not verified: chaining the tests into one trajectory, which D-026 excludes |

Characterisation of the physical mismatch between truth and model (D-007). D-007 asked M0 to report the size of the mismatch under the planned excitation, so that alpha could be confirmed or raised before M1.

* Demonstrated, at the level of the hidden functions. M0-E03 reports, under P3, over the central 90 % of the samples of 20 sequences of 2 h: r_true / r_model from 1.156 to 0.910 on the source and from 1.326 to 0.992 on the target; UA(T) / UA_ref from 0.972 to 1.032 on the source and from 0.997 to 1.023 on the target.
* Demonstrated, at the level of the trajectories, for the conductance alone. M0-E08 compares the true plant with the same plant under a constant conductance anchored at UA(T_nominal), under the three P3 sequences of M0-E05: the trajectories differ by up to 4.4 K and 34 mol/m^3 on the source and 1.8 K and 8.8 mol/m^3 on the target at the peaks of the hot excursions, several sigmas of D-020 there, and by about one sigma in root mean square over a run on the source and half a sigma on the target. Over the whole trajectories UA(T)/UA(T_nominal) runs from 0.952 to 1.079 on the source and from 0.975 to 1.042 on the target; the ranges of M0-E03 left out the peaks. Its verdict now also requires the compared pair to be physically valid (D-027): the primary trajectory fully accepted, the secondary finite, physical and balance-closed, exempt only from its own operating envelope, which stays a reported diagnostic. Re-run from the clean commit `352c9aa`, all three verdicts hold and every number above is unchanged.
* Kept, and left to M1. The values alpha_source = 0.005 1/K and alpha_target = 0.002 1/K are kept for this version. Whether the effect can be separated from a re-estimated constant UA and from the noise, and whether it is useful for the comparison of models, are questions of identifiability that the owner placed at the start of M1; M0-E08 gives the scale of the effect and no more.

Other items:

| Item | State | Detail |
|---|---|---|
| Target data comparable in amount to the source (D-011) | demonstrated for the path, not decided for the study | every data set of M0 has the same runs on both plants. The data that M1 will use, its amount, its seeds and the fixed evaluation set that D-011 mentions, are not defined. The data sets of M0 are verifications of the path and of the protocols; none of them becomes the benchmark of M1 by default |
| Reproducibility by content, not by file bytes (D-012) | demonstrated | versioned content hashes; M0-E04 H4, M0-E05 H5; the six hashes of M0-E05 reproduced at `a58afe8`, `b754d14` and `6578f45` |
| Units at the boundaries (D-013) | demonstrated | `units.py`, `config.py`; SI enforced again where sensors meet states and where data are stored |
| `PT_DATA_DIR`, private repository (D-014) | demonstrated, with a caution | every generated file goes under it. Its default is inside the repository, which is in a synchronised folder; a DuckDB file there is at risk while open. The variable exists to point elsewhere; the default was not changed |
| CI on every push (`AGENTS.md`) | demonstrated | `.github/workflows/ci.yml` runs Ruff and pytest on Ubuntu with Python 3.12 and 3.13. Its first two runs failed on one test, which exposed a stale read from DuckDB's file cache on Linux (`docs/numerical_robustness.md`, eighth review); since `9a16e01` both jobs pass. The result of the last push is recorded in the delivery of each iteration; a push is not a passed CI |
| Whether E/R is known or estimable (`docs/assumptions.md`) | pending, and rightly not M0 | it matters from M1; E/R is not exported as a known parameter |
| Steady-state search (D-009) | demonstrated, with stated limits | a count of one is a statement about the scanned range, not a proof of uniqueness |
| Minimum DuckDB version | demonstrated | D-025: 1.3, from the test suite run on a clean worktree under five versions in isolated environments |
| M0-E08's verdict gates on physical validity, not only anchoring and resolution | demonstrated | D-027: `TrajectoryCheck.physically_valid`, `checks.comparison_is_valid`, `H3_valid` in the script's verdicts; regression tests in `tests/test_checks.py`; re-run at `352c9aa` |
| A pinned reference environment to reconstruct the registered results | demonstrated, for one machine | D-028: `requirements-reference.lock.txt`, `docs/reference_environment.md`; installed and validated in a clean isolated environment on the machine that captured it, not tested on another system |

## What is not verified

* Chained step tests: the eight tests of M0-E07 are independent runs, and one trajectory of eight steps was never simulated (D-026).
* The effect of UA(T) under other excitations than the P3 sequences of M0-E05, the steady run and the single-input steps included.
* Agreement of content hashes across library versions: shown not to hold. On the development machine, at one commit, the six runs of M0-E05 have other hashes under numpy 2.3.5 and scipy 1.16.3 than under 2.5.3 and 1.18.1 (experiment log, re-run of 2026-09-22). A data set is reproduced bit for bit by its code, configuration, seeds and environment together, and the environment is recorded with every attempt. CI does not run the experiments.
* Behaviour under a power cut during the rename that publishes a data set, and two writers on several machines.

## What M0 hands to M1, and what it does not do

* The values of alpha are kept for this version: 0.005 1/K on the source and 0.002 1/K on the target. Their identifiability, and their usefulness for the comparison of models, are studied in M1, with M0-E08 as the measure of the effect they produce under P3.
* The data sets `m0-e05`, `m0-e06` and `m0-e07` are verifications of the path and of the protocols. They do not become the benchmark of M1 automatically; the runs, seeds and fixed evaluation set of M1 are a decision of M1.
* The selection of estimable parameters, the adaptation budgets of D-011 and the fixed evaluation set are not implemented in M0.

## Questions for the project owner

1. ~~Closure.~~ **Resolved 2026-09-22.** Codex's audit of `91206b2` found no blocking defect, CI was confirmed green on that exact commit, and the owner authorised closure. See Closure above.
2. Alpha, provisional in D-007: kept at 0.005 and 0.002 1/K for this version, with the effect measured in M0-E08. Whether to confirm the values or to decide them at the start of M1 with the identifiability study is the owner's call; nothing in M0 depends on it.
3. The data that M1 starts from: how many runs, which seeds, and the fixed evaluation set of D-011. This is the first task of M1 and need not hold M0 open.
