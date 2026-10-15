# M1-E03, the benchmark of M1: draft registration

**Status: draft for review, not registered.** Written on 2026-10-15 for I5. Every choice below is labelled:

* **fixed**: settled by the plan or by a recorded decision, cited with its entry;
* **proposed**: pending approval, with the reason for it.

Nothing proposed here is approved by being written down. When the decisions of section 12 have been approved, the registration is copied into `docs/experiment_log.md` as M1-E03 and committed from a clean tree. Its data set definitions move from `configs/datasets/m1_e03_draft/` to `configs/datasets/`. The registration is then offered to Codex for review, and only after that does I6 begin (`docs/m1_plan.md`, section 13). No benchmark data exist, no benchmark model has been trained, and nothing below has been run.

It builds on `docs/m1_i5_proposal.md`, whose points it carries forward, and on the plan, which it does not repeat.

## 1. What the benchmark asks

The question of M1 (D-029; plan, section 1): does a continuous-time hybrid model improve data efficiency or extrapolation over simple alternatives, when every model is fitted on data of the target plant only? The hypotheses are H1 to H5 of section 10 of the plan, written on 2026-09-22 after the exploratory check of section 7.5. The primary comparisons are those of section 9.7: HK against MR and HK against BN, on test-A10 at every budget and on the extrapolation test.

## 2. Data sets

**Proposed definitions**, in `configs/datasets/m1_e03_draft/`. All five are on the target only (plan, section 5.2) and use the sensors of D-020.

| Set | Protocol | Runs | Excitation seeds | Master seed | Identities | Rows a run | Role | Generated in |
|---|---|---|---|---|---|---|---|---|
| train-A10 | P3, lead 60 s, A10 | 10 x 40 excursions | 31001 to 31010 | 20261101 | `target.p3.e31001.x40.l60.a10.n0` ... | 4811 | development: fitting and validation | I6 |
| train-A5 | P3, lead 60 s, A5 | 10 x 40 excursions | 32001 to 32010 | 20261102 | `target.p3.e32001.x40.l60.a5.n0` ... | 4811 | development, for extrapolation | I6 |
| test-A10 | P3, lead 60 s, A10 | 8 x 10 excursions, 80 windows | 33001 to 33008 | 20261103 | `target.p3.e33001.x10.l60.a10.n0` ... | 1211 | interpolation and extrapolation | I7, after the freeze |
| test-A5 | P3, lead 60 s, A5 | 4 x 10 excursions, 40 windows | 34001 to 34004 | 20261104 | `target.p3.e34001.x10.l60.a5.n0` ... | 1211 | in-region reference of the models trained at A5 | I7, after the freeze |
| test-steps | the eight steps of D-026 | 8 x 30 min | none | 20261105 | `target.step.q-up.l600.h600.r600.n1` ... | 301 | protocol shift | I7, after the freeze |

The development sets hold 577 320 measurements and the test sets 101 640, as section 5.2 of the plan gives.

* The excitation and master seeds are new: none was used by an earlier experiment, test or smoke run. test-steps uses noise realisation 1, since `m0-e07` used 0.
* The noise stream of each run follows from its identity and the master seed of its set (D-021). The 40 runs have distinct identities and distinct streams.
* No run, window or reading is shared between sets, or with the data sets of M0.

`tests/test_m1_e03_definitions.py` checks all of this without simulating or generating anything: every definition loads, every run has its identity, inputs, duration and number of rows, every switch falls on the clock of the sensors, the sizes are those of the plan, and the seeds and streams are as stated.

The repository is public, so these seeds are public. The test sets are not protected by keeping their seeds secret. They are protected by the procedure:

* they are generated after the freeze and evaluated once (plan, section 9.8);
* the code that fits and evaluates never reads a configuration or the private branch, which `tests/test_boundaries.py` enforces.

Generated data stay outside git, under PT_DATA_DIR.

## 3. Budgets, windows, fitting and validation

**Fixed** by the plan:

* the index contract (section 4.3);
* the budgets b = 2, 5, 10, 20 and 40, as prefixes of each development run (section 5.4);
* the division of each budget into F and V, with n_V = max(1, floor(b / 5)) (section 5.5);
* what each stage may read (section 5.6);
* normalisation on F only (section 5.8).

A replicate is one run of a development set. Replicate r, counted from 0, is the run with the r-th excitation seed of its set.

**Proposed** conditions, as the pilot's estimate of cost was sized:

* A10 at b = 2, 5, 10, 20 and 40;
* A5 at b = 10 and 40;
* ten replicates of each.

## 4. Models and training

| Item | Status | Value |
|---|---|---|
| MN | fixed (plan, 8.1) | the textbook values, nothing fitted |
| MR, the comparator | fixed (plan, 8.1; D-032, D-033) | k_350, E/R and UA on all b windows; the five starts of `models.fitting.DEFAULT_STARTS`; `FitSettings` as defined: at most 100 evaluations a start, tolerances 1e-10; the lowest endpoint, the first in declared order on a tie |
| MR_F | fixed (plan, 5.5) | the same on F alone; the start of every hybrid |
| BL | fixed (plan, 8.3; D-036) | on all b windows, input effect on the excited directions only |
| Grid | proposed | four configurations per family, in this declared order: BN with two layers of (16, 16) or (32, 32); HK, HU and HKU with one layer of 8 or 16; each size with a penalty of 1e-4, then 1e-2 |
| Adam | proposed | rates BN 1e-3, HK 1e-2, HU 1e-3, HKU 1e-2; beta1 0.9, beta2 0.999, epsilon 1e-8 |
| Steps | proposed | 3000, with a checkpoint on V at step 0 and every 100 steps, 31 in all |
| Training scheme | proposed | fixed-step RK4 with four steps per row, passed explicitly as `TrainingSettings(substeps=4)`. The default of 2 is left in the code as the history of I3 and is not used. Every summary records the value it ran with. |
| Precision | proposed | double precision (x64), recorded with the backend and the versions of jax and jaxlib by `learning_environment()` (D-040) |
| Selection | fixed (plan, 5.5; D-036) | the checkpoint with the lowest J on V by the reference rollout, the earliest on a tie; then the configuration with the lowest selected criterion, the first in declared order on a tie |
| Refit after selection | fixed (plan, 5.5) | none: the selected checkpoint, fitted on F, is the model |
| Start of the hybrids | fixed (plan, 5.5) | MR_F of the same replicate and budget, fitted on F alone; the correction at the identity |
| Failures | proposed | the present policy (D-036, D-037). A loss, gradient or state of Adam that is not finite ends the training as a failure, and its earlier checkpoints are not used. A grid in which every configuration fails is a training failure of that model, replicate and budget. Nothing is retried with another seed or setting. |
| E/R held | proposed | MR and HK with E/R = 8750 K (D-030), at A10 and b = 5 and 40. MR with the three starts of `STARTS_WITH_FIXED_ACTIVATION`; HK from MR_F fitted with E/R held, with the same grid, rates and settings |
| Reference rollout | fixed (I1) | LSODA, rtol 1e-8, atol 1e-6, the same for every model in selection and evaluation |

The pilot is the source of the proposed grid, rates and steps (experiment log, development runs of I3). It ran on the data of M0 at budgets up to nine windows, at A10 only, so it informs these choices and does not validate them for A5 or for b = 20 and 40. In the pilot, size and penalty barely separated the configurations: median differences of the criterion of at most 0.005. It provides no evidence for a larger grid. Four steps per row kept the scheme within 3e-3 sigma of its equation on the fastest windows met, where two steps departed by 0.05 sigma (D-036, clarification of 2026-10-01).

## 5. Seeds

| Seed | Status | Value and rule |
|---|---|---|
| Excitation | proposed | one block per set, section 2; the corners of a run follow from its seed alone (`p3_corners`) |
| Noise | proposed | one master seed per set, section 2; the stream of a run follows from its identity (D-021) |
| Initial weights | proposed | `training_seed(base, replicate, configuration)` with base 20261111 for the trainings on train-A10, the analysis with E/R held included, and 20261112 for those on train-A5. The replicate is its position in the set, from 0. The configuration is its position in the declared grid of its family, from 0 |
| Bootstrap | proposed | 20261121, for every resampling of section 7.6 |
| MR, MR_F, BL | fixed | none: deterministic from their declared starts |

What these rules imply, stated so that they are chosen knowingly:

* **The same initial weights at every budget of a replicate.** `training_seed` does not depend on the budget (D-036), and this is kept. The budgets of a replicate are prefixes of one run, so their data are nested and their results were never independent. With the same initial weights, the curve of a replicate over the budgets changes with the data alone. No test treats the budgets of a replicate as independent (section 7.4).
* **The same initial weights for the analysis with E/R held.** It shares the base of train-A10, so HK with E/R held starts from the same networks as HK with E/R estimated, and the paired difference isolates the treatment of E/R.
* **The same network weights across hybrid families.** HK's kinetic network, HU's thermal network and the first network of HKU have the same shape. With the same configuration index they start from the same weights; their last layer is zero, so each factor starts at one. This is kept: it pairs the ablations. A seed per family would remove the pairing, and the choice is listed in section 12.
* **No reuse of the pilot.** The bases differ from the pilot's, 20260929, so no benchmark training reuses a pilot initialisation.

## 6. Evaluation

**Fixed** by the plan, sections 9.1 to 9.3:

* the score J of every model, replicate, budget and test set, pooled over every scored reading of that test set, by the reference rollout from each window's context mean;
* the phases;
* failures and physical violations: a training failure, or a test window whose rollout fails, leaves that model without a score on that test set for that replicate and budget. The replicate is not dropped.

**Proposed** primary quantity: E = J^2 - 1, the estimate of the excess mean squared error against the true states, in units of sigma^2 averaged over the two channels (section 9.1). It is preferred to J for comparisons because it is additive over readings and, on test runs, an unbiased estimate of that excess. It is reported as it comes, negative values included.

## 7. Hypotheses and the rules for reading them

Everything in this section is **proposed**, before any benchmark data exist.

### 7.1 Common rules

* **Unit.** The development replicate. For each primary comparison, each condition and each budget, there are ten paired outcomes, one a replicate.
* **Paired difference.** For HK against a comparator C on replicate r, d_r = E_C - E_HK, so a positive d_r favours HK.
* **Outcome of a pair.**
  * HK is better if it has a score and C has none, or if both have scores and d_r > 0.
  * C is better in the converse cases.
  * A tie is two failures, or exactly equal scores. Ties are reported and left out of the sign test.
  * A failure is never given an invented score (plan, section 9.3).
* **Test.** The exact one-sided sign test of the pairs that are not ties, in the direction the hypothesis states.
* **Relevance.** A difference counts only if the median of d_r over the replicates is at least delta = 0.05. A failure of C counts as +infinity and a failure of HK as -infinity; ties are left out.
  * Why 0.05: from the noise alone, E on test-A10 has a standard error of about sqrt(2 / 17 600) = 0.011, over 80 windows, 110 readings and two channels; on test-A5 it is about 0.015. A margin of 0.05 is three to five of those.
  * In physical terms, an excess of 0.05 sigma^2 is an RMS error against the truth of about 0.22 sigma per channel, 0.11 K and 1.1 mol/m^3.
  * The margin is an argument of scale, not a value fitted to anything. Its choice is listed in section 12.
* **Verdict at one budget.**
  * Supported: the test rejects at its level, and the median of d_r is at least delta.
  * Contradicted: the reverse one-sided sign test rejects at the same level, and the median of d_r is at most -delta.
  * Inconclusive: anything else. That includes a difference that the test detects but that stays below delta; it is reported as "detected, below the relevance margin".

### 7.2 H1: interpolation and data efficiency (confirmatory)

Trained on train-A10, scored on test-A10.

* **P1, HK against MR.** H1 says "from some budget on". Fixed sequence over b = 40, 20, 10, 5, 2: each budget is tested only if every larger one was supported.
  * The result is the smallest budget from which HK is supported, b*.
  * H1 is supported against MR if b = 40 is supported.
  * It is contradicted if any budget is contradicted, with Holm's procedure over the five budgets at the level of P1.
* **P2, HK against BN.** H1 says "at small budgets". Fixed sequence over b = 2, 5, 10, 20, 40.
  * The result is the largest budget up to which HK is supported.
  * H1 is supported against BN if b = 2 is supported, and contradicted as for P1.
* **"The gap to BN narrows."** Descriptive. It is the median of d_r against b, with its bootstrap intervals, and it is not tested.

### 7.3 H2: extrapolation (confirmatory)

Trained on train-A5 at b = 10 and 40.

* **The loss from A5 to A10.** For each model m, replicate r and budget b: L = E(test-A10) - E(test-A5).
  * L is in the units of E. A model loses less when its L is smaller.
  * A model without a score on either test set has no L, and is a failure in the pair.
  * The difference is preferred to the ratio J(test-A10) / J(test-A5): near the floor of the noise, a ratio amplifies differences that do not matter.
  * test-A10 and test-A5 are different draws of P3, so L contains the difference between two finite test sets as well as the change of region. Its cluster bootstrap resamples the runs of each test set independently.
* **P3, HK against BN on L,** with d_r = L_BN - L_HK. Holm's procedure over b = 40 and 10. H2 is supported if at least one budget is supported, and contradicted if one is contradicted.
* **P4, HK against MR on L,** the same way. This makes "HK against MR on the extrapolation test" of section 9.7 of the plan concrete. H2 itself only expects MR to lose accuracy.
* **Reported beside them:**
  * E on test-A10 of the models trained at A5;
  * the same E of the models trained at A10 at the same budget.

  A model can lose little because it was poor already on test-A5, and these figures show it.

### 7.4 Multiple comparisons and the dependence between budgets

* The confirmatory claims are the four families P1 to P4. The overall one-sided level of 0.05 is divided equally among them, by Bonferroni, so each family runs at 0.0125.
* Within P1 and P2, the fixed sequence keeps the level of the family. Within P3 and P4, Holm's procedure does.
* All three procedures hold under any dependence between the tests. The tests at different budgets of a replicate are dependent: the data are nested and the initial weights shared (section 5).
* With ten pairs and no ties, a one-sided level of 0.0125 is reached with nine wins or ten. Nine wins give p = 0.0107 and eight give p = 0.055. With ties the number of pairs falls, and the required count follows from the binomial distribution, recorded with each result.
* That count is a consequence of the level, the number of pairs and the procedure. It is not a rule of its own: another division of the level would change it.

### 7.5 Secondary and exploratory analyses

Secondary analyses are named here and reported with every result, but carry no confirmatory claim and no adjustment for multiplicity. Their p-values, where computed, are labelled unadjusted.

* **H3, protocol shift.** Do the candidates rank on the fixed aggregate of test-steps as they rank on test-A10? Measured by Kendall's tau between the two orders of median E, for MN, MR, BL, BN, HK, HU and HKU, at each budget. The ranking in each of the eight conditions is reported beside it. test-steps is not a sample, so no test is made (plan, section 9.7).
* **H4, physical validity.** For BL and BN:
  * the share of windows with a physically invalid prediction, a negative implied rate, or an implied heat flow incompatible with any non-negative conductance (plan, sections 9.3 and 9.4);
  * per budget and test set, with the spread over replicates.

  "More often at small budgets" is read by the sign test of the paired shares between b = 2 and b = 40. "More often outside the training region" is read between test-A10 and test-A5 for the models trained at A5. A window whose implied terms cannot be computed counts as "not computable", in the denominator, and is reported beside the violations; it is never dropped.
* **H5, mechanism (oracle).** The RMS of ln(r_model / r_true) over the states visited by test-A10, for HK and for MR, paired by replicate. It is computed by the oracle script on the truth side, and its outputs go under `experiments/`.
* **Ablations.** HU and HKU against HK, by the rules of section 7.1, without a confirmatory claim.
* **E/R held.** MR and HK with E/R held against the same models with E/R estimated, at b = 5 and 40 on test-A10, paired by replicate. Why b = 5 and 40:
  * b = 5 is the smallest budget with more than one window to fit, and b = 40 the largest;
  * the standard error of E/R falls with the number of windows: 75 K for one excursion and 24 K for ten, in the exploratory check of section 7.5 of the plan.
* **Exploratory.** Everything else, labelled so:
  * phases, corners and the hottest windows;
  * the error conditional on success;
  * the distance of test points from the fitting data;
  * the selection effort;
  * BL against BN.

### 7.6 Uncertainty reported

The summaries of section 9.7 of the plan, each saying which uncertainty it estimates:

* the median, quartiles and extremes of E and J for each model, budget and test set;
* the paired outcomes, with their counts of failures and ties;
* a percentile bootstrap over replicates of the mean paired difference, 10 000 resamples, computed only when both models have a score in every replicate;
* a cluster bootstrap over the runs of test-A10 or test-A5 for the fixed models, coarse with eight and four runs, and presented as such;
* for test-steps, the result for each condition and the fixed aggregate, with their spread over replicates.

## 8. Failures and values that cannot be computed

**Fixed** by the plan:

* training failures (section 8.7);
* failed windows and physical violations (section 9.3);
* how a failure enters a paired outcome (section 7.1 here).

**Proposed:** a window whose implied terms cannot be computed counts as "not computable", in the denominator, and never as a pass or a violation. This was carried from D-033. Every table lists, for each model and budget:

* the replicates with a score;
* the replicates with a training failure;
* the replicates with integration failures;
* the number of failed windows.

## 9. Provenance and reproducibility

* Every summary records:
  * `learning_environment()` and the git state;
  * the command and the definitions with their fingerprints;
  * the `TrainingSettings` as run, `substeps` included;
  * the seeds of each training.
* Runs are made from clean commits, and outputs go under PT_DATA_DIR, outside git.
* A defect found during the development runs is fixed, recorded, and every fit it could have affected is run again (plan, section 5.5).

## 10. Cost, and the checks before the full run

**The estimate.** About 50 hours of fits with four steps per row, about 3.6 hours of wall time in 14 processes on the machine of the pilot, for 1120 trainings and 210 fits of MR, MR_F and BL (experiment log, phase `cost`). The analysis with E/R held adds 80 trainings of HK, 20 fits of MR and the 20 fits of MR_F that start those trainings, about three hours of fits more by the same medians.

**Its assumptions,** none of them measured on benchmark data:

* the machine of the pilot, under the same load;
* b = 10 costing what the pilot's nine windows cost;
* b = 20 and 40 costing what they cost on synthetic runs whose fastest windows are faster than the target's;
* A5 costing what A10 costs, which nothing has measured;
* nothing for generation, evaluation, the oracle or OF.

**Proposed checks before the full run,** in I6 and after the registration:

1. **Ingestion.** Before the development sets are generated, a definition outside the benchmark is run through the generator into a scratch PT_DATA_DIR and discarded. It has two runs of 40 excursions, with the seeds of the tests of the data path. Its time of ingestion, extrapolated with the square of the number of runs (a known limitation of M0), must stay under one hour for each set. If it does not, the schedule is decided before generating, and the science is not changed.
2. **Disk space.** There must be room under PT_DATA_DIR for the sets, their databases and exports, and the summaries.
3. **Training cost.** The first replicate of train-A10 is trained at every budget before the others, and its times are compared with the estimate. If the projected total exceeds twice the estimate, the run stops and the schedule is decided and recorded. More processes or more time may change; the settings may not.
4. **Software.** The script of the benchmark and its tests are committed before any benchmark data are generated, and pass on synthetic records. The definitions pass `tests/test_m1_e03_definitions.py`.

No infrastructure beyond the script of the benchmark is proposed for this.

## 11. What the registration cannot conclude

The limitations of section 11 of the plan apply:

* one plant, one reaction and two hidden mechanisms;
* ideal noise and exact inputs;
* one family of excitation;
* extrapolation in amplitude only;
* small grids;
* ten replicates;
* hypotheses written after an exploratory check.

A result of M1-E03 is about these models, optimisers, grids, data and procedures.

## 12. Decisions pending approval

Each item has a recommendation. Each recommendation is pending until approved.

1. The grid: four configurations per family, as in section 4. Recommended; the pilot gives no reason for more.
2. The optimiser: rates BN 1e-3, HK 1e-2, HU 1e-3 and HKU 1e-2; 3000 steps; checkpoints every 100. Recommended, from the pilot's declared rule.
3. Conditions and replicates: A10 at b = 2, 5, 10, 20 and 40; A5 at b = 10 and 40; ten replicates. Recommended; the pilot's cost leaves room.
4. Four steps per row, passed explicitly, and double precision. Recommended (D-036, clarification of 2026-10-01).
5. The failure policy of D-036, with no retries. Recommended. The alternative, keeping the best checkpoint before a failure, would select from a training that then diverged.
6. E/R held at 8750 K for MR and HK at A10, b = 5 and 40. Recommended.
7. Windows whose implied terms cannot be computed: counted as "not computable", never dropped. Recommended.
8. Seeds:
   * the blocks of section 2 and the bases of section 5;
   * the same initial weights at every budget of a replicate;
   * the configuration index within each family, which pairs the hybrid ablations.

   Recommended. The alternative is a seed per family.
9. The primary families P1 to P4, a one-sided level of 0.05 divided equally among them, exact sign tests, fixed sequences for H1 and Holm for H2. Recommended. The alternative is a level of 0.05 for each family, which would make the four claims weaker together.
10. The relevance margin delta = 0.05 on E, and the median of d_r as the effect. Recommended. The alternatives are no margin, or a margin relative to the comparator's E.
11. The loss of extrapolation as the difference L = E(test-A10) - E(test-A5). Recommended over the ratio.
12. The classification of H3, H4, H5, the ablations and E/R held as secondary, and of the rest as exploratory. Recommended.
13. The checks of section 10, with their thresholds: one hour of ingestion per set, and twice the estimated cost. Recommended.
14. The data set definitions of section 2, with their seeds. Recommended.

## 13. What closes I5 and authorises I6

I5 is closed when the following hold, and not before:

1. The decisions of section 12 are approved, or changed and recorded, before any benchmark data exist.
2. The registration of M1-E03 is written into `docs/experiment_log.md` from a clean commit, with the final definitions in `configs/datasets/`. Their checks pass.
3. Codex has reviewed the registration, and its findings are resolved or recorded.
4. The continuous integration of the commit that records the closure passes.

I6 then begins with the checks of section 10. Its first step is the script of the benchmark, committed and tested before any data are generated. The test sets wait for the technical freeze.
