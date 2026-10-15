# M1-E03, the benchmark of M1: draft registration

**Status: draft for review, not registered.** Written on 2026-10-15 for I5. Every choice below is labelled:

* **fixed**: settled by the plan or by a recorded decision, cited with its entry;
* **proposed**: pending approval, with the reason for it.

Nothing proposed here is approved by being written down. When the decisions of section 12 have been approved, the registration is copied into `docs/experiment_log.md` as M1-E03 and committed from a clean tree. Its data set definitions move from `configs/datasets/m1_e03_draft/` to `configs/datasets/`. The registration is then offered to Codex for review, and only after that does I6 begin (`docs/m1_plan.md`, section 13). No benchmark data exist, no benchmark model has been trained, and nothing below has been run.

It builds on `docs/m1_i5_proposal.md`, whose points it carries forward, and on the plan, which it does not repeat.

**Revision of 2026-10-15, after Codex's review of `35a7cf5`.** Kept apart:

* **What Codex found, as the owner relayed it.** Codex confirmed:
  * the corrections of M1-E02 and the verdicts and hashes of its two summaries;
  * 120 focused tests, ruff and `git diff --check`;
  * the continuous integration of `35a7cf5` on Python 3.12, 3.13 and Docker.

  It found in this draft:
  * verdicts that could support and contradict a hypothesis at once;
  * a reverse test that spent the same level as the forward one;
  * infinite differences given to failures inside a median of errors;
  * a wrong reading of the margin as a change of RMS;
  * bootstrap intervals described differently in two places;
  * no rule for ranking with failures and ties (H3), and no common domain or rule for undefined logarithms (H5).
* **What Claude Code changed.**
  * One verdict for each budget and one for each claim (sections 7.2, 7.4 and 7.5).
  * The level of each family divided between the two directions (section 7.6).
  * Order and magnitude separated, with a rule for relevance when some pairs have no scores (sections 7.1 and 7.2).
  * The margin restated as a difference of mean squared error (sections 6 and 7.3).
  * One table of intervals (section 7.7).
  * Rules for H3 and H5 (section 7.8).

  The rules of sections 7.1 to 7.6 and the ranks of H3 are written as small functions in `process_transfer/evaluation/comparisons.py`. They are checked on synthetic scores in `tests/test_evaluation_comparisons.py`: wins at one budget and losses at another, losses at small budgets and wins at large ones, inconclusive results, ties, failures of one or both models, no pair with scores, a difference below the margin, and the exact limits of the level and of the margin. Nothing in these checks comes from benchmark data.
* **What stays pending.** Every recommendation of section 12. The draft is still a draft, its definitions stay in `configs/datasets/m1_e03_draft/`, and I5 is not closed.

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
| Bootstrap | proposed | 20261121, the base of every resampling of section 7.7, by the rule given there |
| MR, MR_F, BL | fixed | none: deterministic from their declared starts |

What these rules imply, stated so that they are chosen knowingly:

* **The same initial weights at every budget of a replicate.** `training_seed` does not depend on the budget (D-036), and this is kept. The budgets of a replicate are prefixes of one run, so their data are nested and their results were never independent. With the same initial weights, the curve of a replicate over the budgets changes with the data alone. No test treats the budgets of a replicate as independent (section 7.6).
* **The same initial weights for the analysis with E/R held.** It shares the base of train-A10, so HK with E/R held starts from the same networks as HK with E/R estimated, and the paired difference isolates the treatment of E/R.
* **The same network weights across hybrid families.** HK's kinetic network, HU's thermal network and the first network of HKU have the same shape. With the same configuration index they start from the same weights; their last layer is zero, so each factor starts at one. This is kept: it pairs the ablations. A seed per family would remove the pairing, and the choice is listed in section 12.
* **No reuse of the pilot.** The bases differ from the pilot's, 20260929, so no benchmark training reuses a pilot initialisation.

## 6. Evaluation

**Fixed** by the plan, sections 9.1 to 9.3:

* the score J of every model, replicate, budget and test set, pooled over every scored reading of that test set, by the reference rollout from each window's context mean;
* the phases;
* failures and physical violations: a training failure, or a test window whose rollout fails, leaves that model without a score on that test set for that replicate and budget. The replicate is not dropped.

**Proposed** primary quantity: E = J^2 - 1, the estimate of the excess mean squared error against the true states (section 9.1).

* It is preferred to J for comparisons because it is additive over readings and, on test runs, an unbiased estimate of that excess.
* It is reported as it comes, negative values included, and never clipped.

What a difference of E is, and what it is not:

* **A difference of mean squared error, not of RMS.** Between E = 0.05 and E = 0, J goes from sqrt(1.05) to 1, a change of about 0.0247. Between E = 1.00 and E = 0.95, J changes by about 0.0178. A difference of E has no fixed equivalent in RMS: the change of J depends on the level.
* **An average over the two channels.** E = (excess_CA / sigma_CA^2 + excess_T / sigma_T^2) / 2. A difference of E cannot be attributed to a channel. A difference of 0.05 could be an excess of 0.1 sigma_T^2 (0.025 K^2) in T and none in C_A, the converse, or any split.

## 7. Hypotheses and the rules for reading them

Everything in this section is **proposed**, before any benchmark data exist. The rules are those of `process_transfer/evaluation/comparisons.py`, which takes the levels, the margin and the minimum count as arguments, since they are not approved.

### 7.1 One replicate: order and magnitude kept apart

For HK against a comparator C on one replicate, at one budget and on one test set:

| Case | Ordinal outcome | Numeric difference d = E_C - E_HK |
|---|---|---|
| both completed, E_HK < E_C | HK better | d > 0 |
| both completed, E_HK > E_C | C better | d < 0 |
| both completed, equal scores | tie | d = 0 |
| only HK completed | HK better, by reliability | none |
| only C completed | C better, by reliability | none |
| both failed | tie | none |

The ordinal outcome is that of `evaluation.outcomes.paired_outcome`: a failure loses to a score and ties with a failure (plan, section 9.3). A failure is never given a score, and never an infinite difference. Every replicate appears in every table, under one of these six cases.

### 7.2 One budget: the test, the magnitude and its single verdict

* **Test.** The exact one-sided sign test of the pairs that are not ties, in the direction being tested.
* **Magnitude.** The median of d over the replicates where both models completed, with their number, n_both. It is conditional on both completing and is labelled so.
* **Relevance when some pairs have no score.**
  * n_both = 0: the magnitude is not computable.
  * 0 < n_both < 5: the magnitude is not established. Most replicates then have no numeric difference, and a median over the few that do would describe only the trainings where both models happened to complete.
  * n_both >= 5: the confirmed direction is relevant if its median, oriented that way, is at least delta, and below the relevance margin otherwise.
* **The verdict at a budget** is exactly one of:
  * not tested: the procedure did not reach this budget;
  * no decision;
  * HK better or C better, with a magnitude that is relevant, below the relevance margin, not established, or not computable.

  A sign test cannot confirm both directions at one budget; the code checks it.
* **What counts for a claim.** A confirmed direction counts for its claim when its magnitude is relevant, not established or not computable. An advantage that rests on failures of the other model is practical: that model has no prediction there. A direction confirmed but below the margin counts for neither, and is reported as a difference detected but too small to matter.
* **Always reported, whatever the verdict:**
  * the counts of the six cases;
  * n_both;
  * the median of d with its interval (section 7.7).

  A median over the replicates where both completed is never presented without the failures beside it.

### 7.3 The relevance margin

delta = 0.05 on E, proposed as a choice of scale. It is not a value validated by the pilot or derived from the benchmark. What supports it, and its limits:

* **Scale.** From the noise of the readings alone, E on test-A10 has a standard error of about sqrt(2 / 17 600) = 0.011, over 80 windows, 110 scored readings and two channels; on test-A5 it is about 0.015. delta is three to five of those.
* **What that standard error leaves out.** It is only the part of the uncertainty due to the noise of the test readings. It leaves out:
  * the variation between training replicates;
  * the finiteness of the set of test runs;
  * the correlation of a model's errors within a window.

  Section 7.7 reports these, and they are larger.
* **Its meaning.** It is a difference of mean squared error averaged over the two channels (section 6), not a fixed change of RMS and not a change in either channel separately.

### 7.4 H1, confirmatory: an ordered claim

Trained on train-A10, scored on test-A10.

* **P1, HK against MR.** H1 says "HK scores lower than MR from some budget on".
* **P2, HK against BN.** H1 says "HK scores lower than BN at small budgets".
* **Procedure.** In each family there are two fixed sequences, one for each direction, each at alpha_f / 2, one-sided.
  * For P1, both run over b = 40, 20, 10, 5, 2.
  * For P2, both run over b = 2, 5, 10, 20, 40.
  * A sequence stops at the first budget it does not confirm, and the budgets after it are not tested by that sequence.
* **The gate** is the first budget of the order: 40 for P1, 2 for P2.
* **Global verdict.**
  * Supported: the gate counts for HK. The claim is the contiguous run of budgets from the gate that count for HK: "HK better than MR from b* on", or "HK better than BN up to b_max".
  * Contradicted: the gate counts for the comparator. MR better at b = 40 is evidence against "HK better from some budget on", within the budgets tried. BN better at b = 2 is evidence against "HK better at small budgets".
  * Difference below the relevance margin: the gate is confirmed in one direction, below the margin.
  * No confirmatory decision: the gate is confirmed in neither direction. That is an absence of evidence, not evidence against.
* **A loss at a small budget does not refute P1.** The hypothesis allows HK to be worse below b*, and the sequence in MR's direction starts at b = 40. So MR better at b = 2 alone is never a confirmatory claim. It is reported in the table of outcomes, with its counts.
* **Mixed results** are shown as one row per budget with its verdict, under the global verdict.
* "The gap to BN narrows" is descriptive: the median of d against b, with its intervals, not tested.

### 7.5 H2, confirmatory: an unordered claim

Trained on train-A5 at b = 10 and 40.

* **The loss from A5 to A10.** For each model m, replicate r and budget b: L = E(test-A10) - E(test-A5).
  * L is in the units of E; a model loses less when its L is smaller.
  * A model without a score on either test set has no L. In the pair it is a failure, by the rules of section 7.1.
  * The difference is preferred to the ratio J(test-A10) / J(test-A5): near the floor of the noise, a ratio amplifies differences that do not matter.
  * test-A10 and test-A5 are different draws of P3, so L also contains the difference between two finite test sets.
* **P3, HK against BN on L,** with d = L_BN - L_HK. This is H2: HK loses less than BN.
* **P4, HK against MR on L.** This makes "HK against MR on the extrapolation test" of section 9.7 of the plan concrete. H2 itself only expects MR to lose accuracy.
* **Procedure.** Each budget is tested in both directions, by Bonferroni at alpha_f / 4 for each budget and direction.
* **Global verdict.**
  * Supported: some budget counts for HK and none for the comparator.
  * Contradicted: the converse.
  * Budget-dependent: budgets count both ways. In Codex's example, HK better in all ten replicates at b = 10 and worse in all ten at b = 40, H2 is budget-dependent, neither supported nor contradicted. Before this revision the draft could call it both.
  * Difference below the relevance margin: directions are confirmed only below the margin.
  * No confirmatory decision: otherwise.
* **Reported beside them:**
  * E on test-A10 of the models trained at A5;
  * the same E of the models trained at A10 at the same budget.

  A model can lose little because it was poor already on test-A5, and these figures show it.

### 7.6 Multiple comparisons: families, directions and the guarantee

* **Families.** P1 to P4. Each family's claims are directional claims at its budgets: "HK better at b" or "C better at b".
* **Error budget.**
  * The overall one-sided level is 0.05, divided equally among the four families by Bonferroni: alpha_f = 0.0125.
  * Within a family, alpha_f is divided equally between the two directions, so no level is spent twice. Testing the reverse direction at the same level as the forward one, as the earlier draft did, would have doubled the family's level without saying so.
  * Within a direction, the fixed sequence (P1, P2) or Bonferroni over the two budgets (P3, P4) keeps the level.
* **Guarantee.** The probability of at least one false confirmatory claim, in either direction and in any of the four families, is at most 0.05. A false claim is a direction claimed at a budget where, by the null of the sign test, there is none. This holds under any dependence between budgets and between families. The fixed sequence and Bonferroni need no independence, and the directions and families are joined by the union bound. The tests at the budgets of a replicate are dependent: the data are nested and the initial weights shared (section 5).
* **What this means with ten replicates:**

| Family | Level of each test, one-sided | Ten pairs, no ties | With ties |
|---|---|---|---|
| P1, P2 | 0.00625 | 10 of 10 decides (p = 0.00098); 9 of 10 does not (p = 0.0107) | 9 of 9 and 8 of 8 decide; 7 of 7 (p = 0.0078) does not |
| P3, P4 | 0.003125 | 10 of 10 decides | 9 of 9 decides (p = 0.00195); 8 of 8 (p = 0.0039) does not |

  A confirmatory claim therefore needs every replicate that is not a tie to go the same way, and one replicate the other way leaves no decision. If the true chance that HK wins a replicate were 0.90, ten of ten would happen with probability 0.35; at 0.95, with 0.60; at 0.99, with 0.90. These follow from the level, the ten replicates and the procedure. They are not rules of their own.
* **Alternative.** A level of 0.05 for each family:
  * nine of ten would decide in every family;
  * the guarantee would hold for each family, not for the four together, which could then hold a false claim with probability up to 0.2.
* **Symmetry.** The rules do not favour HK: what HK needs to be supported, the comparator needs to contradict.

### 7.7 Uncertainty: what each interval estimates

| Interval | Estimand | Resampled | Held fixed | When not computable | Resamples |
|---|---|---|---|---|---|
| bootstrap over replicates, median | the median of d over the replicates where both completed | those replicates, with replacement | the test set and each replicate's models | n_both < 5: no interval | 10 000, percentile 95 % |
| bootstrap over replicates, mean (plan, 9.7) | the mean of d | all ten replicates | the same | computed only when both completed in every replicate | 10 000, percentile 95 % |
| cluster bootstrap of a test set | E of a fixed model, and d of a fixed pair | whole runs of test-A10 (8) or test-A5 (4), the same runs for both models of a pair | the trained models of one replicate and budget | a model without a score on that test set | 10 000; with four runs there are only 35 distinct resamples, and the interval is presented as coarse |
| cluster bootstrap of L | L of a fixed model | the runs of test-A10 and of test-A5, independently | the same | a model without a score on either test set | 10 000 |

* The seed of each interval is `numpy.random.SeedSequence([20261121, family, condition, budget, kind])`, with the kind numbered in the order of the table.
* Training replicates and test runs are the only units resampled, and each answers a different question: how much the result depends on the training realisation, and how much on the finite test set.
* Windows and readings are never resampled alone. Budgets are never pooled or treated as independent: each interval is for one budget.
* The intervals describe; the verdicts come from the tests of sections 7.4 to 7.6.
* test-steps has no bootstrap, since its conditions are fixed and not sampled; it is reported with its spread over replicates (plan, section 9.7).
* This table replaces the two earlier descriptions, one of an interval for the median and one for the mean.

### 7.8 Secondary and exploratory analyses

Secondary analyses are named here and reported with every result. They carry no confirmatory claim and no adjustment for multiplicity, and their p-values, where computed, are labelled unadjusted.

* **H3, protocol shift.**
  * **Ranks.** In each replicate, the seven candidates (MN, MR, BL, BN, HK, HU and HKU) are ranked by E on test-A10, and by E on the fixed aggregate of test-steps. A model without a score on a test set ranks below every model with one, failures tie among themselves, and ties take the mean of the ranks they span. A model is ranked over every replicate, failures included, never over its successes alone, so failing on hard cases is not rewarded. Each candidate's rank is its mean over the ten replicates (`comparisons.mean_ranks`).
  * **Agreement.** Kendall's tau-b between the two orders, at each budget (`comparisons.rank_agreement`). Tau-b allows ties. If either order has every candidate tied, tau-b is recorded as not computable, with that reason.
  * The orders in each of the eight conditions are reported beside it. test-steps is not a sample, so no test is made (plan, section 9.7).
* **H4, physical validity.** For BL and BN:
  * the share of windows with a physically invalid prediction, a negative implied rate, or an implied heat flow incompatible with any non-negative conductance (plan, sections 9.3 and 9.4);
  * per budget and test set, with the spread over replicates.

  "More often at small budgets" is read by the sign test of the paired shares between b = 2 and b = 40. "More often outside the training region" is read between test-A10 and test-A5 for the models trained at A5. A window whose implied terms cannot be computed counts as "not computable", in the denominator, and is reported beside the violations; it is never dropped.
* **H5, mechanism (oracle, secondary).**
  * **The common set of states.** The true states of the target along the runs of test-A10, at the instants of the scored readings of their 80 windows: 8800 states, taken from the noise-free truth on the truth side. Every model, replicate and budget is judged on this same set, each state with the same weight.
  * **The rates.** At each state, the true rate, MR's fitted first-order law, and HK's law: its own fitted mechanistic parameters times its learned factor. Each is evaluated at the true C_A and T of the state.
  * **The metric.** The RMS over the states of ln(r_model / r_true).
  * **Where the logarithm does not exist.** A state where the true rate, MR's or HK's is zero, negative or not finite is not computable for that comparison. The comparison of HK with MR at a replicate uses the states computable for all three. It reports how many were left out, and is flagged if any were. No epsilon makes a logarithm computable. The laws of MR and HK are positive by construction where C_A > 0, and the true states of P3 keep C_A well above zero (M1-E02 found at least 66.96 mol/m^3 at A10), so a left-out state would point to an overflow, which is reported as such.
  * **Failures.** A model with a training failure has no law, and the pair is a failure for it by the rules of section 7.1.
  * **Reported.** The paired difference RMS_MR - RMS_HK by replicate, with sign counts, unadjusted.
  * **Separation from the models.** H5 is computed by a separate script on the truth side, its outputs go under `experiments/`, and nothing from it reaches a selection or a training.
* **Ablations.** HU and HKU against HK, by the rules of sections 7.1 and 7.2, without a confirmatory claim.
* **E/R held.** MR and HK with E/R held against the same models with E/R estimated, at b = 5 and 40 on test-A10, paired by replicate. Why b = 5 and 40:
  * b = 5 is the smallest budget with more than one window to fit, and b = 40 the largest;
  * the standard error of E/R falls with the number of windows: 75 K for one excursion and 24 K for ten, in the exploratory check of section 7.5 of the plan.
* **Exploratory.** Everything else, labelled so:
  * phases, corners and the hottest windows;
  * the error conditional on success;
  * the distance of test points from the fitting data;
  * the selection effort;
  * BL against BN.

## 8. Failures and values that cannot be computed

**Fixed** by the plan: training failures (section 8.7), and failed windows and physical violations (section 9.3).

**Proposed.** Three summaries are kept apart and reported together:

* **Order.** The paired outcomes over every replicate, failures included (section 7.1).
* **Reliability.** For each model and budget:
  * the replicates with a score;
  * the replicates with a training failure;
  * the replicates with integration failures;
  * the number of failed windows;
  * in a pair, how many replicates were won or lost by a failure.
* **Precision.** The difference of E over the replicates where both completed, with their number, labelled as conditional on success.

No conclusion rests on precision alone while failures are left out of view. A window whose implied terms cannot be computed counts as "not computable", in the denominator, never as a pass or a violation (carried from D-033).

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

Each row is a recommendation, pending until approved.

| # | Decision | Recommended | Why | Practical consequence | Alternative |
|---|---|---|---|---|---|
| 1 | Grid | four configurations per family (section 4) | the pilot: size and penalty barely separated configurations | 16 trainings per replicate and budget | a larger grid, with no evidence for it |
| 2 | Optimiser | rates BN 1e-3, HK 1e-2, HU 1e-3, HKU 1e-2; 3000 steps; checkpoints every 100 | the pilot's declared rule | 31 checkpoints on V per training | rates chosen again on benchmark data, which would spend V |
| 3 | Conditions and replicates | A10 at b = 2, 5, 10, 20, 40; A5 at b = 10, 40; ten replicates | the plan, sized by the pilot's cost | about 50 h of fits (section 10) | more replicates, which would relax section 7.6 at a cost |
| 4 | Precision of training | four steps per row, passed explicitly; double precision | the scheme within 3e-3 sigma of its equation where two steps departed by 0.05 | the steps take twice as long | two steps, the default of the code |
| 5 | Failures | the policy of D-036, no retries | no checkpoint selected from a training that then diverged | a failure is an outcome of its method | keeping the best checkpoint before a failure |
| 6 | E/R held | MR and HK at A10, b = 5 and 40 | the smallest budget with more than one window to fit, and the largest | 80 more trainings, about 3 h of fits | other budgets |
| 7 | Seeds | the blocks of section 2, the bases of section 5, the same initial weights at every budget, the index within each family | new seeds; a curve that changes with the data alone; paired ablations | HK, HU and HKU share their first network's initial weights | a seed per family, unpairing the ablations |
| 8 | Order and magnitude | the six cases of section 7.1; no score and no infinite difference for a failure | a failure has no error to compare | every replicate in every count | treating a failure as the worst score, which invents an error |
| 9 | Relevance with failures | the median of d over n_both >= 5; below that, not established | a median over a minority of replicates does not describe the trainings | a win by failures counts for its claim, labelled as reliability | a minimum other than five |
| 10 | Margin | delta = 0.05 on E | three to five standard errors of the noise alone (section 7.3) | small differences are reported as detected, not as relevant | no margin, or a margin relative to the comparator's E |
| 11 | Families and levels | P1 to P4, 0.05 overall, 0.0125 each, half to each direction; fixed sequences for H1, Bonferroni for H2 | one guarantee for every confirmatory claim, in both directions, under any dependence | with ten replicates a claim needs every replicate that is not a tie to agree | 0.05 for each family: nine of ten decides, the guarantee holds for each family only |
| 12 | Global verdicts | gates for H1; "budget-dependent" for H2 (sections 7.4 and 7.5) | a hypothesis can never be supported and contradicted at once | mixed results are shown budget by budget, under one verdict | testing every budget alike, which would let a small budget refute "from some budget on" |
| 13 | Extrapolation | L = E(test-A10) - E(test-A5) | additive, and stable near the floor of the noise | an L that also contains the difference between two test sets | the ratio of J |
| 14 | Intervals | the four of section 7.7 | each states its estimand, unit and what is held fixed | coarse intervals for test-A5 | a single bootstrap over windows, which would treat dependent readings as independent |
| 15 | H3 and H5 | ranks with failures last and tau-b; one common set of true states with no epsilon (section 7.8) | no model is rewarded for failing, and every law is judged on the same states | some comparisons may be reported as not computable | ranking successes only, or each law on its own states |
| 16 | Classification | H3, H4, H5, the ablations and E/R held secondary; the rest exploratory | the plan, section 10 | no confirmatory claim outside P1 to P4 | |
| 17 | Checks before the full run | section 10: one hour of ingestion per set, twice the estimated cost | a schedule decided before data, never a change of science | a stop and a recorded decision if exceeded | other thresholds |
| 18 | Data set definitions | section 2, with their seeds | new seeds and streams, checked | | |

## 13. What closes I5 and authorises I6

In this order, and not before:

1. **Approval.** The decisions of section 12 are approved, or changed and recorded, before any benchmark data exist.
2. **The definitive registration.** The registration of M1-E03 is written into `docs/experiment_log.md` from a clean commit, without the label of draft. The definitions move from `configs/datasets/m1_e03_draft/` to `configs/datasets/`. When they move, their relative paths to the plant and the sensors change from `../../` to `../`. `tests/test_m1_e03_definitions.py` then points at the new place and is run again, to show that they still load and keep their identities, seeds and streams.
3. **Review.** Codex reviews the registration, and its findings are resolved or recorded.
4. **Closure.** The continuous integration of the commit that records the closure of I5 passes on Python 3.12, 3.13 and Docker.

I6 then begins with the checks of section 10. Its first step is the script of the benchmark, committed and tested before any data are generated. The test sets wait for the technical freeze.
