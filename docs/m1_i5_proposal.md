# I5 of M1: proposal for the registration of the benchmark

Written on 2026-10-08, after I4, for the owner and for review. These are recommendations, pending review: none of them is approved by being written here, and nothing of I5 has been registered, defined or run. I5 is the registration of the benchmark, M1-E03, with the definitions of its data sets. It is committed from a clean tree before any benchmark data exist and offered for review before I6 (`docs/m1_plan.md`, section 13). I6 then generates the development sets and trains; I7 generates the test sets after the technical freeze.

What is already fixed, and stays where it is:

* the question and comparators (D-029);
* the answers to Q1 to Q5 (D-030);
* the data sets and their sizes (plan, section 5.2) and the budgets (section 5.4);
* the division into F and V (section 5.5), the models (section 8) and the evaluation (section 9);
* the framework and the training (D-035 to D-037);
* the lead and A5, with their verification (D-039, D-040, M1-E02).

## Decisions that the registration has to make, with a recommendation for each

1. **Configurations.** The owner's grid as the pilot ran it: BN with two layers of 16 or 32, and HK, HU and HKU with one layer of 8 or 16, each with a penalty of 1e-4 or 1e-2, so four configurations per family. In the pilot, size and penalty barely separated the configurations: median differences of the criterion of at most 0.005. Nothing there argues for another grid.
2. **The optimiser.** The rates the pilot chose by its declared rule: BN 1e-3, HK 1e-2, HU 1e-3 and HKU 1e-2. 3000 steps, with a checkpoint on V every 100. One caveat: these were chosen on the development data of M0 at budgets up to nine windows, at A10 only.
3. **Conditions and replicates.** As the owner sized them: A10 at b = 2, 5, 10, 20 and 40, A5 at b = 10 and 40, and ten replicates of each. The pilot's cost supports ten.
4. **Precision of training.**
   * Four steps of the fixed-step scheme per row, as D-036 proposed. In the pilot they kept the scheme within 3e-3 sigma of its equation on the fastest windows met, where two steps departed by 0.05 sigma.
   * Double precision.
   * Selection and evaluation by the reference rollout, as now.
   * Every run that trains records `learning_environment()` (D-040).
5. **Failures and checkpoints.** The pilot had no failure, so it informs nothing here.
   * Today a training that fails at a step discards its earlier checkpoints. The owner kept that policy until this decision.
   * The alternative is to keep the best checkpoint on V before the failure. That uses checkpoints that were already selectable, at the price of selecting from a training that then diverged.
   * Recommended: keep the present policy. A failure is an outcome of its method, counted and reported per family, budget and replicate with its reason. Nothing is retried with another seed.
6. **The secondary analysis with E/R held at 8750 K**, for MR and HK (D-030). Recommended at b = 5 and b = 40 at A10. b = 5 is the smallest budget with more than one window to fit; M1-E01 found E/R weakly determined there. b = 40 is the largest budget.
7. **A window whose implied terms cannot be computed** (plan, section 9.4; carried from D-033). Recommended:
   * it is counted, in the denominator, as "not computable", and reported per model and budget beside the windows that violate the sign condition;
   * it is never dropped;
   * a window whose rollout failed is already a failure record and is counted as such.
8. **Seeds.**
   * Excitation seeds and one master seed per data set, none used before (the experiment log lists those used).
   * A base for the seeds of training, so that every fit's seed follows from its condition, replicate and configuration.
   * Proposed: one block of excitation seeds per data set, such as 31001 to 31010 for train-A10, 32001 to 32010 for train-A5, 33001 to 33008 for test-A10 and 34001 to 34004 for test-A5. The test steps get a new noise realisation of the steps of D-026.
9. **Thresholds of the hypotheses and the rules for reading them** (plan, section 10). The registration has to fix them. This proposal makes no recommendation beyond the plan, because the thresholds are the scientific core of the registration and belong to the owner.

## Cost, and the controls before any data are generated

* **Cost.** About 50 hours of fits with four steps per row, about 3.6 hours of wall time in 14 processes on this machine. This is the pilot's estimate, with its assumptions (experiment log, phase `cost`). A5 is assumed to cost what A10 costs, which no measurement yet supports.
* **Data.** The development sets hold 577 320 measurements and the test sets 101 640. Ingestion grows with the square of the number of runs, a known limitation of M0, and is measured before the full sets are generated (plan, section 5.2).
* **Before I6 generates anything:**
  * the registration and the definitions are committed from a clean tree and reviewed;
  * every definition loads, its identities and noise streams are distinct, and no seed repeats one already used;
  * the time of ingestion is measured on a small definition outside the benchmark;
  * PT_DATA_DIR is outside the repository and has room.
* **Test sets.** They are defined in the registration, with their seeds, and generated only after the freeze (I7).

## Next action

The owner decides the points above. Then the registration of M1-E03 and the definitions of its data sets are written, committed from a clean tree without generating anything, and offered for review before I6.
