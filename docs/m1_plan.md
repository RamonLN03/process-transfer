# Plan of M1: black-box and hybrid models fitted on the target plant

Proposed on 2026-09-22, the day M0 was closed; revised on 2026-09-23 after Codex's audit of `c2309cc` and `1dd6830`, and on 2026-09-24 when the owner answered the questions of section 15 and two precisions left by Codex's review of revision 1 were taken in. Section 16 lists what each revision changed and why. Nothing described here is implemented. The owner's instruction that started M1 fixes the question, the comparators and what is out of scope (D-029); the owner's answers to Q1 to Q5 fix the choices those questions put (D-030). Section 3 says what is accepted and what is still open.

## 1. The question

Does a continuous-time hybrid model improve data efficiency or extrapolation over simple alternatives, when every model is fitted on data of the target plant only?

Accepted (D-029):

* Comparators: the nominal mechanistic model, as the starting reference; the mechanistic model with its parameters re-estimated on the target; a black-box model trained on the target only; a hybrid model trained on the target only; and an oracle with the complete true physics, a separate diagnostic reference that is never a candidate.
* Not in M1: pretraining on the source, fine-tuning between plants, transfer policies and systematic studies of domain shift (M2, M3 and later).
* The hybrid is not assumed to win. A well-supported negative result is a valid result.
* alpha (0.005 1/K on the source, 0.002 1/K on the target) and the physical configurations of M0 are kept. The difficulty is not adjusted to favour a method.
* Ordinary models never contain alpha, the saturating law, K_sat or the true form of UA(T). A study that uses them is an oracle and is kept apart.

Added by this plan beyond that list, and accepted with the answers of section 15: a linear black box as the simplest data-driven reference (section 8.3) and two variants of the hybrid for comparison (section 8.2).

## 2. What M0 hands over

What exists and can be used as it is:

* The modeller's equations, `modeller/cstr_first_order.py`: first-order Arrhenius kinetics and a constant UA, with the textbook values of `configs/modeller_cstr.yaml` (k0 = 7.2e10 1/min, E/R = 8750 K, UA = 1.0e5 J/(min K)). The package never imports the simulation; a test on the import graph enforces it.
* The generator, `python -m process_transfer.generation <definition>`: starting points verified, true trajectories accepted before anything is observed, the sensors of D-020, immutable Parquet data sets, DuckDB, the SQL quality queries, exports verified bit for bit.
* What a model may read, the export (D-024): one row every 6 s with the readings of C_A and T and the four inputs applied from that instant on (zero-order hold, right-continuous), the noise level of each sensor, and the known parameters of the plant: volume, density, heat capacity, heat of reaction, nominal inputs. No initial state, exact state, hidden parameter or noise seed is in it.
* Three protocols verified on both plants: P3 (D-019), steady operation and single-input steps (D-026).

The facts about the target that shape this design, all registered:

| | Value | Source |
|---|---|---|
| Nominal steady state | 189.67 mol/m^3, 355.17 K | M0-E01 |
| Eigenvalues at the nominal point | -0.964 +- 1.804i 1/min: slowest time constant 62 s, period 209 s | M0-E01 |
| Temperature under P3 | 342.37 to 376.19 K, against the limit of 380 K | M0-E04, M0-E03b |
| C_A under P3 | from 67 mol/m^3 (M0-E04) to 347 mol/m^3 at the end of the cold corner `++--` (M0-E03) | M0-E03, M0-E04 |
| True rate against the textbook first-order rate, 2 / (1 + K_sat C_A) | 1.14 at the nominal state; 1.33 to 0.99 over the central 90 % of P3 samples | D-006, M0-E03 |
| Conductance at the nominal point | 1347 W/K; the modeller's 1667 W/K is 24 % too high for the target | M0-E08 |
| Effect of UA(T) alone on P3 trajectories (oracle) | up to 1.8 K and 8.8 mol/m^3 at the hot peaks; about half of sigma_T in root mean square over a run | M0-E08 |
| Step responses | non-monotone in T under every input, two-signed under q | M0-E07 |

Two consequences. The textbook values were chosen to reproduce the source's nominal point (D-006), and both plants have the same nominal inputs (D-008), so the textbook model puts the target's steady state where the source's is: one steady state, at 250.01 mol/m^3 and 350.00 K, found with the repository's steady-state search on the modeller's equations while writing this plan. The nominal model's steady state for the target is therefore 12 sigma_CA and 10 sigma_T away from the real one. And the target's hidden conductance law is weak: what a constant UA equal to the conductance at the nominal point leaves out moves the trajectories by about half a sigma_T in root mean square over a run, and by up to 3.5 sigma_T at the hot peaks (M0-E08).

What M0 does not hand over, and M1 has to build: a division of data into fitting, validation and test; budgets; windows and initial states; metrics; any model code. The data sets `m0-e05`, `m0-e06` and `m0-e07` were built to verify the data path and the protocols, and none of them is the benchmark of M1 (section 5.1).

Carried over from M0 and not blocking: the test of the exit code of M0-E08 checks the same aggregation that `main()` uses instead of calling `main()` (`docs/roadmap.md`). Nothing in M1 depends on it. It is left for whoever next touches that test, and no iteration below is planned around it. No other defect of M0 was found while preparing this plan or its revision.

## 3. What is accepted and what is still open

* Accepted: the list of section 1 (D-029), and the owner's answers to Q1 to Q5 (D-030, section 15). They settle the treatment of E/R, the extrapolation test, the budgets and how fitting and validation share one, the main hybrid, and the primary evaluation with its scoring.
* The working design of M1: sections 4 to 14, which carry those answers out. A part that the answers do not cover can still change, through a recorded revision (section 16) reviewed like the others.
* Open, and not closed by the answers: the two budgets of the secondary analysis with E/R held fixed, and the number of replicates, both fixed in the registration of the benchmark (section 13, I5), the second from the cost of a fit measured in I3; the training framework (section 8.6, I3); the grids of the learned models; the thresholds of the hypotheses and the rules for interpreting them (I5); P3 at A5, which is used only once its verification has passed (I4); whether M1 is repeated on the source plant as a second case, after the target.

## 4. The predictive task

### 4.1 What is observed and what is predicted

A model predicts the state x = (C_A, T) of the target from two things: an initial state estimated from readings taken before the prediction starts, and the inputs over the horizon, which are known exactly. In a planned test, and in any use of a model for design or control, the future inputs are chosen rather than measured, so giving them to the model is not a leak. No reading taken after the start of a prediction is given to it. Predictions are scored against readings at the sensor instants of the horizon, and never against a reading the prediction was computed from.

### 4.2 The initial state

Both states are measured, with noise. The initial state of a prediction that starts at t0 is the mean of the ten readings in (t0 - 60 s, t0], that is at t0 - 54 s, ..., t0, taken under constant nominal inputs with the plant settled: at the end of a rest of P3, at the end of the lead of a step test, or during the lead of a run (below). These ten readings are the context of the prediction. The reading at t0 is of the state before any change applied at t0 (right-continuous convention), so it belongs to the context. The mean has a standard deviation of sigma / sqrt(10), 1.6 mol/m^3 and 0.16 K. A single reading would start every prediction one sigma off, an error that decays with the plant's time constant of about a minute and therefore weighs most during the two minutes of the excursion. Every model receives the same initial state; none estimates its own.

A run of P3 starts with an excursion at t = 0, so its first excursion has no reading before it. Proposed: every P3 run of M1 starts with a lead of 60 s at the nominal inputs. The lead is steady operation from the verified steady state, where M0-E06 showed the target's true state staying to within 1.3e-8 mol/m^3 and 2.1e-9 K over two hours, and what follows is P3 from that same state, so the verification of P3 applies as it stands. It needs a token in the identity of a run, a technical change of the grammar of the kind D-026 made.

### 4.3 Windows, phases and the index contract

Ticks are counted from the start of a run, tick k at t = 6k s. Intervals are half-open, (a, b], the right end included.

A P3 run of M1 has its lead at ticks 0 to 10 (0 to 60 s). Excursion j has its onset at tick o_j = 10 + 120 (j - 1), that is at t = 60 + 720 (j - 1) s; the corner is applied from o_j to o_j + 20 and the rest from o_j + 20 to o_j + 120, which is the next onset. The row at a switching tick carries the new inputs and the reading of a state that has not yet responded.

Window j, around the onset t_o of excursion j:

| Part | Ticks | Time | Readings | Use |
|---|---|---|---|---|
| context | o_j - 9 to o_j | (t_o - 60 s, t_o] | 10 | their mean is the initial state; never scored in window j |
| excursion | o_j + 1 to o_j + 20 | (t_o, t_o + 120 s] | 20 | scored |
| return | o_j + 21 to o_j + 70 | (t_o + 120 s, t_o + 420 s] | 50 | scored |
| settled | o_j + 71 to o_j + 110 | (t_o + 420 s, t_o + 660 s] | 40 | scored |
| next context | o_j + 111 to o_j + 120 | (t_o + 660 s, t_o + 720 s] | 10 | the context of window j + 1; scored in no window |

A window scores 110 readings over 660 s. In the evaluation by windows, every reading of a run is the context of exactly one window, scored in exactly one window, or unused: tick 0, and at the end of a run or of a budget the ten readings that would be the context of a window that is not there. No reading is both a context and scored. The first version of this plan scored 120 readings per window and left the boundary readings ambiguous; section 16 records the change.

The window is where a prediction is made and scored. It is not an independent experimental unit: the windows of one run share its excitation sequence, its noise stream (independent from sample to sample, D-020) and a carried state of the order of millikelvins after a rest (M0-E03b). The independent units are runs (section 5.3).

A single-input step run (lead, hold and recovery of 600 s each, 301 rows) has one window: onset at tick 100 (600 s), context at ticks 91 to 100, (540 s, 600 s], and 200 scored readings at ticks 101 to 300, (600 s, 1800 s], in four phases of 50: hold transient (600 s, 900 s], hold settled (900 s, 1200 s], recovery transient (1200 s, 1500 s], recovery settled (1500 s, 1800 s].

Windows are found from the known inputs and the known nominal inputs alone. On the P3 runs of M0, which have no lead, the first excursion of a run has no context and forms no window.

### 4.4 One step ahead against free rollout

One step is 6 s. A prediction one step ahead starts from a reading that is one sigma off and is scored against another reading, so the error of any reasonable model is dominated by the two noises: in 6 s a plant with a time constant of a minute moves by about a tenth of its deviation. Differences between models, which are differences in how the state evolves over minutes, barely show. It is a weak test of a model meant to predict the response to a planned change.

Proposed as the primary evaluation: free rollout over each window. The model receives the initial state and the inputs of the window, integrates without seeing any further reading, and is scored on the 110 scored readings of the window.

Secondary, on test runs only: free rollout over a whole run, from the context at ticks 1 to 10 of the lead, scored at every tick from 11 to the end of the run. There is no re-initialisation, so the readings that are contexts of windows are scored here; the prediction depends on ticks 1 to 10 only. It exposes drift, instability and a wrong steady state, which a restart at every window would partly hide.

Diagnostic only: predictions k steps ahead, k = 1 and 10, each from a single reading taken as the initial state and scored against the reading k ticks later, never against itself.

## 5. Data and partitions

### 5.1 Three kinds of data

* Development runs of the benchmark: train-A10 and, if Q2 is accepted, train-A5. Budgets are drawn from them, and each budget is divided into a fitting part and a validation part (section 5.5).
* Test runs of the benchmark: test-A10, test-A5 and test-steps. They are generated after the technical freeze and evaluated once (section 9.8).
* The data sets of M0, `m0-e05`, `m0-e06` and `m0-e07`: software development and smoke runs, and the profile of M1-E01 (section 7.4). Every use of them is exploratory. No number obtained on them is a result of M1, and when one of them informs a later choice, what it showed is recorded in the experiment log as exploratory, as section 7.5 does for the first check. M1 reuses none of their excitation seeds or noise realisations.

Development runs and test runs belong to different data sets, with different excitation seeds and different master seeds, so their noise is independent. No run, window or reading is shared between them.

### 5.2 New data sets, all on the target

| Set | Protocol | Runs | Record | Role |
|---|---|---|---|---|
| train-A10 | P3 with the lead, A10 | 10 runs of 40 excursions | 10 x 481 min | development: fitting and validation |
| train-A5 | P3 with the lead, half amplitudes (A5), if Q2 is accepted | 10 runs of 40 excursions | 10 x 481 min | development, for the extrapolation test |
| test-A10 | P3 with the lead, A10 | 8 runs of 10 excursions, 80 windows | 8 x 121 min | the fixed evaluation set of D-011, for interpolation and extrapolation |
| test-A5 | P3 with the lead, A5 | 4 runs of 10 excursions, 40 windows | 4 x 121 min | the in-region reference of the models trained at A5 |
| test-steps | the eight single-input steps of D-026, a new noise realisation | 8 runs, one per fixed condition | 8 x 30 min | protocol shift |

Excitation seeds and master seeds are new: none that an earlier experiment, test or smoke run used (the experiment log lists them), and one master seed per data set. The test sets are defined in the registration with the others, seeds included, but generated only after the technical freeze.

Sizes: a run of 40 excursions has 4811 rows of six channels. The two development sets together hold 577 320 measurements and the three test sets 101 640, well within what the path of M0 handled; ingestion time grows with the square of the number of runs (a known limitation of M0) and will be measured before the full sets are generated.

### 5.3 Replicates

A replicate is one run of a development set: its own excitation sequence and its own noise. Budgets are prefixes of it, the lead and the first b excursions with their rests. The data of b = 5 are part of the data of b = 10, so the curve of a replicate is monotone in data. The replicate is the independent unit of the variability of training: the ten replicates measure how much a result depends on which corners and which noise happened to come.

P3 draws its corners independently and with replacement. For every budget the number of distinct corners in the fitting part, and the rank of their sign vectors, are recorded; both follow from the known inputs. At b = 2 the fitting part of a method that selects holds one excursion, so its inputs move along a single direction of the input space. Nothing is added to compensate: results at small budgets are read against the excitation they had.

Excitation and noise are separated without handing a private seed to anyone. The excitation seed is part of the identity of a run, as in `target.p3.e0.x10.n0`. That is harmless: it only determines the inputs, and the inputs are recorded exactly anyway. The noise realisation is the `n<k>` of the identity, and its stream follows from the identity under the private master seed of the data set (D-021), which stays under `private/`. The same excitation with independent noise would be the same identity with another `n<k>`; no such pair is planned for the first benchmark.

The randomness of training (initial weights, order of the windows) has seeds of its own, recorded with each fit and unrelated to the seeds of the data.

### 5.4 Budgets

| b, excursions | Record, exact | Label | Time at a corner | Rows | Context and scored readings | Fitting and validation windows of a method that selects |
|---|---|---|---|---|---|---|
| 2 | 1500 s, 25 min | 0.4 h | 4 min | 251 | 20 and 220 | 1 and 1 |
| 5 | 3660 s, 61 min | 1 h | 10 min | 611 | 50 and 550 | 4 and 1 |
| 10 | 7260 s, 121 min | 2 h | 20 min | 1211 | 100 and 1100 | 8 and 2 |
| 20 | 14 460 s, 241 min | 4 h | 40 min | 2411 | 200 and 2200 | 16 and 4 |
| 40 | 28 860 s, 481 min | 8 h | 80 min | 4811 | 400 and 4400 | 32 and 8 |

The exact record is the lead of 60 s plus b cycles of 720 s. The labels are rounded and are only labels; results are reported against b, with the exact record and the minutes at a corner alongside. Eleven readings of each prefix are unused, tick 0 and the ten that would be the context of window b + 1; the budget counts them because they were recorded.

The budgets follow those of D-011, 30 min to 8 h, in the unit that P3 is built of. Hours of record overstate the information: of the twelve minutes of an excursion and its rest, two are at a corner, about five are the return, and the last five are the nominal steady state again, which after the first rest adds little beyond averaging its noise.

### 5.5 Fitting and validation inside a budget

A method that selects anything from data, a checkpoint or a configuration of its grid, divides the windows of its budget in time. The fitting part F is windows 1 to b - n_V, the validation part V the last n_V windows, with n_V = max(1, floor(b / 5)). Each part is made of the contexts and the scored readings of its own windows (section 4.3). The context of the first validation window, the last 60 s of the rest of the last fitting excursion, therefore belongs to V, and the fitting part does not score it. No reading is in both.

V is a temporal hold-out inside one run. It shares the run's plant, instruments and excitation generator and is separated from F by a rest, so it is not an independent trial. It serves selection and nothing else, and its score is never reported as an estimate of how well a model generalises.

V is used for one thing: the validation criterion, the score J of section 9.1 on V's scored readings, from V's contexts. It chooses a checkpoint during training and a configuration from the grid declared in the registration. Checkpoints are evaluated at declared points; the choice is the lowest criterion among the checkpoints whose validation rollouts all complete, the earliest on a tie. The grids are fixed in the registration. Once validation scores on benchmark data have been seen, no configuration is added to a grid and none is changed: that would use V a second time, for a new decision. A grid in which every configuration fails is a training failure of that model, replicate and budget (section 8.7), and is reported as one. A defect of the software found during the development runs is fixed and recorded, and every fit it could have affected is run again, as section 9.8 does for the test sets; that is not a change of a grid.

V is never used for the fitting loss, for normalisation scales, for initial values of parameters, for the MR that starts a hybrid, or for anything else computed before selection.

The fitting loss of every method is the score of section 9.1 on the scored readings of its fitting windows, each window started from the mean of its own context. Contexts serve as initial states only.

No refit after selection. The final model of a method that selects is the selected checkpoint, fitted on F. A refit on F and V together would need a stopping rule that no held-out data could check. The cost is that methods that select fit on b - n_V windows while the methods with nothing to select, MR and BL, fit on all b: that is the price of selection, and the comparison keeps it (question Q3).

Hybrids start from MR_F, the mechanistic model re-estimated on F alone, never from the comparator MR fitted on all b. Their correction starts at the identity, so the first checkpoint of a hybrid is MR_F, and a hybrid whose correction does not improve the criterion on V ends as MR_F.

### 5.6 What each stage may read

| Stage | Reads | Never reads |
|---|---|---|
| fitting a method that selects | contexts, scored readings and inputs of F | V, test runs, truth |
| normalisation scales, initial values, MR_F | F, as above | V, test runs, truth |
| choice of checkpoint and configuration | contexts, scored readings and inputs of V; the candidates fitted on F | test runs, truth |
| fitting MR (the comparator) and BL | contexts, scored readings and inputs of all b windows | test runs, truth |
| evaluation | test runs: contexts as initial states, inputs, scored readings as scores | nothing is chosen from it; truth |
| oracle diagnostics | the truth side, the budgets (OF is fitted like MR) and the test runs; outputs under `experiments/` | nothing flows back into a choice |

Every stage may also read the known plant parameters of the export, the noise levels of the data sheet and the modeller's textbook values, which are starting points. No stage reads rows outside the budget it works on.

### 5.7 A worked example

b = 2: the lead at ticks 0 to 10, excursions with onsets at ticks 10 and 130 (60 s and 780 s), the prefix ending at tick 250 (1500 s).

| Ticks | Time, s | Role at b = 2 |
|---|---|---|
| 0 | 0 | unused |
| 1 to 10 | 6 to 60 | context of window 1, in F; tick 10 is the first onset: its row carries the corner and its reading the state before it |
| 11 to 30 | 66 to 180 | window 1, excursion, scored, in F |
| 31 to 80 | 186 to 480 | window 1, return, scored, in F |
| 81 to 120 | 486 to 720 | window 1, settled, scored, in F |
| 121 to 130 | 726 to 780 | context of window 2, in V; tick 130 is the second onset |
| 131 to 150 | 786 to 900 | window 2, excursion, scored, in V |
| 151 to 200 | 906 to 1200 | window 2, return, scored, in V |
| 201 to 240 | 1206 to 1440 | window 2, settled, scored, in V |
| 241 to 250 | 1446 to 1500 | unused: the context a third window would have |

A method that selects reads ticks 1 to 120 to fit and ticks 121 to 240 to validate; MR and BL read ticks 1 to 240. At b = 5, F is ticks 1 to 480 (windows 1 to 4) and V ticks 481 to 600 (window 5); at b = 10, F is ticks 1 to 960 and V ticks 961 to 1200. At every budget the data of b are a prefix of the data of any larger budget. These counts, and the disjointness of contexts, scored readings, F and V, were checked by enumeration while writing this revision; I1 turns them into tests.

### 5.8 Normalisation, and variables that do not vary

Scales used inside a model, the centring and scaling of the inputs of a network or the scale of its outputs, are computed from F only, for each replicate and budget: the mean and standard deviation of each variable over the rows of F. The weights of the loss and of the metrics are the noise levels of the data sheet, fixed and independent of any data.

A standard deviation is positive only if the variable takes at least two values. In the development protocols of M1 every excursion moves all four inputs and the states respond, so a fitting part with one excursion gives positive scales for every variable. That is a property of P3, not a guarantee of the code: if a scale is zero, the fit is refused with an error that names the variable, and the refusal is a training failure of that model, replicate and budget (section 8.7). No scale is invented, no epsilon is added, and no data outside F is consulted to find one.

## 6. Interpolation and extrapolation

### 6.1 Definitions

* A new sequence from the same distribution: a test run made by the same protocol and amplitudes as the development runs, with another excitation seed and another noise. At small budgets many of its excursions go to corners that the fitting part never visited. That is still in distribution, because the distribution is P3's, not the list of corners seen.
* Outside the training region: a test that takes the inputs, and with them the states, beyond what the development runs covered. Here the development runs are made with every amplitude halved (A5: q and C_Af +-5 %, T_f and T_c +-2.5 K) and the test runs with A10. The training region is a box inside the test region, and every excursion of the test goes beyond it on all four inputs.
* A new kind of sequence: the single-input steps, whose 600 s holds reach the steady states of the stepped inputs, at the centres of the faces of the input box, which P3 never applies. It is a change of protocol inside the validated region, a weaker notion of extrapolation.

A descriptive measure, computed from available data only: for each test point, reading and inputs, the distance to the nearest point of the fitting data in normalised units. It describes how far out a window lies; it does not define the partitions.

### 6.2 Staying inside the validated domain

The test region is A10 under P3, exactly what D-019 accepted. The training region A5 is new. Smaller excursions from the verified steady state are expected to stay further inside the envelope and to recover sooner, but expected is not verified. Before any A5 data are generated, P3 at A5 receives the verification P3 received at A10 in M0-E03b: recovery after every corner within the tolerances of P3, all 256 ordered pairs of corners accepted, and the envelope. The lead needs no new physics, only its token in the identity. Nothing else new is proposed: no amplitude above A10, no other rest, hold, plant or operating point.

## 7. Knowledge, estimation and identifiability

### 7.1 What a model may know

| Quantity | In M1 |
|---|---|
| V, rho, cp, dH, the four inputs and their nominal values | known, from the export |
| Balances of a CSTR with A -> B, perfect mixing, constant volume | known |
| Noise levels of the sensors | known, from the data sheet |
| First-order Arrhenius form, constant UA | the modeller's assumption |
| k at 350 K, UA | estimated on the target, from the textbook values as a start |
| E/R | question Q1 |
| The saturating form, K_sat, the true k0 | hidden; oracle only |
| UA_ref, alpha, T_ref, the linear form of UA(T) | hidden; oracle only |
| Initial states, exact states, the nominal steady state computed by the simulator | hidden; initial states are estimated from readings |

The rate constant is estimated at a reference temperature, k(T) = k_350 exp(-(E/R)(1/T - 1/(350 K))), where 350 K is the known nominal feed temperature. Estimated as k0 and E/R, the two are correlated almost perfectly; at a reference inside the range of the data most of that correlation goes. The model is unchanged; only what the optimiser sees changes.

### 7.2 E/R

The value the modeller holds, 8750 K from the textbook, is also the true value, by the design of D-005 and D-006. Fixing it hands the modeller the correct value of a parameter that is otherwise estimated. Estimating it exposes it to the compensation below. This is question Q1.

### 7.3 Compensation between kinetics and UA

When the structure of a model is wrong, its estimates settle where the wrong structure imitates the right one best under the data and the objective used, not at the true values. Here:

* At the nominal steady state, the mass balance fixes k(T_ss) and the energy balance then fixes UA. A re-estimated model can reproduce the steady state whatever E/R is: k absorbs the saturation at the nominal concentration, UA the conductance at the nominal temperature.
* Away from it, a first-order law with the right rate at the nominal state responds too strongly to C_A: its sensitivity exceeds the true one by the factor 1 + K_sat C_A, 1.76 at the target's nominal state. The constant UA misses the extra cooling that UA(T) gives when the reactor is hot. A shifted E/R changes how strongly the rate responds to temperature, and since C_A and T move together during the excursions, hot with little A and cold with much, it can offset part of both errors. The estimate of E/R then depends on the excitation and not only on the chemistry.
* k and UA are coupled through the energy balance, since at the steady state the heat released by the reaction must equal the heat removed. They are estimated together, and their correlation is reported.
* In a hybrid that also estimates k_350, E/R and UA, the learned correction and those parameters can trade off against each other, and the split between them cannot be identified from data. A trade-off guarantees nothing about predictions: two splits that fit the fitting data equally well can predict differently where the data did not go, which is where extrapolation is judged. Mechanism is judged on the whole rate and the whole conductance, never on a correction factor.

M0-E08 showed that UA(T) changes the trajectories when everything else is known and there is no noise. It did not show that UA(T), or anything else, can be identified from noisy readings by a model that re-estimates a constant UA.

### 7.4 The minimal diagnostic (M1-E01, to be registered before it is run)

Three parts, one script, no new benchmark data:

1. Available, a priori: the Fisher information of (ln k_350, E/R, ln UA) for the first-order model under P3 at each budget, with the noise of D-020, at parameters that reproduce the nominal steady state observed in the development data of M0. It is computed twice. First with the initial state of every window taken as exact. Then with the error of the context mean included as a covariance of the outputs, Sigma = D_sigma + G Sigma_0 G^T, where D_sigma is the diagonal of the noise variances of the scored readings, G the sensitivity of the outputs of a window to its initial state, and Sigma_0 = diag(sigma_CA^2, sigma_T^2) / 10 the covariance of the context mean; windows are independent of each other, since their contexts are different readings; the covariance of the least-squares estimator actually used, with weights 1/sigma^2 and the initial state fixed at the context mean, is then the sandwich (S^T W S)^-1 S^T W Sigma W S (S^T W S)^-1. The first says what a budget could determine if the model were right and the initial state known; it is optimistic, and the difference between the two says what the rule for the initial state costs.
2. Available, on the development data of M0: the profile of the fitting objective against E/R, with E/R held on a grid and k_350 and UA re-estimated, on the target runs of `m0-e05`. It shows how sharp the minimum is, and where it lies, under the real and wrong structure. It is exploratory in the sense of section 5.1.
3. Oracle, kept apart: the first-order model fitted to the noise-free true trajectories of the 16 corner windows, each started from the exact nominal steady state, with equal weight per corner and the objective of section 9.1 over the scored readings of each window. Its minimiser, called pseudo-true below, is defined by that set of windows and that objective. It is not the limit of infinite data of the benchmark, whose windows start from noisy context means, whose corners are drawn with replacement, and whose fits use prefixes of runs; how far the two differ is not something this part shows. And, in the manner of M0-E08, how far the kinetic mismatch alone moves the trajectories, beside what M0-E08 measured for UA(T) alone. Outputs under `experiments/` only.

Parts 1 and 2 use available information only. Part 3 is never used to choose or tune an ordinary model. Nothing more is planned: no global sensitivity analysis, no identifiability analysis of network weights, which have no physical meaning to identify, and no design of experiments. For the hybrid, the substitute is a recovery test on synthetic data with a known correction (I3), which separates the recovery of trajectories from the recovery of the mechanism, and the oracle comparison of the learned rate with the true one at the end.

### 7.5 An exploratory check made while writing the first version (2026-09-22)

Status. A script outside the repository, run on the repository's code at `3e8f0d1`, not registered and not a result of M1. It is kept on record so that what was known when this plan was written can be traced. M1-E01 recomputes what matters of it with a committed script, with the windows of section 4.3 and the initial state treated as section 7.4 says, before any of it is used as a result. The first version of this plan described it without its numbers; they are given here for traceability, with the status above.

Method. Target plant; the 16 corner windows of P3, 120 s at a corner then 600 s of rest, each started from the exact nominal steady state; the 120 readings from 6 s to 720 s of each window, not the 110 of section 4.3; the noise of D-020; LSODA with rtol = atol = 1e-10. The point at which the Fisher information was evaluated took the nominal steady state from the simulator, where M1-E01 takes it from readings.

| Quantity | Value |
|---|---|
| Fisher information of (ln k_350, E/R, ln UA), initial state taken as exact, at k_350 and UA that reproduce the nominal steady state with E/R = 8750 K: standard error of E/R | 75 K (0.85 %) for one excursion, 24 K for ten |
| Correlations of the estimates | k_350 with E/R -0.48; k_350 with UA +0.73, and +0.90 with E/R fixed |
| Oracle fit of the first-order model to the noise-free truth, equal weight per window | E/R = 9590 K against the true 8750 K; UA = 1326 W/K against 1347 W/K at the nominal point |
| Root mean square error in sigmas, C_A and T | nominal model 11.8 and 10.6; oracle fit with E/R free 2.5 and 1.8, with E/R fixed at 8750 K 2.7 and 1.9 |
| Kinetic mismatch alone: first-order kinetics with the nominal rate, true UA(T) | 2.9 and 2.2 |
| Conductance mismatch alone: true kinetics, constant UA at the nominal point | 0.34 and 0.45; largest temperature difference 1.76 K, against 1.79 K in M0-E08 on longer sequences |
| The same on the source, for comparison | oracle fit E/R = 9637 K; conductance mismatch alone 0.72 and 0.78 |

What it may have informed, and so cannot be presented as independent of it: the framing of Q1 as a question of bias rather than of variance, and the recommendation to estimate E/R; the recommendation of Q4, although its argument rests on the structure of the balances and on M0-E08; the hypotheses and expectations of section 10; the content of M1-E01. Its Fisher information took the initial state as exact and is optimistic for that reason (section 7.4).

## 8. Models and a fair comparison

### 8.1 Mechanistic

* MN, nominal: the modeller's equations with the textbook values, nothing fitted.
* MR, re-estimated, the comparator: the same equations with k_350 and UA, and E/R according to Q1, estimated on all b windows of the budget with the loss of section 5.5, by `scipy.optimize.least_squares`. It selects nothing from data. It starts from the textbook values and from a list of further starting points declared in advance; the reported fit is the endpoint with the lowest objective, the first in the declared order on a tie, and every endpoint is recorded (section 8.7). No new dependency.
* MR_F: the same procedure on the fitting part F only. It is the starting point of the hybrids (section 5.5), and its score is reported beside MR's as a measure of what holding out V costs a mechanistic model.

### 8.2 Hybrid

The main direction of `AGENTS.md`: a continuous-time model whose learned parts are physical terms inserted in the known balances.

* HK, proposed as the first hybrid. The rate is the modeller's first-order Arrhenius law multiplied by a learned positive factor, r = k_350 exp(-(E/R)(1/T - 1/(350 K))) C_A g(C_A, T), with g > 0 and g = 1 when training starts. It enters both balances, as -r in the mass balance and as (-dH / (rho cp)) r in the energy balance. By construction the rate is never negative and vanishes without A, and the stoichiometry and the heat of reaction are the known ones. It starts from MR_F of the same replicate and budget (section 5.5). UA is constant and estimated. The arguments of g are those a rate law can depend on, composition and temperature, chosen for that reason and not from the simulator.
* HU, ablation: the constant UA multiplied by a learned positive factor of T and T_c, the temperatures a film coefficient depends on; first-order kinetics as in MR.
* HKU, ablation: both factors.

Why the kinetic correction first. The kinetic law is the least certain part of the modeller's model, while a constant UA is an ordinary approximation. The mass balance contains the rate and no other unknown, with C_A measured and q, V and C_Af known, so a kinetic correction is constrained by both balances and a thermal one by the energy balance alone. Telling the two apart rests on the C_A channel: in the energy balance, a rate that is wrong in its dependence on temperature and a conductance that is wrong in its dependence on temperature produce similar errors in T, and only C_A says which it is. Two things make that harder. The excursions of P3 move C_A and T together, so each learned function is seen along a narrow band of its arguments, and its dependence on each argument separately is weakly constrained. And on the target the thermal mismatch is small against the noise (section 2), so HU has little to learn from, least of all at small budgets. HK is therefore the hybrid about which the data carry the most direct information. HU and HKU are run beside it so that the comparison does not rest on that argument.

### 8.3 Black box

* BN, a neural ODE: dx/dt = D N((x - m_x) / s_x, (u - m_u) / s_u), with the scales m, s and D computed from F (section 5.8). It keeps the temporal formulation of the hybrid, continuous time with the inputs held between samples, and the same data, initial states, rollout and loss, so that the main remaining difference between BN and HK is the structure. Their capacity and parameterisation differ too, and are reported (section 8.5). The comparison therefore speaks for these two models as trained here, not for physical structure in general. A discrete model at 6 s, of the NARX kind, would be simpler to train but would differ from the hybrid in two things at once, structure and time. `AGENTS.md` allows it later as a baseline; it is not proposed now.
* BL, the simplest data-driven reference: a continuous-time linear model around the nominal inputs, dx/dt = A (x - x_e) + B (u - u_n), with u_n the known nominal inputs and 14 parameters, fitted on all b windows with the loss of section 5.5, and with the effect of the inputs estimated only along the directions the data excite (section 8.7). If BN does not beat BL at a budget, then this network, trained this way, drew no advantage from nonlinearity at that budget; that does not show that no model could.

### 8.4 Oracle

* OT: the true equations with the true parameters, run on the truth side only, on the same windows and from the same context means as every candidate. Its score is a diagnostic reference: what the true equations obtain under this protocol, including the error of the initial state. It is not a lower bound. On a finite test set a candidate can score below it by chance, and a model whose dynamics soon forget a noisy initial state, for instance by returning quickly to a steady state it learned from many rests, can score below it systematically early in a window. A model that predicted the true state exactly would have E[J^2] = 1 (section 9.1); no candidate can be expected to reach that, since every prediction starts from a noisy context mean.
* OF, optional: the true structure with its parameters estimated on the budget, by the same procedure as MR. It separates the cost of estimating parameters from the cost of a wrong structure. It is fitted on benchmark data, so it is part of the registration and runs after it.

Both are reported in their own table, never ranked with the candidates, and computed by a script on the truth side whose outputs go under `experiments/`.

### 8.5 What is the same and what differs

The same for every candidate: the budget and its data, the windows, the contexts and initial states, the inputs, the weights of the loss for those that are fitted, the integrator of the evaluation and the test sets.

What differs, reported with every result:

| Model | Knows | Parameters fitted | Data used to fit | Selection |
|---|---|---|---|---|
| MN | balances, known parameters, textbook values | 0 | none | none |
| MR | the same; textbook values only as a start | 2 or 3 | all b windows | none |
| BL | the data only | 14, the input effect restricted to excited directions | all b windows | none |
| BN | the data only | of the order of 10^3 | F | checkpoint and grid, on V |
| HK | as MR; starts from MR_F | those of MR, plus of the order of 10^2 | F | checkpoint and grid, on V |
| HU, HKU | as HK | as HK | F | as HK |
| OT | the true physics | 0 | none | none |

Selection effort, for every model that selects and every budget: the configurations tried, the fits run, their time, the validation data used. The grids are small, declared in the registration and the same for every budget.

The integrator of the evaluation is one reference integrator for every model, with tolerances tight enough that its error is below 1 % of sigma, checked on a sample of windows against a tighter tolerance, so that no difference between models is a difference between integrators. A model may be trained with another scheme; what is evaluated is its continuous-time equation.

### 8.6 The training framework is not chosen

MN, MR and BL need nothing beyond SciPy. The networks need gradients through a rollout. Candidates: PyTorch with torchdiffeq; JAX with diffrax; CasADi, which writes a small network as an expression and estimates with IPOPT; SciPy with sensitivity equations written by hand. Criteria: exact gradients through a rollout with input changes, checked against finite differences; results that are deterministic on a CPU with fixed seeds; installation on Windows and on the Linux CI with Python 3.12 and 3.13, and what it adds to the time of CI; speed on windows of 110 scored readings; maturity; whether an equation stays readable. The choice is made in I3 from a short comparison of two candidates on development data, and recorded as a decision. The line of `docs/architecture.md` that named PyTorch was the charter's guess, not a decision.

### 8.7 Fitting at small budgets: identifiability, several solutions, failures

1. A model does not have to be identifiable to be fitted and scored: what is evaluated is its predictions. What is required is a fit that can be reproduced, from declared starting points, seeds, regularisation and optimiser settings, with a declared rule for choosing among endpoints.
2. MR and MR_F. The conditioning of the fit, the singular values of the Jacobian of the scaled residuals at the solution, is reported as a diagnostic and does not accept or reject a fit. Among the declared starting points the endpoint with the lowest objective is kept, the first in the declared order on a tie. Every endpoint is recorded with its objective, and the spread of each parameter over the endpoints is reported, so that several solutions show up without a threshold deciding what counts as one.
3. BL. The matrix B can only be determined along the input directions its fitting data excite. It is estimated on the span of those directions, in level units (the signs of the corners), and is zero on the complement: the minimum-norm convention, part of the definition of BL. The rank of the excited directions is computed exactly from the integer levels. At b = 2, BL sees at most two corners and so at most two of the four directions; its predictions for the others follow from the convention, which is stated with every result of BL at small budgets.
4. Networks, BN and the hybrids. Their weights are not identifiable and are not expected to be. The fitted network is the result of the declared training from the declared initialisation and seed, with the regularisation of the chosen configuration of the grid, which is part of the method. Where there are several minima, the seed decides which is found; there is one declared seed per replicate and configuration, and no reseeding after a result.
5. Training failures. The procedure of every method declares how it treats a rollout that fails during fitting, for instance as a rejected step of the optimiser; a failure it does not treat ends the fit. A training failure is then: an optimiser that stops with an error, a loss that is not finite, a failed rollout that ends the fit, a refused scale (section 5.8), or a grid whose every checkpoint fails on V. Each is a training failure of that model, replicate and budget, recorded with its reason, not retried with other settings and not replaced. The replicate stays in every table (section 9.3).

## 9. Evaluation

### 9.1 Metrics, and what readings can and cannot say about the true state

For each channel, the root mean square error of the predictions against the scored readings, in physical units, mol/m^3 and K, per window, per phase and pooled over a test set.

To put the channels together, and to read an error against the noise, each is divided by its sigma from the data sheet. The combined score is J = sqrt((MSE_CA / sigma_CA^2 + MSE_T / sigma_T^2) / 2), pooled over the scored readings it is computed on. The sigmas are fixed, known and independent of the data. Alternatives set aside: the spread of the fitting data, which changes with the budget and the replicate; the nominal values, which would make any temperature error look negligible.

A scored reading is y = x + e, the true state plus the noise of the sensor. If e has zero mean and variance sigma^2 and is independent of everything the prediction p was computed from, the fitted model, the data it was fitted and selected on, and the context of its window, then the expectation of (p - y)^2 given p is (p - x)^2 + sigma^2. Pooled over scored readings, MSE - sigma^2 is then an unbiased estimate of the mean squared error against the true states at those instants, and J^2 - 1 its normalised form. In M1 the conditions hold for the scored readings of test runs: context and scored readings are different samples (section 4.3), and D-020 makes the noise independent from sample to sample, of zero mean and of known variance. For a real instrument they would hold only as far as its noise is known that well.

They do not hold for:

* a context reading scored against a prediction started from it: for a constant state and ten readings of variance sigma^2, the error of their mean has variance 0.1 sigma^2, but its difference from one of those readings has variance 0.9 sigma^2, and subtracting sigma^2 gives -0.1 sigma^2, not the error. The index contract never scores a context reading in its own window;
* the error on fitting data, since the model was fitted to that noise: MSE - sigma^2 is biased low there;
* the validation score of the configuration selected on it, which selection makes optimistic.

The estimate is reported as it comes, negative values included, never clipped. Its square root is not an unbiased estimate of a root mean square error and is not reported as one. The error against the exact states is computed directly only by the oracle diagnostics (section 9.6).

### 9.2 Phases

Every score is given pooled and for each phase of section 4.3, so that the long settled phases cannot hide an error during the excursions. The rollout over whole runs is reported apart. Two further views: by corner, and for the hottest windows.

### 9.3 Failures and physical violations

Three kinds, kept apart:

* a training failure (section 8.7): there is no model for that model, replicate and budget;
* an integration failure: a test window whose rollout fails, because the integrator reports a failure, a state stops being finite, or the guard on the number of evaluations of the right-hand side is exceeded;
* a physical violation: a completed rollout whose prediction breaks a bound given below. It keeps its score, and the violation is reported beside it.

How failures enter the conclusions:

* The primary score of a model, replicate and budget on a test set exists only if the model was trained and every window of that test set completed. Otherwise that replicate is a failure of the model at that budget. No error is invented for it, and the replicate is not dropped: every table lists, for each model and budget, the replicates with a score, the replicates with a training failure, the replicates with integration failures, and the number of failed windows.
* In a paired comparison on a replicate, a failure loses to any score and ties with another failure. The primary comparisons of section 9.7 are counted on these paired outcomes. A model that fails on hard windows gains nothing from their absence, because it loses those replicates.
* The error conditional on success, pooled over the windows that completed, is reported separately and labelled so. No primary conclusion rests on it alone.

A prediction is physically invalid if, at any sensor instant of its window, C_A is negative, C_A exceeds the larger of its initial value and the richest feed applied, or T falls below the smaller of its initial value and the coldest of feed and coolant applied. These bounds follow from the balances for any rate that is not negative and vanishes without A, and any conductance that is not negative; they use nothing hidden. MN, MR and the hybrids satisfy them by construction, which a test checks; the black boxes do not, and their violations are counted per window and replicate. A comparison won with physically invalid predictions is reported with that caveat.

### 9.4 The terms a black-box equation implies

For a continuous-time black box with right-hand side f = (f_CA, f_T), evaluated along its own predicted trajectory at the scored instants:

* the implied rate, r_imp = (q/V)(C_Af - C_A) - f_CA, which must not be negative;
* the implied heat flow through the wall, Q_imp = V rho cp [(q/V)(T_f - T) + (-dH / (rho cp)) r_imp - f_T], in W: the heat that the energy balance of the model says leaves through the wall.

A conductance that is not negative carries heat from the hotter side to the colder: Q_imp (T - T_c) >= 0, with Q_imp = 0 where T = T_c. A point where Q_imp and T - T_c have opposite signs, or where T = T_c and Q_imp is not zero, is incompatible with any non-negative conductance, and is counted as such. That sign condition is the reported test; it needs no division.

The quotient Q_imp / (T - T_c) is not reported as a conductance. Where T = T_c the energy balance does not determine a conductance at all; near it the quotient amplifies any error of Q_imp without limit; and where the sign condition fails, a negative quotient would not be a conductance but a sign that the model's energy balance cannot be closed by heat transfer through the wall. To show the implied law without dividing, Q_imp is plotted against T - T_c over the visited states. For MN, MR and the hybrids the implied rate and conductance are the ones written in their equations, non-negative by construction.

### 9.5 Three claims kept apart

* Conservation by construction. The mechanistic and hybrid models close the mass and energy balances of their own states because their equations are those balances. It is a property of the structure, stated once per model, and it does not show that their rate or their conductance is right.
* Predictive error: sections 9.1 to 9.3.
* Mechanism: whether the fitted or learned rate and conductance are the true ones. Only an oracle comparison can say (section 9.6).

### 9.6 Available metrics and oracle diagnostics

Available, computed from exports and the known specification: every metric of sections 9.1 to 9.4, the selection effort, the distance of test points from the fitting data, the distinct corners and the rank of the excitation of every budget.

Oracle, computed by a separate script on the truth side, outputs under `experiments/`, never fed back into a choice: the errors against the exact states; the error of the predicted peak of each hot excursion against the true peak; the fitted or learned rate against r_true, and the conductance against UA_true(T), over the states the test visits; the estimates against the pseudo-true values of section 7.4; OT and OF.

### 9.7 Replicates, test runs, and what each uncertainty means

The independent units are the development replicates, for the variability of training, and the runs of a P3 test set, for the variability of that test set. Windows and readings are not independent units.

The runs of test-A10 and test-A5 are draws from the P3 distribution: each has its own random sequence of corners, and another draw would give other runs. The eight runs of test-steps are not a sample of anything: they are the eight fixed conditions of D-026, each input moved up and down once. Their results are reported for each condition and as a fixed aggregate, the score pooled over the eight, each with its spread over the training replicates. No bootstrap is computed over them, and no repetition of them is added now.

| Summary | What it estimates | What it holds fixed |
|---|---|---|
| median, quartiles and extremes of the per-replicate score | the spread of the score over training realisations: excitation, noise and training seed | the test set |
| paired outcomes over replicates: wins, losses, failures | how consistently one model beats another across training realisations | the test set |
| bootstrap over replicates of the mean paired difference, computed only when both models have a score in every replicate | the uncertainty of that mean over training realisations | the test set |
| cluster bootstrap over the runs of test-A10 or test-A5, resampling whole runs and never windows | the sampling variability of that finite P3 test set | the fitted models |
| test-steps: per condition and the fixed aggregate over the eight, with their spread over replicates | the result on those eight conditions, and how it varies with the training realisation | the conditions, which are fixed and not sampled |

None of them alone is the whole uncertainty, and every interval says which one it is. With eight runs in test-A10 and four in test-A5 the cluster bootstrap is coarse, and it is presented as such.

The primary comparisons are named in the registration: HK against MR and HK against BN, on test-A10 at every budget and on the extrapolation test. With ten replicates a paired sign count can reach a two-sided 5 % level, at 9 of 10 (p = 0.021) or 10 of 10 (p = 0.002), and 8 of 10 cannot (p = 0.109). The thresholds of the hypotheses and the rules for reading them are fixed in the registration.

Every conclusion of M1 is about the models, optimisers, grids, data and procedures evaluated here.

### 9.8 The test sets

The test sets are defined, seeds included, in the registration, and generated only after the technical freeze (section 12). They are evaluated once. Nothing is chosen on them: not an architecture, a hyperparameter, a budget, a component, nor which results to show. If a defect of the software is found after the evaluation, it is fixed and recorded, every model is evaluated again, and the log says so.

## 10. Hypotheses, and when they were formed

* H1, interpolation and data efficiency: on test-A10, HK scores lower than MR from some budget on, and lower than BN at small budgets; the gap to BN narrows as the budget grows.
* H2, extrapolation: of the models trained at A5, HK loses less than BN between test-A5 and test-A10. MR is expected to lose accuracy as its structural error grows away from the nominal point.
* H3, protocol shift: the candidates rank on the fixed aggregate of test-steps as they rank on test-A10; the ranking in each of the eight conditions is reported beside it.
* H4, physical validity: the black boxes produce physically invalid predictions, or imply a negative rate or a heat flow incompatible with any non-negative conductance, in some windows, more often at small budgets and outside the training region. That the structured models produce none is a check of their implementation, not a finding.
* H5, mechanism (oracle, secondary): over the visited states, the rate of HK is closer to the true rate than MR's first-order law is.

When they were formed. These hypotheses, and the expectations below, were written on 2026-09-22 after the exploratory check of section 7.5 had been seen. They are informed by it and are not independent of it. Expectations: MR's error will be dominated by its structure rather than by its parameters, so its curve against the budget will be nearly flat, which is what the check's Fisher information and oracle fit suggested; BN starts from nothing; HK starts from MR_F and improves on it only if its correction, trained as declared, lowers the criterion on V.

What outcomes would and would not show. HK no better than MR at any budget would show that this hybrid, with this correction, architecture and training, did not improve on MR at these budgets; it would not show that no correction can be learned from these data. BN reaching HK at the larger budgets would show that, for these two models, the structure made a difference at small budgets and not at large ones, within the budgets tried.

Confirmatory status. Only the hypotheses, thresholds and rules of interpretation written into the registration of the benchmark, before any fit on benchmark data, are confirmatory. Anything proposed after benchmark fits have been seen is exploratory and labelled so.

## 11. Limitations of the design

* One target plant, one process, one reaction, and two hidden mechanisms chosen in D-004, one of them weak on the target. The conclusions are about this setting.
* The noise is ideal (D-020) and the inputs exact. Real data would bring bias, drift, delays and gaps.
* One family of excitation for training, the corners of P3, drawn with replacement. Data efficiency depends on the design of the excitation, which is not varied.
* Extrapolation is examined in one direction, amplitude, from a box of half the size. A new operating point belongs to M3.
* Validation is a temporal hold-out inside the run of each replicate, not an independent run; it serves selection only.
* Scores against readings contain sigma; the distance to it is estimated under the conditions of section 9.1, not observed.
* Ten replicates and one test set per question: differences smaller than their spread will not be resolved, and the report will say so.
* The grids are small. A learned model might do better with more tuning; the selection effort is reported so that this can be judged.
* The Fisher information of section 7.4 assumes the model is right. The profile and the oracle fit are there because it is not.
* The hypotheses were written after an exploratory check (sections 7.5 and 10).

## 12. Definition of done for M1

1. The confirmatory benchmark, M1-E03, is registered in the experiment log from a clean commit before any benchmark data are generated or any model is fitted on them: the definitions of every data set with their seeds, test sets included; budgets; replicates; the index contract; the division into fitting and validation; metrics; grids; primary comparisons; thresholds; rules of interpretation and of failure; the secondary and exploratory analyses, named as such.
2. MN, MR, BL, BN and HK, with HU and HKU as ablations, are implemented and tested, and fitted on every budget and replicate as registered. Every failure is recorded, and every deviation from the registration is recorded before the technical freeze.
3. The technical freeze: a commit that fixes the code, the configurations, and the identities of the fitted models and of their selection records, before any test data exist.
4. The test sets are generated after the freeze from their registered definitions and evaluated once. The curves of data efficiency, the extrapolation and the protocol shift are reported with their failures, physical violations and the uncertainty each summary estimates, whatever they show.
5. The identifiability diagnostic is reported and the treatment of E/R recorded as a decision.
6. The oracle diagnostics are reported apart, from the truth side only.
7. The boundary holds: `models` and `evaluation` never import `simulation` or `generation`, which a test on the import graph enforces; no hidden quantity reaches a model; the scan of the available branch finds nothing in the new data; and the access rules of section 5.6 are enforced by the code that builds contexts, scored sets, fitting and validation parts, with tests on the worked example of section 5.7.
8. Tests and ruff pass and CI is green. The experiment log, the decision log and a `docs/m1_audit.md` of the kind of M0's are written. Codex reviews and the owner accepts.

## 13. Iterations

Iterations I1 and I3 are software work on the data of M0 and on synthetic data; whatever they show is exploratory. The registration (I5) precedes any benchmark data. The technical freeze (I6) precedes any test data. The registration fixes the science before any benchmark result is seen; the freeze fixes the implementation before any test result is seen.

| Iteration | Deliverables | Accepted when |
|---|---|---|
| I0 Design, revisions 1 and 2 | this plan and its revisions; the active milestone in `AGENTS.md`, `CLAUDE.md`, `README.md` and the roadmap; D-029 and D-030 | the owner has answered section 15 and the plan follows the answers; Codex has reviewed revision 1. Both hold since 2026-09-24 (D-030, revision 2) |
| I1 Evaluation contract and mechanistic models | `evaluation`: the index contract, windows, phases, contexts, fitting and validation parts, metrics, records of failures, the validity bounds, the implied rate and heat flow. `models`: the interface of a continuous-time model, rollout with failures recorded, MN, MR, MR_F. A test on the import graph for both packages | tests on the worked example of section 5.7: no context scored in its own window, F and V disjoint, the counts of section 5.4; metrics equal to values computed by hand; the integration error of the rollout below 1 % of sigma against a tighter tolerance; MR recovers the parameters of data simulated by the modeller's own model, first noise-free from an exact initial state to the tolerance of the optimiser, then with the noise of D-020 and the context rule, each estimate within four standard errors of the true value over a declared set of seeds, the standard errors being those of the sandwich of section 7.4, which include the error of the initial state; suite and CI pass. A smoke run on `m0-e05` shows the code runs on a real export; it is exploratory |
| I2 Identifiability diagnostic, M1-E01 | parts 1 to 3 of section 7.4; the decision on E/R, if Q1 leaves it to the diagnostic | registered before it is run; run from a clean commit; logged as hypothesis, method, result and interpretation; oracle outputs only under `experiments/` |
| I3 Framework and learned models | the comparison of two frameworks and the decision; BL, BN, HK, HU, HKU; tests | HK with its correction at the identity equals MR_F to the precision of the integrator; gradients agree with finite differences; on data simulated by the modeller's model with a planted correction, and without the truth, recovery is tested in two separate ways: the fitted hybrid reproduces the trajectories, and the mechanism is recovered either as the correction itself, with the mechanistic parameters held at the values used to simulate, or, when parameters and correction are fitted together, as the total rate (the total heat flow for HU) over a domain of the states declared with the test; a joint fit is not required to recover the correction alone, since its split with the parameters is not identifiable (section 7.3); BL's convention holds on a rank-deficient case; tests pass on CI with the new dependency; the cost of one fit is measured on development data |
| I4 Protocol extensions and their verification | the lead, and the amplitude if Q2 is accepted, in the grammar of identities and in the data contract; the verification of P3 at A5 (M1-E02, registered, as M0-E03b) | identities and contract changed with tests; M1-E02 passes, or its failure is reported and Q2 is reopened |
| I5 Registration of the benchmark, M1-E03 | the registration and the data set definitions, test sets included | committed from a clean commit before any benchmark data exist; offered to Codex for review before I6 |
| I6 Development runs and technical freeze | the development sets generated; every fit on every budget and replicate; selection on V; deviations recorded; the freeze commit | the registration precedes all of it in the history; every fit logged with its failures and selection effort; no test data exist before the freeze |
| I7 Test evaluation | the test sets generated from their registered definitions; one evaluation of every frozen model; the oracle diagnostics by their own script; results in the log | generated after the freeze; every window and replicate accounted for; every hypothesis reported as it came out |
| I8 Analysis and closure | mechanism, from the oracle; `docs/m1_audit.md`; Codex's review; the owner's acceptance | the definition of done of section 12 |

## 14. The next iteration, recommended

I1. The owner has answered the questions of section 15 (D-030), and revision 2 takes in Codex's review of revision 1. I1 depends on the answers to Q5 and on the part of Q3 that says how fitting and validation share a budget; the other answers do not change it. It needs no new dependency and gives the mechanistic baseline a place before any network exists. Concretely:

* `process_transfer/evaluation/`: the index contract of section 4.3 and the division of section 5.5, built from the known inputs; the rule for initial states; metrics per channel in physical units and in sigmas, per phase; records of failures; the validity bounds; the implied rate and heat flow.
* `process_transfer/models/`: the interface of a continuous-time model; rollout under piecewise-constant inputs, sampled at the sensor instants, returning failures as records; MN, MR and MR_F on the equations of `modeller/cstr_first_order.py`, parameterised by k_350, E/R and UA.
* The access rules of section 5.6 enforced where data are handed over: a fit receives the index sets of its part and nothing else.
* A test on the import graph: `models` and `evaluation` never import `simulation` or `generation`. The generic integration code that both sides need is either moved to a neutral module or written again on the model side; which, is decided on inspection in I1 and recorded.
* Models read exports through `data.export.open_export_directory`, the known parameters from the export and the modeller's values from `configs/modeller_cstr.yaml`, never a plant configuration file.

## 15. Questions for the owner, and the answers

The questions are kept as revision 1 put them. The owner answered all five on 2026-09-24 by accepting each recommendation (D-030); the answer follows each question.

**Q1. E/R: fixed or estimated with the other parameters.**
(a) Estimated together with k_350 and UA. Realistic, and in the exploratory check of section 7.5 the better predictor of the two, because a free E/R can offset part of the missing physics; its estimate then depends on the excitation and not only on the chemistry (section 7.3), and in the hybrid it trades off against the correction, so mechanism is judged on the whole rate. (b) Fixed at 8750 K. This gives MR and the hybrids the exact value of a parameter that the black boxes have to learn from data. An engineer may know an activation energy from laboratory work, but rarely exactly; here the exactness is an artefact of the design. Attribution is cleaner, and MR loses the freedom to offset part of the missing physics, so it is expected to predict worse. (c) Decided by the diagnostic of section 7.4. If the difficulty is bias rather than variance, as the exploratory check suggested, the diagnostic will call E/R identifiable and (c) becomes (a) with an extra step that depends on data. Recommendation: (a) for every model, with (b) as a secondary analysis for MR and HK at two budgets, declared in the registration.
Answer (D-030): (a). E/R is estimated together with the other parameters; E/R held at 8750 K is a secondary analysis for MR and HK at two budgets, which the registration names.

**Q2. The extrapolation test.**
(a) Train at half amplitude, A5, and test at A10. Clean and symmetric, entirely inside the validated region at test time; it needs the verification of P3 at A5 before any data (section 6.2) and a token for the amplitude in the identities. (b) Train without the corners where T_c is high and test on them: extrapolation in one direction, towards the hot region that matters for the envelope; needs a restricted variant of P3, safe by the argument of the 256 pairs but new code. (c) The change of protocol only, the single-input steps: no new verification, but a weak notion of extrapolation. Recommendation: (a) as the primary test and (c) as a secondary one.
Answer (D-030): (a) as the primary test, once P3 at A5 has passed its verification; (c) as a secondary evaluation of a change of protocol.

**Q3. Budgets, replicates, and how fitting and validation share a budget.**
Proposed: b = 2, 5, 10, 20 and 40 excursions (exact records of 25, 61, 121, 241 and 481 min, 4 to 80 min at a corner); ten replicates, each one run of 40 excursions whose prefixes are the budgets; for a method that selects, validation on the last max(1, floor(b / 5)) windows of the prefix, a temporal hold-out under the index contract (section 5.5); no refit after selection; hybrids started from MR_F; MR and BL fitted on all b windows; training at A5 on b = 10 and 40 only. Alternatives: (i) validation on a separate run of the same replicate, counted in the budget: independent of the fitting run in its noise stream and carried state, but every budget then needs two leads and two prefixes, which changes what a budget is; (ii) a refit on F and V together after selection, with a schedule fixed in advance: every method then fits on all b, but the final model is checked by no held-out data; (iii) five replicates instead of ten: about half the cost, but a paired sign count over five replicates cannot reach a two-sided 5 % level even when one model wins all five (p = 0.0625); (iv) independent runs for each budget instead of prefixes, which adds noise to the shape of the curves. Recommendation: as proposed, with the number of replicates confirmed after the cost of one fit is measured (I3) and fixed in the registration (I5).
Answer (D-030): as proposed. Ten replicates is the initial proposal, to be confirmed from the cost of a fit before the registration.

**Q4. The first hybrid.**
(a) The kinetic correction, HK, as the hybrid of the primary comparisons, with HU and HKU as ablations. (b) The thermal correction first. (c) Both from the start. Recommendation: (a), for the reasons of section 8.2: the rate is constrained by both balances, and the thermal signal on the target is weak, so (b) would probably have little to learn and (c) invites a trade-off that the data may not resolve. The recommendation was written after the exploratory check of section 7.5 (section 10).
Answer (D-030): (a). HK is the main hybrid; HU and HKU are variants for comparison.

**Q5. The primary predictive task, and how it is scored.**
(a) Free rollout over the windows of section 4.3: the initial state is the mean of the ten context readings in (t0 - 60 s, t0]; 110 readings in (t0, t0 + 660 s] are scored, in phases of 20, 50 and 40; the last ten readings of each cycle are the context of the next window and are scored in no window; a lead of 60 s at the start of each P3 run; the primary score of a replicate exists only if every test window completed, and a failure loses every paired comparison (section 9.3); rollout over whole runs as a secondary evaluation; k steps ahead as a diagnostic. (b) Rollout over whole runs as the primary evaluation, which weighs steady-state bias more and dynamics less. (c) One step ahead as the primary evaluation, dominated by noise (section 4.4). A variant of (a) would keep a replicate with some failed windows and score it on the windows that completed; it is not proposed, because the score would then be computed on the windows the model happened to survive. Recommendation: (a).
Answer (D-030): (a). In a paired comparison a failure loses to a completed evaluation, and two failures tie.

Defaults that stand unless the owner objects, and none was raised with the answers of D-030: the target only, with a repetition on the source considered only after the target is done; BN and BL as the black boxes; the sigmas of the data sheet as the scale of the metrics; scales inside models computed from the fitting part, a zero scale refused as a training failure; failures reported, never replaced by a value; test sets generated only after the technical freeze; the data of M0 for software development and exploratory runs only.

## 16. Revisions

**Revision 1, 2026-09-23.** Codex audited `c2309cc` and `1dd6830`; the owner reviewed the findings and authorised a documentary correction before I1. The findings were checked against the text before anything was changed, and all nine held. What changed:

1. Validation leaked into the hybrid's start: MR was fitted on the whole budget, including the validation windows of the hybrid it initialised. The comparator MR still fits on all b windows; the hybrids now start from MR_F, fitted on F alone; every scale, initial value and pre-fitted component of a method that selects comes from F; V chooses checkpoints and configurations and nothing else; there is no refit after selection (sections 5.5, 5.6, 8.1, 8.2).
2. Partitions: the first version promised that no reading crosses partitions, yet divided one run between fitting and validation, and the first validation context reused readings of the last fitting window. The index contract of section 4.3 now makes every reading a context, a scored reading or unused, never two of these; a window scores 110 readings in phases of 20, 50 and 40, where the first version had 120 readings and 20, 50 and 50; F and V are disjoint, contexts included, and V is described as a temporal hold-out, not an independent trial; sections 5.1 and 5.7 define the three kinds of data and give a worked example with indices. The unit of the experiment and the meaning of the budgets are unchanged.
3. Scoring against noisy readings: a context reading is never scored in its own window; section 9.1 states when MSE - sigma^2 estimates the error against the true state without bias, and where it does not; section 8.4 describes OT as a reference under the protocol, not a lower bound.
4. Small budgets: sections 5.3, 5.8 and 8.7 say what happens with one excited direction, repeated corners, parameters that the data do not determine, several solutions, zero scales and failed fits.
5. The implied conductance is replaced by the implied heat flow and a sign condition, with no division (section 9.4).
6. Order of registration: the first version fitted MR on benchmark data (I4) before registering the benchmark (I6). The registration now precedes any benchmark data, the technical freeze is distinguished from it, and development runs are exploratory (sections 12 and 13). Section 7.5 keeps the exploratory check, now with its numbers, and sections 7.5 and 10 say which recommendations and expectations it may have informed.
7. Failures and statistics: training failures, integration failures and physical violations are kept apart; a failure counts as a loss at the level of the replicate instead of being excluded; the bootstrap of the test set resamples runs, and each summary says which uncertainty it estimates (sections 9.3 and 9.7).
8. Interpretations that claimed too much, about BN against BL, HK against MR, trade-offs between parameters and corrections, and the oracle fit as a limit of infinite data, are restricted to the models, procedures and data evaluated, and pseudo-true is defined by its windows and objective (sections 7.3, 7.4, 8.3 and 10).
9. Precisions: exact durations of the budgets beside rounded labels (section 5.4); the initial state in the Fisher information and in the recovery test of I1 (sections 7.4 and 13); the claim that five replicates halve the power is replaced by what a sign count can reach (Q3, section 9.7); `AGENTS.md` and `CLAUDE.md` said that `91206b2` carries the tag `m0-v1.0`, which is on `3e8f0d1`.

**Revision 2, 2026-09-24.** The owner answered Q1 to Q5 by accepting each recommendation (D-030), and Codex's review of revision 1 left two precisions, which the owner asked to take in. What changed:

1. The answers are recorded after each question of section 15 and in D-030. Section 3 says what they settle and what they leave open; sections 4 to 9 are no longer headed as proposals; the header, section 1, the row of I0 in section 13 and section 14 follow.
2. The cluster bootstrap over test runs is limited to the P3 test sets, whose runs are draws of the protocol. The eight single-input steps are fixed conditions: they are reported for each condition and as a fixed aggregate, with their spread over the training replicates, and no repetition is added (sections 5.2, 9.7 and 10, H3).
3. The recovery test of the hybrid in I3 no longer asks a joint fit of parameters and correction to recover the correction alone. It tests the trajectories, and the mechanism either with the parameters held fixed or as the total rate or heat flow over a declared domain (sections 7.4 and 13).
