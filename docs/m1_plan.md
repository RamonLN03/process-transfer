# Plan of M1: black-box and hybrid models fitted on the target plant

Proposed on 2026-09-22, the day M0 was closed. Nothing described here is implemented. The owner's instruction that started M1 fixes the question, the comparators and what is out of scope (D-029). Everything else in this file is a proposal until the owner has answered the questions of section 15. Each section says which of its parts are accepted, proposed or open.

## 1. The question

Does a continuous-time hybrid model improve data efficiency or extrapolation over simple alternatives, when every model is fitted on data of the target plant only?

Accepted (D-029):

* Comparators: the nominal mechanistic model, as the starting reference; the mechanistic model with its parameters re-estimated on the target; a black-box model trained on the target only; a hybrid model trained on the target only; and an oracle with the complete true physics, a separate diagnostic reference that is never a candidate.
* Not in M1: pretraining on the source, fine-tuning between plants, transfer policies and systematic studies of domain shift (M2, M3 and later).
* The hybrid is not assumed to win. A well-supported negative result is a valid result.
* alpha (0.005 1/K on the source, 0.002 1/K on the target) and the physical configurations of M0 are kept. The difficulty is not adjusted to favour a method.
* Ordinary models never contain alpha, the saturating law, K_sat or the true form of UA(T). A study that uses them is an oracle and is kept apart.

Proposed here, beyond that list: a linear black box as the simplest data-driven reference (section 8.3) and two ablations of the hybrid (section 8.2).

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

Two consequences. The textbook values were chosen to reproduce the source's nominal point (D-006), and both plants have the same nominal inputs (D-008), so the textbook model puts the target's steady state where the source's is: one steady state, at 250.01 mol/m^3 and 350.00 K, found with the repository's steady-state search on the modeller's equations while writing this plan. The nominal model's steady state for the target is therefore 12 sigma_CA and 10 sigma_T away from the real one. And the target's hidden conductance law is weak: what a constant UA equal to the conductance at the nominal point leaves out moves the trajectories by amounts of the order of the sensor noise (M0-E08).

What M0 does not hand over, and M1 has to build: a division of data into training, validation and test; budgets; windows and initial states; metrics; any model code. The data sets `m0-e05`, `m0-e06` and `m0-e07` were built to verify the data path and the protocols, and none of them is the benchmark of M1 (section 5.1).

Carried over from M0 and not blocking: the test of the exit code of M0-E08 checks the same aggregation that `main()` uses instead of calling `main()` (`docs/roadmap.md`). Nothing in M1 depends on it. It is left for whoever next touches that test, and no iteration below is planned around it. No other defect of M0 was found while preparing this plan.

## 3. What is accepted, proposed and open

* Accepted: the list of section 1 (D-029).
* Proposed, awaiting the owner: sections 4 to 14. The five choices that change the design most are the questions of section 15.
* Open, to be settled during M1: the training framework (section 8.6); the grids of the learned models; the thresholds of the hypotheses, fixed when the benchmark is registered and before any training run; whether M1 is repeated on the source plant as a second case, after the target.

## 4. The predictive task (proposed)

### 4.1 What is observed and what is predicted

A model predicts the state x = (C_A, T) of the target from two things: an initial state estimated from readings taken before the prediction starts, and the inputs over the horizon, which are known exactly. In a planned test, and in any use of a model for design or control, the future inputs are chosen rather than measured, so giving them to the model is not a leak. No reading taken after the start of a prediction is given to it. Predictions are compared with the readings at the sensor instants of the horizon.

### 4.2 The initial state

Both states are measured, with noise. The initial state of a prediction that starts at t0 is the mean of the ten readings at t0 - 54 s, ..., t0, taken under constant nominal inputs with the plant settled: at the end of a rest of P3, at the end of the lead of a step test, or during the lead of a run (below). The reading at t0 is of the state before any change applied at t0 (right-continuous convention), so it belongs to the set. The mean has a standard deviation of sigma / sqrt(10), 1.6 mol/m^3 and 0.16 K. A single reading would start every prediction one sigma off, an error that decays with the plant's time constant of about a minute and therefore weighs most during the two minutes of the excursion. Every model receives the same initial state; none estimates its own.

A run of P3 starts with an excursion at t = 0, so its first excursion has no reading before it. Proposed: every P3 run of M1 starts with a lead of 60 s at the nominal inputs. The lead is steady operation from the verified steady state, where M0-E06 showed the target's true state staying to within 1.3e-8 mol/m^3 and 2.1e-9 K over two hours, and what follows is P3 from that same state, so the verification of P3 applies as it stands. It needs a token in the identity of a run, a technical change of the grammar of the kind D-026 made.

### 4.3 Windows and phases

An excursion of P3 and its rest form the unit of evaluation: a window starts at the onset of an excursion and lasts 720 s, 120 readings, up to the onset of the next one. Its phases:

* excursion: the 120 s at the corner, 20 readings;
* return: the first 300 s of the rest, 50 readings, about five time constants, while the plant comes back;
* settled: the last 300 s of the rest, 50 readings, where the plant is back at its steady state to a small fraction of a sigma (M0-E03b: 1.3 mK at the end of the rest).

For a single-input step test the window starts at the step, at 600 s, and lasts 1200 s, in four phases of 300 s: hold transient, hold settled, recovery transient, recovery settled. Windows are found from the known inputs and the known nominal inputs alone.

### 4.4 One step ahead against free rollout

One step is 6 s. A prediction one step ahead starts from a reading that is one sigma off and is scored against another reading, so the error of any reasonable model is dominated by the two noises: in 6 s a plant with a time constant of a minute moves by about a tenth of its deviation. Differences between models, which are differences in how the state evolves over minutes, barely show. It is the wrong test for a model meant to predict the response to a planned change.

Proposed as the primary evaluation: free rollout over each window. The model receives the initial state and the inputs of the window, integrates without seeing any further reading, and is scored on the 120 readings of the window.

Secondary: free rollout over a whole run, two hours for a test run of ten excursions, from the initial state at its start. It exposes drift, instability and a wrong steady state, which a restart every 720 s would partly hide.

Diagnostic only: predictions k steps ahead from each reading, for k = 1 and 10.

## 5. Data and partitions (proposed)

### 5.1 The data sets of M0

`m0-e05`, `m0-e06` and `m0-e07` serve software development only: to check that loaders, rollouts, fits and metrics run on real exports, in tests and smoke runs. No number obtained on them is reported as a result of M1, and nothing that the benchmark fixes, a grid or a training setting, is chosen from how a model scores on them. The one exception is the profile of section 7.4, declared as a diagnostic. M1 reuses none of their excitation seeds or noise realisations.

### 5.2 New data sets, all on the target

| Set | Protocol | Runs | Record | Role |
|---|---|---|---|---|
| train-A10 | P3 with the lead, A10 | 10 runs of 40 excursions | 10 x 8 h | training, and validation inside the budget |
| train-A5 | P3 with the lead, half amplitudes (A5), if Q2 is accepted | 10 runs of 40 excursions | 10 x 8 h | training for the extrapolation test |
| test-A10 | P3 with the lead, A10 | 8 runs of 10 excursions, 80 windows | 16 h | the fixed evaluation set of D-011, for interpolation and extrapolation |
| test-A5 | P3 with the lead, A5 | 4 runs of 10 excursions, 40 windows | 8 h | the in-region reference of the models trained at A5 |
| test-steps | the eight single-input steps of D-026, a new noise realisation | 8 runs | 4 h | protocol shift |

Excitation seeds and master seeds are new: none that an earlier experiment, test or smoke run used (the experiment log lists them), and one master seed per data set. The test sets are defined with the others, seeds included, but generated only after every model configuration has been frozen (section 9.7).

Sizes: a run of 40 excursions has 4811 rows of six channels. The two training sets together hold about 580 000 measurements, the test sets about 100 000, well within what the path of M0 handled; ingestion time grows with the square of the number of runs (a known limitation of M0) and will be measured before the full sets are generated.

### 5.3 Partitions and replicates

The unit of partition is the run. Every run belongs to one partition, and no window, reading or run crosses from one to another. Training and test sets are different data sets, with different excitation seeds and different master seeds, so their noise is independent.

A replicate is one run of train-A10: its own excitation sequence and its own noise. Budgets are prefixes of it, the lead and the first b excursions with their rests. The data of b = 5 are part of the data of b = 10, so the curve of a replicate is monotone in data, and the ten replicates measure how much the result depends on which corners and which noise happened to come.

Excitation and noise are separated without handing a private seed to anyone. The excitation seed is part of the identity of a run, as in `target.p3.e0.x10.n0`. That is harmless: it only determines the inputs, and the inputs are recorded exactly anyway. The noise realisation is the `n<k>` of the identity, and its stream follows from the identity under the private master seed of the data set (D-021), which stays under `private/`. The same excitation with independent noise would be the same identity with another `n<k>`; no such pair is planned for the first benchmark.

The randomness of training (initial weights, order of the windows) has seeds of its own, recorded with each fit and unrelated to the seeds of the data.

### 5.4 Budgets

| Excursions, b | Record, h | Time at a corner, min | Division for a method that selects |
|---|---|---|---|
| 2 | 0.4 | 4 | 1 to fit, 1 to validate |
| 5 | 1.0 | 10 | 4, 1 |
| 10 | 2.0 | 20 | 8, 2 |
| 20 | 4.0 | 40 | 16, 4 |
| 40 | 8.0 | 80 | 32, 8 |

The record includes the minute of lead. The budgets follow those of D-011, 30 min to 8 h, in the unit that P3 is built of. Hours of record overstate the information: of the twelve minutes of an excursion and its rest, two are at a corner, about five are the return, and the last five are the nominal steady state again, which after the first rest adds little beyond averaging its noise. Results are therefore reported against excursions, with hours of record and minutes at a corner alongside.

Selection is paid for from the budget. A method that has something to choose from data, early stopping, a hyperparameter, whether to include a component, takes its validation data from inside b: the last fifth of the excursions of the prefix, at least one. A method with nothing to choose, the re-estimated mechanistic model, fits on all of b. That is the advantage of having nothing to tune, and the comparison keeps it. Nothing is chosen on the test sets, and nothing on data outside the budget. The grids and training settings of section 8 are fixed before any training run and are the same for every budget; if developing the code on the data of M0 changed one of them, a setting under which training does not run at all for instance, the registration says so.

### 5.5 Normalisation

Any scale used inside a model, such as the normalisation of the inputs of a network or the scale of its outputs, is computed from the fitting part of the budget only, for each replicate and budget, never from validation or test data. The weights of the loss and of the metrics are the noise levels of the data sheet, known and fixed, and depend on no data.

## 6. Interpolation and extrapolation (proposed)

### 6.1 Definitions

* A new sequence from the same distribution: a test run made by the same protocol and amplitudes as the training runs, with another excitation seed and another noise. At small budgets many of its excursions go to corners that the training prefix never visited. That is still in distribution, because the distribution is P3's, not the list of corners seen.
* Outside the training region: a test that takes the inputs, and with them the states, beyond what the training runs covered. Here the training runs are made with every amplitude halved (A5: q and C_Af +-5 %, T_f and T_c +-2.5 K) and the test runs with A10. The training region is a box inside the test region, and every excursion of the test goes beyond it on all four inputs.
* A new kind of sequence: the single-input steps, whose 600 s holds reach the steady states of the stepped inputs, at the centres of the faces of the input box, which P3 never applies. It is a change of protocol inside the validated region, a weaker notion of extrapolation.

A descriptive measure, computed from available data only: for each test point, reading and inputs, the distance to the nearest training point in normalised units. It describes how far out a window lies; it does not define the partitions.

### 6.2 Staying inside the validated domain

The test region is A10 under P3, exactly what D-019 accepted. The training region A5 is new. Smaller excursions from the verified steady state are expected to stay further inside the envelope and to recover sooner, but expected is not verified. Before any A5 data are generated, P3 at A5 receives the verification P3 received at A10 in M0-E03b: recovery after every corner within the tolerances of P3, all 256 ordered pairs of corners accepted, and the envelope. The lead needs no new physics, only its token in the identity. Nothing else new is proposed: no amplitude above A10, no other rest, hold, plant or operating point.

## 7. Knowledge, estimation and identifiability (proposed)

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

When the structure of a model is wrong, its estimates settle where the wrong structure imitates the right one best, not at the true values. Here:

* At the nominal steady state, the mass balance fixes k(T_ss) and the energy balance then fixes UA. A re-estimated model can reproduce the steady state whatever E/R is: k absorbs the saturation at the nominal concentration, UA the conductance at the nominal temperature.
* Away from it, a first-order law with the right rate at the nominal state responds too strongly to C_A: its sensitivity exceeds the true one by the factor 1 + K_sat C_A, 1.76 at the target's nominal state. The constant UA misses the extra cooling that UA(T) gives when the reactor is hot. A shifted E/R changes how strongly the rate responds to temperature, and since C_A and T move together during the excursions, hot with little A and cold with much, it can offset part of both errors. The estimate of E/R then depends on the excitation as much as on the chemistry.
* k and UA are coupled through the energy balance, since at the steady state the heat released by the reaction must equal the heat removed. They are estimated together, and their correlation is reported.
* In a hybrid that also estimates k_350, E/R and UA, the learned correction and those parameters can trade off against each other, and the split between them cannot be identified from data. Predictions do not suffer, but what the network learned cannot be read off the network alone. Mechanism is judged on the whole rate and the whole conductance, never on a correction factor.

M0-E08 showed that UA(T) changes the trajectories when everything else is known and there is no noise. It did not show that UA(T), or anything else, can be identified from noisy readings by a model that re-estimates a constant UA.

### 7.4 The minimal diagnostic (M1-E01, to be registered before it is run)

Three parts, one script, no new data:

1. Available, a priori: the Fisher information of (ln k_350, E/R, ln UA) for the first-order model under P3 at each budget, with the noise of D-020, at parameters that reproduce the observed nominal steady state. It says what each budget could determine if the model were right: standard errors and correlations.
2. Available, on development data: the profile of the fitting objective against E/R, with E/R held on a grid and k_350 and UA re-estimated, on the target runs of `m0-e05`. It shows how sharp the minimum is, and where it lies, under the real and wrong structure.
3. Oracle, kept apart: the first-order model fitted to the noise-free true trajectories of the 16 corner windows, the limit of infinite data, set against the true values; and, in the manner of M0-E08, how far the kinetic mismatch alone moves the trajectories, beside what M0-E08 measured for UA(T) alone. Outputs under `experiments/` only.

Parts 1 and 2 use available information only. Part 3 is never used to choose or tune an ordinary model. Nothing more is planned: no global sensitivity analysis, no identifiability analysis of network weights, which have no physical meaning to identify, and no design of experiments. For the hybrid, the substitute is a recovery test on synthetic data with a known correction (I5), and the oracle comparison of the learned rate with the true one at the end.

### 7.5 An exploratory check made while writing this plan

A scratch calculation outside the repository, not registered and not a result of M1, looked at parts 1 and 3 on the target. It suggested that the difficulty with E/R is bias rather than variance: the Fisher information would pin E/R down from a single excursion if the model were right, while the oracle fit of the first-order model moved E/R well above its true value and lowered the prediction error in doing so. It also suggested that on the target the kinetic mismatch moves the trajectories several times more than the conductance mismatch. Its numbers are not quoted here; M1-E01 measures them with the code of the repository. The recommendations of this plan are argued without them. They are recorded so that a reader knows they were seen.

## 8. Models and a fair comparison (proposed)

### 8.1 Mechanistic

* MN, nominal: the modeller's equations with the textbook values, nothing fitted.
* MR, re-estimated: the same equations with k_350 and UA, and E/R according to Q1, estimated on the budget by minimising the rollout error of section 9 over the windows of the budget, each channel weighted by 1/sigma^2 of the data sheet, with `scipy.optimize.least_squares`. The textbook values are the starting point, with a few restarts declared in advance against a local minimum. No new dependency.

### 8.2 Hybrid

The main direction of `AGENTS.md`: a continuous-time model whose learned parts are physical terms inserted in the known balances.

* HK, proposed as the first hybrid. The rate is the modeller's first-order Arrhenius law multiplied by a learned positive factor, r = k_350 exp(-(E/R)(1/T - 1/(350 K))) C_A g(C_A, T), with g > 0 and g = 1 when training starts. It enters both balances, as -r in the mass balance and as (-dH / (rho cp)) r in the energy balance. By construction the rate is never negative and vanishes without A, the stoichiometry and the heat of reaction are the known ones, and the hybrid starts where MR of the same replicate and budget ended. UA is constant and estimated. The arguments of g are those a rate law can depend on, composition and temperature, chosen for that reason and not from the simulator.
* HU, ablation: the constant UA multiplied by a learned positive factor of T and T_c, the temperatures a film coefficient depends on; first-order kinetics as in MR.
* HKU, ablation: both factors.

Why the kinetic correction first. The kinetic law is the least certain part of the modeller's model, while a constant UA is an ordinary approximation. The mass balance contains the rate and no other unknown, with C_A measured and q, V and C_Af known, so a kinetic correction is constrained by both balances and a thermal one by the energy balance alone. Telling the two apart rests on the C_A channel: in the energy balance, a rate that is wrong in its dependence on temperature and a conductance that is wrong in its dependence on temperature produce similar errors in T, and only C_A says which it is. Two things make that harder. The excursions of P3 move C_A and T together, so each learned function is seen along a narrow band of its arguments, and its dependence on each argument separately is weakly constrained. And on the target the thermal mismatch is small against the noise (section 2), so HU has little to learn from, least of all at small budgets. HK is the hybrid expected to be learnable; HU and HKU run beside it so that the answer does not depend on that expectation.

### 8.3 Black box

* BN, a neural ODE: dx/dt = D N((x - m_x) / s_x, (u - m_u) / s_u), with the scales m, s and D computed from the fitting data. It has the temporal formulation of the hybrid: continuous time, inputs held between samples, the same initial state, rollout and loss. HK against BN therefore isolates what the physical structure adds. A discrete model at 6 s, of the NARX kind, would be simpler to train but would differ from the hybrid in two things at once, structure and time. `AGENTS.md` allows it later as a baseline; it is not proposed now.
* BL, the simplest credible data-driven reference: a continuous-time linear model around the nominal inputs, dx/dt = A (x - x_e) + B (u - u_n), with u_n the known nominal inputs and 14 parameters, fitted by the same rollout loss. If BN does not beat BL at a budget, the nonlinearity is not learnable from that much data.

### 8.4 Oracle

* OT: the true equations with the true parameters, run on the truth side only, on the same windows and from the same initial states. Its error is the floor the evaluation leaves, the noise of the readings and of the initial state.
* OF, optional: the true structure with its parameters estimated on the budget. It separates the cost of estimating parameters from the cost of a wrong structure.

Both are reported in their own table, never ranked with the candidates, and computed by a script on the truth side whose outputs go under `experiments/`.

### 8.5 What is the same and what differs

The same for every candidate: the budget and its data, the windows, the initial states, the inputs, the weights of the loss for those that are fitted, the integrator of the evaluation and the test sets.

What differs, reported with every result:

| Model | Knows | Parameters fitted | Selection |
|---|---|---|---|
| MN | balances, known parameters, textbook values | 0 | none |
| MR | the same; textbook values only as a start | 2 or 3 | none; all of b is used to fit |
| BL | the data only | 14 | none |
| BN | the data only | of the order of 10^3 | a small grid, on validation inside b |
| HK | as MR | those of MR, plus of the order of 10^2 | a small grid, on validation inside b |
| HU, HKU | as MR | as HK | as HK |
| OT | the true physics | 0 | none |

Selection effort, for every learned model and budget: the configurations tried, the fits run, their time, the validation data used. The grids are small, declared before training and the same for every budget.

The integrator of the evaluation is one reference integrator for every model, with tolerances tight enough that its error is below 1 % of sigma, checked on a sample of windows against a tighter tolerance, so that no difference between models is a difference between integrators. A model may be trained with another scheme; what is evaluated is its continuous-time equation.

### 8.6 The training framework is not chosen

MN, MR and BL need nothing beyond SciPy. The networks need gradients through a rollout. Candidates: PyTorch with torchdiffeq; JAX with diffrax; CasADi, which writes a small network as an expression and estimates with IPOPT; SciPy with sensitivity equations written by hand. Criteria: exact gradients through a rollout with input changes, checked against finite differences; results that are deterministic on a CPU with fixed seeds; installation on Windows and on the Linux CI with Python 3.12 and 3.13, and what it adds to the time of CI; speed on windows of 120 readings; maturity; whether an equation stays readable. The choice is made in I5 from a short comparison of two candidates on development data, and recorded as a decision. The line of `docs/architecture.md` that names PyTorch was the charter's guess, not a decision.

## 9. Evaluation (proposed)

### 9.1 Metrics for each variable

For each channel, the root mean square error of the predictions against the readings, in physical units, mol/m^3 and K, per window, per phase and pooled over a test set.

To put the channels together, and to read an error against the noise, each is divided by its sigma from the data sheet. The combined score is J = sqrt((MSE_CA / sigma_CA^2 + MSE_T / sigma_T^2) / 2). The sigmas are fixed, known and independent of the data. Alternatives set aside: the spread of the training data, which changes with the budget and the replicate; the nominal values, which would make any temperature error look negligible.

The readings carry independent noise, so the expected squared error against readings is the squared error against the true state plus sigma^2. J = 1 is the floor of a perfect model started from a perfect state. MSE - sigma^2 estimates the error against the true state without bias and without the truth. It is reported as it comes, negative values included, never clipped.

### 9.2 Phases

Every score is given pooled and for each phase of section 4.3, so that the long settled phases cannot hide an error during the excursions. The rollout over whole runs is reported apart. Two further views: by corner, and for the hottest windows.

### 9.3 Failures and physical violations

A rollout fails when the integrator reports a failure, when a state stops being finite, or when a guard on the number of evaluations of the right-hand side is exceeded. A failed window is recorded with its reason. It is never dropped from the counts or given a substitute value. Every error is reported with the number of failures beside it, and a paired comparison uses the windows on which both models succeeded and says how many those are.

A prediction is physically invalid if, at any sensor instant of its window, C_A is negative, C_A exceeds the larger of its initial value and the richest feed applied, or T falls below the smaller of its initial value and the coldest of feed and coolant applied. These bounds follow from the balances for any rate that is not negative and vanishes without A, and any conductance that is not negative; they use nothing hidden. MN, MR and the hybrids satisfy them by construction, which a test checks; the black boxes do not, and their violations are counted.

For the continuous-time black boxes the terms their equations imply are checked too: the rate implied by the mass balance, (q/V)(C_Af - C_A) - dC_A/dt, and the conductance implied by the energy balance once that rate and the known heat of reaction are accounted for. Both should be non-negative over the visited states. This uses the known balances and parameters, not the truth.

### 9.4 Three claims kept apart

* Conservation by construction. The mechanistic and hybrid models close the mass and energy balances of their own states because their equations are those balances. It is a property of the structure, stated once per model, and it does not show that their rate or their conductance is right.
* Predictive error: sections 9.1 to 9.3.
* Mechanism: whether the fitted or learned rate and conductance are the true ones. Only an oracle comparison can say (section 9.5).

### 9.5 Available metrics and oracle diagnostics

Available, computed from exports and the known specification: every metric of sections 9.1 to 9.3, the implied terms, the selection effort, the distance of test points from training points.

Oracle, computed by a separate script on the truth side, outputs under `experiments/`, never fed back into a choice: the errors against the exact states; the error of the predicted peak of each hot excursion against the true peak; the fitted or learned rate against r_true, and the conductance against UA_true(T), over the states the test visits; estimates against the pseudo-true values of section 7.4; OT and OF.

### 9.6 Replicates and summaries

The independent unit is the replicate: one training run, with its excitation and its noise, and the models fitted on each of its budgets. Readings and windows inside a replicate are not independent units. For each model and budget, the score J over a whole test set is computed once per replicate, and the ten values are reported by their median, quartiles and extremes.

Comparisons are paired: the difference between two models on the same replicate, summarised by its median, by the number of replicates in which each model is better, and by a bootstrap interval of the mean difference over replicates. The primary comparisons are named when the benchmark is registered, before any training run: HK against MR and HK against BN, on test-A10 at every budget and on the extrapolation test. A fixed test set is one sample of the P3 distribution; a bootstrap over its windows is reported as a secondary measure of that variability.

### 9.7 The test sets

The test sets are defined, seeds included, in the iteration that generates the training data, and generated only after every configuration, grid and budget has been frozen and committed. They are evaluated once. Nothing is chosen on them: not an architecture, a hyperparameter, a budget, a component, nor which results to show. If a defect of the software is found after the evaluation, it is fixed and recorded, every model is evaluated again, and the log says so.

## 10. Hypotheses (to be registered, not results)

* H1, interpolation and data efficiency: on test-A10, HK scores lower than MR from some budget on, and lower than BN at small budgets; the gap to BN narrows as the budget grows.
* H2, extrapolation: trained at A5, HK loses less from test-A5 to test-A10 than BN does. MR's loss is set by its structural error, which grows away from the nominal point.
* H3, protocol shift: the candidates rank on test-steps as they rank on test-A10.
* H4, physical validity: the black boxes produce physically invalid predictions, or imply negative rates or conductances, in some windows, more often at small budgets and outside the training region. That the structured models produce none is a check of their implementation, not a finding.
* H5, mechanism (oracle, secondary): over the visited states, the rate of HK is closer to the true rate than MR's first-order law is.

Expectations stated before anything is run, argued from the structure and from M0: MR's error should be dominated by its structure rather than by its parameters, so its curve against the budget should be nearly flat; BN starts from nothing; HK starts where MR ends and improves only if the correction can be learned from the budget. Any of H1 to H5 may fail and each failure is a result. HK no better than MR at any budget would say the correction cannot be learned from these data; BN reaching HK at large budgets would say the structure buys data efficiency and not accuracy.

The thresholds, how much lower and in how many replicates, are fixed when the benchmark is registered, before any training run, not here.

## 11. Limitations of the design

* One target plant, one process, one reaction, and two hidden mechanisms chosen in D-004, one of them weak on the target. The conclusions are about this setting.
* The noise is ideal (D-020) and the inputs exact. Real data would bring bias, drift, delays and gaps.
* One family of excitation for training, the corners of P3. Data efficiency depends on the design of the excitation, which is not varied.
* Extrapolation is examined in one direction, amplitude, from a box of half the size. A new operating point belongs to M3.
* Scores against readings contain sigma; the distance to the floor is estimated, not observed.
* Ten replicates and one test set per question: differences smaller than their spread will not be resolved, and the report will say so.
* The grids are small. A learned model might do better with more tuning; the selection effort is reported so that this can be judged.
* The Fisher information of section 7.4 assumes the model is right. The profile and the oracle fit are there because it is not.

## 12. Definition of done for M1

1. The benchmark, its data, budgets, replicates, windows, metrics, grids, primary comparisons and thresholds, is registered in the experiment log before any training run, from a clean commit.
2. MN, MR, BL, BN and HK, with HU and HKU as ablations, are implemented with tests, fitted on every budget and replicate with every failure recorded, and frozen in a commit.
3. The test sets are generated after that commit and evaluated once. The curves of data efficiency, the extrapolation and the protocol shift are reported with their variability, failures and physical violations, whatever they show.
4. The identifiability diagnostic is reported and the treatment of E/R recorded as a decision.
5. The oracle diagnostics are reported apart, from the truth side only.
6. The boundary holds: `models` and `evaluation` never import `simulation` or `generation`, which a test on the import graph enforces; no hidden quantity reaches a model; the scan of the available branch finds nothing in the new data.
7. Tests and ruff pass and CI is green. The experiment log, the decision log and a `docs/m1_audit.md` of the kind of M0's are written. Codex reviews and the owner accepts.

## 13. Iterations

| Iteration | Deliverables | Accepted when |
|---|---|---|
| I0 Design (this one) | this plan; the active milestone in `AGENTS.md`, `CLAUDE.md`, `README.md` and the roadmap; D-029 | the owner has answered section 15 and the plan follows the answers |
| I1 Evaluation contract and mechanistic models, on development data | `evaluation`: windows, phases, initial states, metrics, failures, validity rules, implied terms. `models`: the interface of a continuous-time model, rollout with failures recorded, MN, MR. A test on the import graph for both packages | windows, phases and initial states right on hand-built observations; metrics equal to hand-computed values; integration error of the rollout below 1 % of sigma against a tighter tolerance; MR recovers the parameters of data simulated by the modeller's own model with the noise of D-020, within four of the standard errors its Fisher information predicts; suite and CI pass. A smoke run on `m0-e05` shows that the code runs on a real export; its numbers are not reported |
| I2 Identifiability diagnostic, M1-E01 | parts 1 to 3 of section 7.4; the decision on E/R, if Q1 leaves it to the diagnostic | registered before it is run; run from a clean commit; logged as hypothesis, method, result and interpretation; oracle outputs only under `experiments/` |
| I3 Data | the lead, and the amplitude if Q2 is accepted, in the grammar of identities and in the data contract; the verification of P3 at A5 (M1-E02, registered, as M0-E03b); the definitions of every data set, test sets included; the generation of the training sets only (M1-E03, registered) | every true trajectory accepted; the ten checks of the generator pass; seeds disjoint from every earlier one; no test data exist |
| I4 Mechanistic baselines on the benchmark | MN and MR on every replicate and budget; the variant with E/R fixed, if Q1 asks for it; residuals per balance on the validation parts; OT and OF on the truth side | every fit converged or its failure is recorded; no test set read |
| I5 Framework and learned models | the comparison of two frameworks and the decision; BL, BN, HK, HU, HKU; tests | HK with a zero correction equals MR to the precision of the integrator; gradients agree with finite differences; a correction planted in data simulated by the modeller's model is recovered, without using the truth; tests pass on CI with the new dependency; the cost of one fit is measured |
| I6 Registration and training | the registration of the benchmark (M1-E04), committed before any training run; every fit; the frozen configurations, committed | the registration precedes the runs in the history; every fit logged with its failures and selection effort; no test set read |
| I7 Final evaluation | the test sets generated from their committed definitions; one evaluation of every frozen model; the oracle diagnostics by their own script; results in the log | every window accounted for; every hypothesis reported as it came out |
| I8 Analysis and closure | mechanism, from the oracle; `docs/m1_audit.md`; Codex's review; the owner's acceptance | the definition of done of section 12 |

## 14. The next iteration, recommended

I1. Everything after it depends on it, it needs no scientific decision beyond the evaluation contract itself (windows, initial states, phases, metrics, failures), no new dependency, and it gives the mechanistic baseline a place before any network exists. Concretely:

* `process_transfer/evaluation/`: windows and phases from the known inputs; the rule for initial states; metrics per channel in physical units and in sigmas, per phase; records of failures; the validity rules; the implied terms.
* `process_transfer/models/`: the interface of a continuous-time model; rollout under piecewise-constant inputs, sampled at the sensor instants, returning failures as records; MN and MR on the equations of `modeller/cstr_first_order.py`, parameterised by k_350, E/R and UA.
* A test on the import graph: `models` and `evaluation` never import `simulation` or `generation`. The generic integration code that both sides need is either moved to a neutral module or written again on the model side; which, is decided on inspection in I1 and recorded.
* Models read exports through `data.export.open_export_directory`, the known parameters from the export and the modeller's values from `configs/modeller_cstr.yaml`, never a plant configuration file.

If the answer to Q5 changes the primary task, I1 changes with it. Answers to Q1 to Q4 do not change I1.

## 15. Questions for the owner

**Q1. E/R: fixed or estimated with the other parameters.**
(a) Estimated together with k_350 and UA. Realistic, and the strongest simple baseline, since a free E/R can offset part of the missing physics; its estimate then says as much about the excitation as about the chemistry (section 7.3), and in the hybrid it trades off against the correction, so mechanism is judged on the whole rate. (b) Fixed at 8750 K. This gives MR and the hybrids the exact value of a parameter that the black boxes have to learn from data. An engineer may know an activation energy from laboratory work, but rarely exactly; here the exactness is an artefact of the design. Attribution is cleaner, and MR loses the freedom to offset part of the missing physics, so it is expected to predict worse. (c) Decided by the diagnostic of section 7.4. If the difficulty is bias rather than variance, as the scratch check suggested, the diagnostic will call E/R identifiable and (c) becomes (a) with an extra step that depends on data. Recommendation: (a) for every model, with (b) as a secondary analysis for MR and HK at two budgets, declared in the registration.

**Q2. The extrapolation test.**
(a) Train at half amplitude, A5, and test at A10. Clean and symmetric, entirely inside the validated region at test time; it needs the verification of P3 at A5 before any data (section 6.2) and a token for the amplitude in the identities. (b) Train without the corners where T_c is high and test on them: extrapolation in one direction, towards the hot region that matters for the envelope; needs a restricted variant of P3, safe by the argument of the 256 pairs but new code. (c) The change of protocol only, the single-input steps: no new verification, but a weak notion of extrapolation. Recommendation: (a) as the primary test and (c) as a secondary one.

**Q3. Budgets, replicates and how selection is counted.**
Proposed: b = 2, 5, 10, 20 and 40 excursions (0.4 to 8 h of record, 4 to 80 min at a corner); ten replicates, each a run of 40 excursions whose prefixes are the budgets; validation taken from inside the budget for any method that selects; training at A5 on b = 10 and 40 only. Alternatives: five replicates to halve the cost, which halves the power of the paired comparisons; independent runs for each budget instead of prefixes, which adds noise to the shape of the curves; validation data outside the budget, which would hide the price of tuning. Recommendation: as proposed, with the number of replicates confirmed after the cost of a fit is measured in I5 and before registration.

**Q4. The first hybrid.**
(a) The kinetic correction, HK, as the hybrid of the primary comparisons, with HU and HKU as ablations. (b) The thermal correction first. (c) Both from the start. Recommendation: (a), for the reasons of section 8.2: the rate is constrained by both balances, and the thermal signal on the target is weak, so (b) would probably have little to learn and (c) invites a trade-off that the data may not resolve.

**Q5. The primary predictive task.**
(a) Free rollout over windows of 720 s, an excursion and its rest, from the mean of ten readings at the nominal inputs, with a lead of 60 s at the start of each run; rollout over whole runs as a secondary evaluation; k steps ahead as a diagnostic. (b) Rollout over whole runs as the primary evaluation, which weighs steady-state bias more and dynamics less. (c) One step ahead as the primary evaluation, dominated by noise (section 4.4). Recommendation: (a).

Defaults that stand unless the owner objects: the target only, with a repetition on the source considered only after the target is done; BN and BL as the black boxes; the sigmas of the data sheet as the scale of the metrics; failures reported, never replaced by a value; test sets generated only after the models are frozen; the data of M0 for software development only.
