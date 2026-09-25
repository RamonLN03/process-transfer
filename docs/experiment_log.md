# Experiment log

One entry per experiment or data-generation run, in order. Each entry separates hypothesis, method, result and interpretation, and records the code version, configuration and seeds. Results are not written before they exist.

## M0

M0-E01 to M0-E03b are deterministic verifications of the virtual plants. M0-E04 generates observations in memory, seeded, and checks them. M0-E05 is the first entry to store a data set, `m0-e05`, and it does so to test the data path, not to provide training data.

### M0-E01 Operating points, local stability and envelope (2026-09-18)

Code: `experiments/00_verify_operating_points.py` at commit `06f8927`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`. Deterministic; no random seed involved.

**Hypothesis.** (H1) With the design of D-005 to D-008, source and target each have a unique steady state for the nominal inputs, locally stable with a margin of 0.5 1/min, at the values given by the preliminary scratch calculation. (H2) Under the D-010 excitation amplitudes both plants stay inside the documented envelope, T in [335, 380] K, for every combination of extreme inputs.

**Method.** Steady states by continuation in temperature: the mass balance is solved for C_A at each temperature, the energy residual is scanned for sign changes between 280 and 480 K on a 0.25 K grid, roots are refined with Brent's method and polished on the full system. Jacobian by central differences, validated against hand-derived analytical Jacobians of both models; eigenvalues from the numerical Jacobian. The search was validated on the textbook case (Seborg et al., Example 2.5), where it finds the three known steady states, and against an independent pure-Python implementation. Envelope: 8 single-input and 16 corner input cases, each applied as a step from the nominal steady state and integrated for 40 min (LSODA, rtol = atol = 1e-9, sampled every second); steady states and eigenvalues recomputed for every case.

**Result.**

| | Source | Target |
|---|---|---|
| Steady states for nominal inputs | 1 | 1 |
| C_A | 250.021 mol/m^3 | 189.673 mol/m^3 |
| T | 349.999 K | 355.169 K |
| Eigenvalues | -1.6050 +- 1.3625i 1/min | -0.9636 +- 1.8041i 1/min |
| Input cases with a unique stable steady state | 24 of 24 | 24 of 24 |
| Least stable case, max real part | -0.896 1/min | -0.381 1/min |
| Temperature range over all step responses | 339.67 to 372.44 K | 341.19 to 385.29 K |
| Concentration range | 110 to 440 mol/m^3 | 46 to 392 mol/m^3 |
| Hottest case | q+ C_Af+ T_f+ T_c+ | q+ C_Af+ T_f+ T_c+ |

The nominal values agree with the preliminary ones at their printed precision, and time integration from a perturbed state converges to the same points. Every step response converges to the steady state of its inputs. On the target the peak of 385.29 K is a transient overshoot (final value 372.97 K). Its least stable case, q+ C_Af+ T_f+ T_c-, had not decayed to one part in a million after 40 min; an 8 h integration confirms that it decays and that there is no sustained oscillation.

**Interpretation.** H1 holds. H2 holds for the source and is rejected for the target, which leaves the envelope by 5.3 K. The narrower envelope quoted in the first version of `docs/assumptions.md` was wrong: the scratch calculation simulated four cases and took the low-flow corner to be the hot one, whereas more flow of a richer feed brings more reactant and more heat. At none of the 24 tested input cases is there more than one steady state or a loss of stability, so at those points the violation is a matter of amplitude and weak damping, not of a bifurcation. The 380 K limit is a documented convention, not a property of the fluid. Remedies are evaluated in M0-E02.

Correction, 2026-09-21. The interpretation first read "no multiplicity and no loss of stability appears anywhere on the input box". That extrapolated from 24 points, 8 single-input excursions and 16 corners, to the whole box; the interior and the rest of the boundary were never examined, and the sentence now says so. The experiment also covers single steps from the nominal steady state only. Its conditions are the original D-010 amplitudes (q and C_Af +-20 %) and are kept as they were run; re-running the script at `a3b5eaf`, after the fixes to the sampling grid and to the steady-state scan, reproduces every number above.

### M0-E02 Design options for the target envelope (2026-09-18)

Code: `experiments/01_envelope_design_options.py` at commit `e8ff275`. All variations are applied in memory; no configuration file is changed. Deterministic.

**Hypothesis.** At least one small change brings the target back inside the envelope without removing the mismatch the project needs: (A) smaller amplitudes on q and C_Af, (B) more cooling on both plants, (C) a milder source-target shift.

**Method.** The functions and the 24 input cases of M0-E01, applied to each variation. The visibility of the kinetic mismatch is reported as the range of r_true / r_model = 2 / (1 + K_sat C_A) over the concentrations visited.

**Result.** Target plant:

| Option | Nominal point | Eigenvalues, 1/min | Least stable case, 1/min | T range, K | r_true / r_model | Envelope |
|---|---|---|---|---|---|---|
| Baseline, D-010 | 189.7 mol/m^3, 355.17 K | -0.964 +- 1.804i | -0.381 | 341.19 to 385.29 | 1.69 to 0.78 | violated |
| A, q and C_Af +-15 % | unchanged | unchanged | -0.610 | 341.78 to 380.56 | 1.63 to 0.80 | violated |
| A, q and C_Af +-10 % | unchanged | unchanged | -0.723 | 342.37 to 376.19 | 1.58 to 0.83 | met |
| B, UA_ref x1.5 on both, T_c = 341.667 K | 216.1 mol/m^3, 352.86 K | -1.846 +- 1.709i | -1.176 | 342.78 to 372.49 | 1.42 to 0.76 | met |
| C, target UA_ref = 0.9 x source | 224.0 mol/m^3, 352.19 K | -1.235 +- 1.573i | -0.632 | 340.19 to 382.30 | 1.63 to 0.74 | violated |

The source stays inside the envelope under every option (339.67 to 372.44 K at most). Under B the source keeps its nominal point with eigenvalues -2.651 +- 0.737i 1/min. No option produces multiple steady states in any of the 24 tested input cases.

**Interpretation.** A at +-10 % and B restore the envelope; A at +-15 % misses by 0.6 K and C does not help. A at +-10 % costs little visibility of the kinetic mismatch, because the temperature inputs drive most of the concentration excursion. B gives the strongest damping but changes the accepted plant design and brings the target's operating point closer to the source's. The choice modifies accepted decisions (D-010 or D-005) and is recorded as the open decision D-018.

Outcome and limit, 2026-09-21. The project owner chose A at +-10 % (A10). Every option in this experiment was judged on single steps from the nominal steady state at 24 input cases. M0-E03 shows that this is not enough: with chained input changes A10 takes the target to 395.6 K. The numbers above stand as results about steps from nominal; re-running the script at `a3b5eaf` reproduces them.

### M0-E03 Sequential excitation at the A10 amplitudes (2026-09-21)

Code: `experiments/02_sequential_excitation.py`. Protocols, seeds and acceptance criteria were committed at `58cc633` before any comparison was run; reporting and figures were extended at `0421a53`, with the pre-registered block and the sequence, ramp and verdict logic byte-identical to `58cc633`. The only result known beforehand was the two-stage counterexample found by the reviewer on `f6c5781`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`. Seeds 0 to 19 for every protocol and plant, all reported. Three runs gave identical numbers.

Command: `python experiments/02_sequential_excitation.py`. Run time about 68 s on the development machine (Windows 11, Python 3.13.7): counterexample 1.5 s, reference steps 1.2 s, corner transitions 19 s, seeded sequences 36 s, ramps 0.5 s, cross-checks 7 s, figures 2 s. Four figures and `summary.json` are written to `PT_DATA_DIR/m0_e03`, which git ignores.

**Hypothesis.** A10 may not keep both plants inside [335, 380] K when input changes are chained every 120 s, because the state at each change depends on the history, whereas D-018 was judged on steps from the nominal steady state only.

**Method.** Integration is restarted at every input change, one solver call per segment (LSODA, rtol = atol = 1e-9), sampled every 0.1 s and at every switching instant. A trajectory is accepted when the integrator succeeds, the states are physical (C_A positive and never above the richest feed so far, T never below the coldest stream so far), T stays inside [335, 380] K judged on a parabolic refinement of the largest sample, and the integrated mass and energy balances close to 1e-6 relative, by Simpson quadrature of the physical terms on the stored samples. Nothing is clipped.

Protocols, all on a 120 s clock, fixed before the comparison:

| Key | Protocol | Reason for including it |
|---|---|---|
| P0 | A10, binary levels: every input at -1 or +1 on each tick | the baseline reading of D-010 |
| P1 | thermal inputs +-2.5 K, binary levels | halves the thermal swing that drives the counterexample |
| P2 | A10, three levels, at most one level of change per tick | forbids a cold-to-hot jump in one tick |
| P3 | A10, 120 s excursions to a corner separated by 600 s at nominal | restores the premise of M0-E02 |
| P4 | thermal inputs +-2.5 K, three levels, one level per tick | both mitigations together |

Parts: (A) the counterexample, with three integrators at 1e-12 and three sampling periods; (B) reference steps from the nominal steady state, 24 cases of 40 min, for A10 and for thermal inputs at +-2.5 K; (C) all 240 ordered transitions between corners, the second corner held 600 s, the first either held 120 s from nominal or fully settled; (D) 20 seeded sequences of 2 h per protocol and plant; (E) adversarial ramps, the fastest cold-to-hot route each protocol allows with q and C_Af high, for cold dwells of 120, 240 and 600 s; (F) the worst seeded sequence of every protocol and plant recomputed with DOP853 at 1e-12 and with LSODA sampled every 0.01 s. A protocol passes on the tested cases for a plant only if every seeded sequence, every ramp and, for binary protocols, every corner transition is accepted.

**Result.**

Counterexample, target: the cold stage raises C_A from 189.67 to 347.40 mol/m^3 at 345.14 K; after the change T peaks at 395.6297 K, 42.55 s later, and stays 19.05 s above 380 K. LSODA, DOP853 and Radau at 1e-12 agree to 1e-4 K. The sampled peak is 395.5431 K at 1 s, 395.6288 K at 0.1 s and 395.6297 K at 0.01 s; the refinement at 0.1 s gives 395.6297 K. The reviewer's 395.6288 K is the 0.1 s sampled value. Balances close to 3.7e-9 and 2.7e-9. The source, on the same inputs, peaks at 375.40 K and is accepted.

Reference steps from nominal, none rejected: A10 source 340.42 to 365.75 K, target 342.37 to 376.19 K; thermal inputs +-2.5 K source 344.28 to 358.23 K, target 346.90 to 369.14 K.

Target plant:

| Protocol | Seeded peak, K: min / median / max | Seeds rejected | Worst ramp, K | Corner transitions rejected | Verdict on tested cases |
|---|---|---|---|---|---|
| P0 | 381.51 / 386.77 / 396.57 | 20 of 20 | 396.41 | 17 of 240, worst 395.63 K (120 s dwell) and 396.26 K (settled) | fails |
| P1 | 367.86 / 372.69 / 378.06 | 0 | 377.31 | 0 of 240, worst 377.31 K | passes |
| P2 | 366.98 / 372.52 / 376.99 | 0 | 372.56 | not applicable | passes |
| P3 | 364.93 / 376.19 / 376.19 | 0 | 376.19 | not applicable | passes |
| P4 | 362.60 / 365.71 / 369.14 | 0 | 366.71 | not applicable | passes |

Source plant: every protocol passes on the tested cases. Largest peaks: P0 376.34 K, P1 361.01 K, P2 366.69 K, P3 365.75 K, P4 358.65 K.

In all 17 rejected transitions T_c rises from its low to its high level; T_f rises in 8 of them (9 in the settled variant) and 11 end with q and C_Af high.

Range excited by each protocol, central 90 % of the samples pooled over seeds, with the mismatch it exposes (r_true / r_model = 2 / (1 + K_sat C_A), and UA(T) / UA_ref):

| Protocol | Target C_A, mol/m^3 | Target r_true / r_model | Target T, K | Target UA / UA_ref | Source r_true / r_model |
|---|---|---|---|---|---|
| P0 | 86 to 326 | 1.49 to 0.87 | 343.2 to 369.5 | 0.986 to 1.039 | 1.28 to 0.82 |
| P1 | 121 to 266 | 1.35 to 0.97 | 347.9 to 364.7 | 0.996 to 1.029 | 1.15 to 0.87 |
| P2 | 107 to 305 | 1.40 to 0.90 | 344.9 to 365.5 | 0.990 to 1.031 | 1.23 to 0.83 |
| P3 | 127 to 254 | 1.33 to 0.99 | 348.7 to 361.7 | 0.997 to 1.023 | 1.16 to 0.91 |
| P4 | 138 to 248 | 1.29 to 1.00 | 349.5 to 362.0 | 0.999 to 1.024 | 1.12 to 0.89 |

Every trajectory of the experiment integrated successfully, kept physical states and closed both balances (worst residual 4.1e-9, among the seeded sequences). Every rejection is caused by the temperature envelope alone. In the cross-checks the three computations of each worst sequence differ by at most 1.3e-5 K.

**Interpretation.** The hypothesis holds for the target. With A10 and binary levels every one of the 20 sequences leaves the envelope, by 1.5 to 16.6 K; the source stays inside in every tested case. The behaviour is consistent with the mechanism proposed by the reviewer: a cold stage stores reactant, and raising the thermal inputs releases its heat at once. Every rejected transition raises the coolant temperature across its whole range. Steps from the nominal steady state are therefore not a sufficient test of an excitation, and the basis on which A10 was chosen in D-018 was too narrow.

P1 to P4 pass on the tested cases. That is evidence about 20 sequences of 2 h per protocol and plant, a few ramps and, for the binary protocols, all two-stage corner transitions. It is not a guarantee over the sequences a protocol can generate, and two observations limit what may be inferred. First, designed worst cases were not worst cases: for P1 the worst seeded peak, 378.06 K, is 0.75 K above the worst of all 240 two-stage transitions, and for P2 the worst seeded peak, 376.99 K, is 4.4 K above its designed ramp. Histories longer than two stages matter. Second, the margins to the limit are 1.9 K for P1, 3.0 K for P2, 3.8 K for P3 and 10.9 K for P4, the first two of the same size as the history effects just described.

P3 differs in kind from the others. After 600 s at nominal, about ten time constants of the slowest mode, each excursion starts practically at the nominal steady state, so its responses reduce to the 16 corner steps, which are enumerated exhaustively; its worst peak equals the step reference, 376.19 K. This is a structural argument in addition to the tested sequences, not a proof: it assumes that the rest is long enough and it covers corner excursions only.

Safety has a price in information. P1 and P4 narrow the excited range; on the target P4 exposes the kinetic mismatch only between 1.29 and 1.00. P3 keeps the A10 amplitudes but spends five sixths of the time at nominal. P2 excites almost as widely as P0.

No protocol, amplitude, seed or criterion was changed after seeing results. The physical design and the 380 K limit were not touched. The choice of a protocol modifies D-010 and is recorded as the open decision D-019.

Limitations. Twenty seeds of 2 h sample a very large set of sequences thinly. Transitions were scanned exhaustively only between corners and only for two stages. The three-level protocols have no exhaustive scan. Only a 120 s clock was examined. The results hold for the present source and target; a different target, as in the domain-shift study of M3, would need the same verification.

Re-run with corrected validators, 2026-09-21. Two acceptance checks used by this experiment were found defective in the review of `50dc509` and corrected: the refined peak looked at one segment only (`41695aa`), and the balance residuals were divided by a signed scale (`e9ccfb2`). Further argument checks were added to the simulation modules (`docs/numerical_robustness.md`). The experiment was run again at `852fee0`, from a clean working tree. Every peak, every count of rejected seeds, ramps and transitions, and the whole verdict table are identical to the run recorded above. No acceptance changed. Two things differ, neither of them a result. The relative balance residuals are about half of what they were, because they are now compared with the traffic through each balance, feed plus reaction plus heat exchange, instead of with the reaction term alone: 1.9e-9 and 1.3e-9 on the counterexample (3.7e-9 and 2.7e-9 before), and at most 1.9e-9 among the seeded sequences (4.1e-9 before); the tolerance of 1e-6 was not touched. And the seed reported as the worst for P3 on the source is 1 instead of 17: several seeds share the same peak, 365.7505 K, and the tie is broken in the last digits. From this run on, every run writes to its own directory `PT_DATA_DIR/experiments/m0_e03/<run id>` with a provenance block (commit and state of the working tree, fingerprints and copies of the configurations, protocols, seeds, criteria, integrator settings, environment); earlier runs overwrote `PT_DATA_DIR/m0_e03`. Run time 70 to 75 s.

### M0-E03b Recovery and carried state under protocol P3 (2026-09-21)

Code: `experiments/03_p3_recovery_and_pairs.py`; the protocol and its tolerances are defined in `src/process_transfer/simulation/protocols.py`. Results below are from the run at `852fee0`, clean working tree, identified in its provenance block; two earlier runs gave identical numbers. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`. Deterministic; no random seed. Command: `python experiments/03_p3_recovery_and_pairs.py`, about 15 s (recovery 0.5 s, pairs 13 s). Outputs go to `PT_DATA_DIR/experiments/m0_e03b/<run id>`.

Process note. The script was meant to be committed before its first run, as M0-E03 was. A lint error stopped that commit, and the sweep ran once from the uncommitted file; its provenance block says so (`f458185-dirty`). The tolerances and the logic committed afterwards (`f768f90`) are the ones that ran. That the tolerances preceded the results is therefore supported by the file and by their derivation, not by the history. The reviewer's numbers for both checks were also known beforehand.

**Hypothesis.** Under P3 (A10 amplitudes, 120 s at a corner, 600 s at the nominal inputs) every excursion starts, for all practical purposes, at the nominal steady state, even though the state is never reset; so the worst temperature of a P3 sequence is the worst of the 16 corner steps from nominal, 365.75 K on the source and 376.19 K on the target.

**Method.** No reset anywhere: each segment starts from the final state of the previous one. (1) Recovery: for each of the 16 corners, an excursion from the nominal steady state and the rest that follows; the distance to the nominal steady state at the end of the rest. (2) Carried state: all 256 ordered pairs of corners, repeats included; excursion a, rest, excursion b, rest. For each pair the acceptance checks of `simulation/checks.py` on the whole trajectory, its peak, and the difference between the peak of b inside the pair and the peak of b started from the exact steady state. Integration and sampling as in M0-E03. Tolerances, derived from quantities that do not depend on the results: for the recovery 0.005 K and 0.038 mol/m^3, one hundredth of the planned sensor noise, taking for C_A the stricter of the two readings open in D-020; for the agreement of peaks 0.05 K, a tenth of sigma_T and about 1 % of the 3.8 K between the worst step and the limit.

**Result.**

| | Source | Target |
|---|---|---|
| Largest residual after the rest, C_A | 2.199e-5 mol/m^3 | 6.505e-3 mol/m^3 |
| Largest residual after the rest, T | 7.308e-7 K | 1.333e-3 K |
| Corner leaving the largest residual | ++-- | ++-- |
| Decay of the slowest mode at nominal over 600 s | 1.07e-7 | 6.53e-5 |
| Within the recovery tolerance | yes | yes, by a factor of 3.8 in T and 5.8 in C_A |
| Pairs rejected, of 256 | 0 | 0 |
| Largest peak over the pairs | 365.7505 K, ++-- then ++++ | 376.1895 K, ++-- then ++++ |
| Largest change of a peak caused by the carried state | 1.04e-6 K | 8.95e-4 K |
| Within the 0.05 K agreement | yes | yes |

The reviewer's independent figures, 2.20e-5 mol/m^3 and 7.31e-7 K on the source, 0.006505 mol/m^3 and 0.001333 K on the target, and peaks of 365.7505 K and 376.1895 K, are reproduced. The residuals agree with the linear decay of the slowest mode: deviations of order 100 mol/m^3 and 20 K at the end of an excursion, multiplied by 6.5e-5, give what is observed on the target. On the target the envelope of the decay crosses the temperature tolerance after roughly 480 to 500 s of rest.

**Interpretation.** The hypothesis holds for these plants and conditions. After the rest the target is within 1.3 mK and 0.0065 mol/m^3 of its nominal steady state, the source closer by two orders of magnitude, and carrying that residual into the next excursion changes its peak by less than a millikelvin. The largest peak over all 256 pairs is the peak of the hottest corner started from the exact steady state. Because each rest reduces what is carried by a factor of about 6.5e-5 on the target, residuals do not build up from one excursion to the next; that is an argument, supported by the 20 seeded P3 sequences of ten excursions in M0-E03, not a separate test of long histories.

This is evidence about the present source and target, the A10 amplitudes, corner excursions of 120 s and a rest of 600 s. It is not a guarantee for another plant, amplitude, hold or rest, and a different target, as in M3, needs the same verification. The rest was measured, not optimised: on the target 600 s meets the temperature tolerance by a factor of about four, so a rest much shorter than eight minutes would not.

Note on the tolerances, 2026-09-21. D-020 has since been accepted with sigma_CA = 5 mol/m^3 on both plants. The recovery tolerance for C_A stays at 0.038 mol/m^3, the value that was verified; it is now 1/132 of sigma_CA instead of 1/100 and was not relaxed. The script and `simulation/protocols.py` said that a residual a hundred times below the sensor noise "cannot be told from an exact restart in any data generated later". That was too strong and has been corrected in both places. It is a practical criterion, not a statistical guarantee: averaging N independent readings resolves an offset of about 2 sigma / sqrt(N), so a constant offset of sigma / 100 would show after some 40 000 readings, 67 h at one reading every 6 s. No number, tolerance or verdict of this entry changes.

### M0-E04 Reproducible observations of P3 on source and target (2026-09-21)

Code: `experiments/04_observations_from_p3.py`, with the reusable parts in `src/process_transfer/measurement/` and `src/process_transfer/simulation/operating_run.py`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`, `configs/sensors_cstr.yaml`. Command: `python experiments/04_observations_from_p3.py`. Outputs go to `PT_DATA_DIR/experiments/m0_e04/<run id>`.

Registration. The definition below, and the block "Fixed before the first run" of the script, were committed before the script was run with the registered seeds. Known at that point: the P3 sequence of excitation seed 0, ten excursions, was accepted on both plants in M0-E03, so H1 was expected to hold; the unit tests of the measurement code had been run, with seeds 7, 8, 11, 12 and others, none of them a seed of this experiment; and the script had been exercised once outside the repository on a shorter run, two excursions with seeds 900, 901 and 902, to debug the plotting code. The registered realisation had not been generated. Seeds are not replaced after seeing a result, and a criterion that fails is reported as failing.

**Hypothesis.** (H1) The true P3 trajectories are accepted by `simulation/checks.py` at 0.1 s resolution on both plants. (H2) The observations form one row every 6 s from 0 to 7200 s, 1201 rows, strictly increasing and without duplicates; every row is a stored sample of the true trajectory; every change of the inputs falls on a row, and that row carries the new inputs. (H3) Reading minus exact state behaves as independent zero-mean Gaussian noise with sigma_CA = 5 mol/m^3 and sigma_T = 0.5 K, the same on both plants, uncorrelated between the two sensors, between the two plants, between sensor seeds and with the exact states. (H4) The same configuration and seeds reproduce the numerical content exactly, and another sensor seed changes every reading while leaving the corners, the true trajectory, the instants and the inputs exactly as they were.

**Method.** Plants: source and target. Protocol: P3 as accepted in D-019 and defined in `simulation/protocols.py`, A10 amplitudes, 120 s at a corner, 600 s at the nominal inputs, ten excursions, 2 h per run, the first adaptation budget of D-011. The state is carried from segment to segment and never reset. Both plants receive the same input sequence, since they share nominal inputs (D-008) and the excitation seed.

Two kinds of randomness, kept apart. The corners come from `numpy.random.default_rng(excitation seed)`, excitation seed 0, which is the P3 sequence of seed 0 in M0-E03. The sensor noise comes from `numpy.random.SeedSequence(entropy=sensor seed, spawn_key=(plant index, run index, channel))`, with plant index 0 for the source and 1 for the target, run index 0, and channel 0 for C_A and 1 for T. No global generator is used, and the two generators share nothing.

| Scenario | Excitation seed | Sensor seed | Purpose |
|---|---|---|---|
| A | 0 | 2026 | the reference observations |
| A-again | 0 | 2026 | A generated a second time, from nothing but configuration and seeds |
| B | 0 | 2027 | the same excitation read with another realisation of the noise |

Sampling and numerical settings. The truth is integrated with LSODA, rtol = atol = 1e-9, restarted at every input change, and stored every 0.1 s: 72 001 samples per run, on which peaks and balances are judged. The sensors read every 0.1 min = 6 s (D-020, `configs/sensors_cstr.yaml`): every sixtieth true sample, never an interpolated value. A true trajectory that is not accepted is not observed. The criteria for true states, physical bounds and closed balances, are not applied to readings, and readings are neither clipped nor corrected.

Acceptance of H3. Each statistic is a z-score, its distance from what correct noise would give in units of the standard deviation that correct noise gives it over n = 1201 readings (`measurement/noise_statistics.py`). For each of the 8 error series of scenarios A and B (2 scenarios, 2 plants, 2 variables): mean, spread about zero, lag-one correlation, and the fractions of errors beyond one and beyond two sigma; 40 scores. Correlations, 24 scores: C_A with T within a plant (4), each variable of the source with each of the target (8), the same series under the two sensor seeds (4), and each error series with the exact state it was added to (8). The criterion is |z| <= 4 for all 64. For correct noise a single score exceeds 4 with probability 6.3e-5, so a correct sensor fails the criterion with a probability of about 0.4 %. Expected variation of one series of 1201 readings, one standard deviation: the mean within 0.029 sigma (0.144 mol/m^3, 0.0144 K) and the spread within 0.020 sigma. What the limit resolves: a standard deviation wrong by 8 % or more, a bias of 0.12 sigma or more, a correlation of 0.12 or more. The reading of D-010 that was not chosen, 3.8 mol/m^3 on the target, would score about -12.

Acceptance of H1, H2 and H4 is exact: acceptance by `check_trajectory`; equality of instants, row counts and switching instants; and bit-for-bit equality of arrays, also recorded as SHA-256 digests of the content of each observation set.

Diagnostics: the inputs and the true response of both plants; exact states against readings over the first 360 s of scenario A; histograms of the standardised errors and all 64 z-scores. The last two use exact states and errors and are marked as diagnostics. No array is written to disk, since the storage layer does not exist yet.

**Result.** First run with the registered seeds: run `20260921T204823Z_2a912ac`, from the commit that registered the definition, clean working tree, identified in its provenance block. Run time 3.2 s on the development machine (Windows 11, Python 3.13.7): generation of the six runs 0.8 s, checks under 0.1 s, figures 2.0 s. All four hypotheses hold.

H1. Both true trajectories are accepted.

| | Source | Target |
|---|---|---|
| True samples, every 0.1 s | 72 001 | 72 001 |
| Temperature, K | 340.42 to 365.7505 | 342.37 to 376.1884 |
| C_A, mol/m^3 | 136.0 to 347.1 | 67.0 to 279.6 |
| Balance residuals, mass and energy | 2.8e-10, 6.8e-10 | 4.2e-10, 4.6e-10 |

Corners of excitation seed 0, in order: `+++-`, `----`, `-+++`, `++++`, `++++`, `-++-`, `-++-`, `+++-`, `-+-+`, `----`. The hottest corner comes up twice in a row.

H2. On both plants: 1201 rows, at 0, 6, ..., 7200 s exactly, strictly increasing; every row is a stored true sample, one in every 60; 19 changes of the inputs, all of them on rows, at the 19 switching instants, each row carrying the new inputs. The inputs are identical on the two plants. The true peak of the target, 376.1884 K at t = 2926.9 s, falls between the readings at 2922 and 2928 s; the largest exact temperature on the 6 s grid is 0.038 K lower (0.051 K on the source). That is why the truth keeps its own sampling.

H3. All 64 z-scores are within +-4. The largest is 2.31, the mean of the C_A errors of the target in scenario A; for 64 independent standard normal scores the largest would typically be about 2.5.

| Series, n = 1201 | Mean | z | Root mean square | z | Lag-one | z | Beyond 1 sigma | Beyond 2 sigma | Largest error |
|---|---|---|---|---|---|---|---|---|---|
| A, source, C_A, mol/m^3 | +0.083 | +0.58 | 5.195 | +1.92 | +0.020 | +0.68 | 0.336 | 0.054 | 3.02 sigma |
| A, source, T, K | -0.0257 | -1.78 | 0.4970 | -0.28 | -0.029 | -1.02 | 0.301 | 0.049 | 3.28 sigma |
| A, target, C_A, mol/m^3 | +0.334 | +2.31 | 5.152 | +1.50 | -0.059 | -2.05 | 0.336 | 0.052 | 3.23 sigma |
| A, target, T, K | -0.0128 | -0.88 | 0.5063 | +0.63 | -0.011 | -0.36 | 0.331 | 0.047 | 3.37 sigma |
| B, source, C_A, mol/m^3 | +0.170 | +1.18 | 4.831 | -1.65 | +0.022 | +0.77 | 0.308 | 0.037 | 3.69 sigma |
| B, source, T, K | +0.0098 | +0.68 | 0.5038 | +0.38 | +0.040 | +1.39 | 0.329 | 0.051 | 3.35 sigma |
| B, target, C_A, mol/m^3 | +0.101 | +0.70 | 5.036 | +0.37 | +0.011 | +0.38 | 0.323 | 0.054 | 3.45 sigma |
| B, target, T, K | +0.0247 | +1.71 | 0.5150 | +1.48 | -0.061 | -2.10 | 0.330 | 0.051 | 4.18 sigma |

Expected for correct noise: mean 0 within 0.144 mol/m^3 or 0.0144 K, root mean square 5 mol/m^3 or 0.5 K within 2 %, fractions 0.317 and 0.0455. Of the 24 correlations the largest are C_A with T on the target in scenario A, r = -0.058 (z = -2.01), the T errors of the target under the two sensor seeds, r = -0.054 (z = -1.89), and the T errors of the source with the exact temperature, r = -0.051 (z = -1.77). The target carries 5.15 and 5.04 mol/m^3, not the 3.8 mol/m^3 of the reading that was not chosen. One reading of the 9608 lies 4.18 sigma from the truth, and about a third lie beyond one sigma: sigma is not a bound, and nothing was clipped.

H4. Scenario A generated a second time gives the same instants, readings, inputs and true states, bit for bit, and the same digests (source `ee056623981d139a...`, target `2d658c5e43e41c80...`). With sensor seed 2027 the true states, the instants and the inputs are bit-identical to those of A, and all 1201 rows of readings differ on both plants (digests `c77d9c611fca1113...` and `4aead82c02b7a187...`). Digests identify content on one machine and one set of library versions; the last bits of a simulated state may differ elsewhere.

Amounts. Six runs were generated and none stored: per run 1201 rows of 7 numbers, 8407 numbers or 67 kB as 64-bit floats, against 72 001 true samples of 2 states. `Observations` holds eleven fields, listed in `summary.json`, none of them an exact state, an error, a parameter or a seed.

**Interpretation.** For these two plants, this protocol and these seeds, the observations are what D-020 specifies: the same absolute noise on source and target, in SI units, independent between sensors, plants, samples and seeds, and independent of the states it is added to. They are reproducible from the configuration and the two seeds alone, and the two kinds of randomness do not touch: another sensor seed changes every reading and nothing else. The grid is the sensor's, the truth keeps the resolution on which it is validated, and the record of the inputs is exact because every change falls on a row.

Two observations outside the registered criteria, both made after the run and recorded as such.

The scores lean to positive values: their mean is +0.25 and their root mean square 1.15, and the means by family are +0.56 for the mean, +0.54 for the spread and -0.29 for the lag-one score, each over 8 series with a standard error of 0.35. The scores are not independent, since the spread and the two tail fractions of a series move together, so no single test applies to the 64. To see whether the sensors or the statistics are biased, the same statistics were computed through `measure` for seeds 0 to 999, two streams and two channels, 4000 series: every family has a mean within 0.02 of zero, with a standard error of 0.016, and a standard deviation between 0.99 and 1.02. A mean spread score of +0.54 or more over eight series happens in 5 % of such sets. The lean of this realisation is chance. `tests/test_sensors.py` now holds that calibration, on 250 seeds that nobody chose.

The ten excursions are a longer history than the pairs of M0-E03b. Every excursion of the target starts within 5.4e-3 mol/m^3 and 6.8e-4 K of the nominal steady state, against tolerances of 0.038 mol/m^3 and 0.005 K, and its peak differs from that of the same excursion started exactly at the steady state by at most 4.1e-4 K, against 0.05 K; on the source the figures are 1.5e-5 mol/m^3, 6.1e-7 K and 7.0e-7 K. The second of the two consecutive `++++` excursions is no worse than the first. This is one sequence, not a study of histories; it is now a regression test in `tests/test_p3_protocol.py`, with the tolerances unchanged.

Limitations. Two runs of 2 h per plant and one excitation sequence: the statistics say that these sensors behave as specified, not how much data a model will need. The noise is ideal by decision, Gaussian, white, unbiased and without delay or gaps, and the inputs carry no error; none of that is a finding. A limit of 4 over 1201 readings cannot see a standard deviation wrong by less than 8 % or a correlation below 0.12. Noise streams are keyed by plant index and run index, and two runs given the same seed and stream would share their noise exactly: nothing yet prevents that mistake when many runs are generated, and the storage layer will need a rule for it. No data set has been stored: the observations exist in memory and are identified by their digests.

Second run, 2026-09-21. Run `20260921T205347Z_32f0009`, clean working tree, after the tests and the documentation of this entry were committed: the same digests, the same 64 scores and the same verdicts. Run time 4.2 s.

Note on the digests, 2026-09-21. The digests above were computed with the first encoding of the content, which did not hash structure and is replaced in `aea49db` by the versioned encoding `observations/v1` (`docs/data_contract.md`). The strings change and the content does not: in run `20260921T212519Z_aea49db`, clean working tree, every noise statistic and correlation of the entry is identical to those of the registered run, the verdicts are the same, and scenario A generated a second time still equals A. The digests are now `a8508ef8a442ba5f...` and `bab51b85b4b95ba6...` for scenario A on source and target, and `15a2ce99bc6e86b2...` and `5ddba1f793ad3b3d...` for scenario B. The same run confirms that the fixes of `51e8d45` to noise keys and sensor units leave M0-E04 unchanged.

### Re-runs of M0-E01 to M0-E03b after the structural check (2026-09-21)

Since `875e83a` every `Trajectory` is checked when it is built: instants strictly increasing, segments sharing their switching instant, the state continuous across it, the comparison exact. The check runs on every trajectory the earlier experiments construct, and an exact comparison that refused a legitimate trajectory would stop them, which only running them can show. Each was therefore run once more, from a clean working tree. M0-E03b at `9603ade`: every number of its entry is reproduced, 0 of 256 pairs rejected, 365.7505 K and 376.1895 K. M0-E01 and M0-E02 at `32f0009`: the printed results are those of the runs at `852fee0`, character for character for M0-E02. M0-E03 at `32f0009`, run `20260921T205407Z_32f0009`, 90 s: all 921 numbers and verdicts of its `summary.json` outside the provenance and timing blocks are identical to those of the run at `852fee0`. No legitimate trajectory is refused, and no result changed.

### M0-E05 The full data path, from configuration files to an export (2026-09-22)

Code: `experiments/05_full_data_path.py`; the reusable parts are `src/process_transfer/generation/` and `src/process_transfer/data/`, and the SQL is under `sql/`. Definition of the data set: `configs/datasets/m0_e05.yaml`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`, `configs/sensors_cstr.yaml`. Command: `python experiments/05_full_data_path.py`; the data set alone is generated by `python -m process_transfer.generation configs/datasets/m0_e05.yaml`.

Registration. The definition below, the definition file and the block "Fixed before the first run" of the script were committed before the script was run on that definition. Known at that point: the true P3 sequences of excitation seeds 0, 1 and 2 were accepted on both plants in M0-E03, so H1 was expected to hold; seed 0 was observed in M0-E04, under other noise streams; the unit tests of the data path had been run, on small data sets with excitation seeds 41 to 45 and other master seeds; and the pipeline and this script had each been exercised once outside the repository, on definitions of one and three excursions with excitation seeds 901, 902, 911 and 912, to debug them. The noise streams of this data set follow from its run identities under master seed 20260922 and had not been drawn. No seed or tolerance is changed after a result, and a criterion that fails is reported as failing.

What this experiment is and is not. It is a test of software: whether the path from the simulator to an export keeps the data intact and the truth out. A failure of H1 to H8 would be a defect of the code or of the contract, to be corrected; it would not be a scientific result about the plants. H9 alone says something about the generated noise. Nothing here says that six runs of two hours are enough to train a model, and this data set is not a benchmark for M1.

**Hypothesis.** (H1) The six true trajectories are accepted, from verified starting points. (H2) The observations are identical before and after Parquet, after reconstruction from DuckDB, and in the exported aligned series: arrays bit for bit, and content hashes. (H3) Instants and input changes are preserved: every run has the 20 input settings of its protocol, the corner or the nominal inputs of each of its 20 segments, at the instants of its protocol, and the row of a change carries the new inputs. (H4) The SQL quality queries find nothing in the valid data set, find every defect put on purpose into a separate copy of a run, and do not report valid oddities. (H5) A second generation reproduces the content: the same hashes, and not a file of the available branch is touched. (H6) Ingesting again changes nothing and is reported as such. (H7) The data set can be read, ingested and exported from a place that holds nothing but the data set itself. (H8) The available branch holds no hidden information in files, tables, metadata or exports, and the scan that says so finds a leak when one is planted in a copy. (H9) The noise of the six runs is as specified in D-020 and independent between plants and between runs.

**Method.** Plants: source and target. Protocol P3 as accepted in D-019 and sensors as accepted in D-020, both unchanged. Three excitation seeds, 0, 1 and 2, the same on both plants, so the two plants receive the same three input sequences; ten excursions per run, 2 h, the state never reset; six runs, 12 h of process time in all. The truth is integrated with LSODA, rtol = atol = 1e-9, restarted at every input change and stored every 0.1 s; the sensors read every 6 s, 1201 rows per run. Runs are processed one after another, so one dense trajectory is in memory at a time.

Randomness. The corners come from `numpy.random.default_rng(excitation seed)`. The noise of a run comes from `SeedSequence(entropy=20260922, spawn_key=(w0, w1, w2, w3, channel))`, where the four words are the first 16 bytes of the SHA-256 of the identity of the run, such as `target.p3.e0.x10.n0`, and the channel is the index of the state (D-021). The identity names the plant, so the noise of the two plants is independent although their inputs are the same. The master seed and the streams are recorded in the private branch only.

Steps: (1) the pipeline, `generation/pipeline.py`, with its ten mandatory checks; (2) the pipeline a second time on the same definition; (3) a reader in another process whose `PT_DATA_DIR` holds only a copy of the data set, which opens it, ingests it into a new database, runs the quality queries and exports it; (4) ten defects, each put into its own copy of the run `target.p3.e0.x10.n0` in the staging schema and rolled back, with the quality query that must report each, and three valid oddities that must not be reported: a negative concentration reading, a reading of 1e6 K, and an instant one bit off its clock; (5) on the truth side, the runs generated again to obtain the exact values, and the stored readings compared with them; (6) a copy of the export with a seed planted in its manifest, and another with an exact state planted as a column, given to the scan.

Acceptance. H1 to H8 are exact: acceptance by `check_trajectory` and by `load_virtual_plant`; bit-for-bit equality of arrays and equality of content hashes; equality of instants and input values; no row returned by a quality query, or the expected query among those that return rows; equal modification times before and after the second generation; no finding of the scan, and at least one on each planted copy. H9 uses the z-scores of `measurement/noise_statistics.py` with the limit of 4 used in M0-E04: for each of the 12 error series of 1201 errors, mean, spread, lag-one and two tail fractions, 60 scores; and 42 correlations, each error series with the exact value it was added to (12), C_A with T within a run (6), each variable of the source with each of the target for each excitation seed (12), and the same variable between two runs of a plant (12). For correct noise at least one of the 102 exceeds 4 with a probability of about 0.6 %.

Expected costs, from two smaller runs outside the repository, stated before the run so that a surprise is visible. Rows: 2 plants, 16 parameters, 12 channels, 6 runs and 43 236 measurements, 6 x 1201 x 6. Disk: about 0.3 to 0.6 MB for the data set, about the same for the export, and 5 to 12 MB for the database, most of it fixed overhead of DuckDB. Time: 2 to 5 s to generate, observe and write, 5 to 20 s to ingest, because the whole database is checked after every run, and under a minute for the whole script.

Expected artefacts. `PT_DATA_DIR/available/datasets/m0-e05/` with `manifest.json`, four small Parquet tables and six files under `measurements/`; `PT_DATA_DIR/available/databases/m0-e05.duckdb`; `PT_DATA_DIR/available/exports/m0-e05/` with `export.json` and six Parquet files; one directory per generation under `PT_DATA_DIR/private/datasets/m0-e05/` with `provenance.json`, `pipeline_report.json` and copies of the configurations; and `PT_DATA_DIR/experiments/m0_e05/<run id>/` with `summary.json`, `readings_C_A.png`, `readings_T.png` and copies of the configurations. The exit code is 0 only if every hypothesis holds.

**Result.** First run on the registered definition: run `20260921T221211Z_fa0378c`, from the commit that registered it, clean working tree, identified in its provenance block. The data set `m0-e05` did not exist before it. All nine hypotheses hold, and the exit code was 0.

| | Expected before the run | Found |
|---|---|---|
| Rows: plants, parameters, channels, runs, measurements | 2, 16, 12, 6, 43 236 | 2, 16, 12, 6, 43 236 |
| Data set on disk | 0.3 to 0.6 MB | 0.345 MB, 11 files |
| Export on disk | about the same | 0.247 MB, 7 files, a little under the range |
| Database on disk | 5 to 12 MB | 6.83 MB |
| Generate, observe and write Parquet | 2 to 5 s | 1.73 s |
| Ingest into DuckDB | 5 to 20 s | 1.52 s, well under: the expectation was pessimistic |
| Whole script | under a minute | 16.2 s |

Stages of the first generation, in seconds, on the development machine (Windows 11, Python 3.13.7, pyarrow 25.0.1, duckdb 1.5.5): configurations 0.004; plants and starting points 0.11; generate, validate, observe and write 1.73; read Parquet back 0.14; ingest 1.52; SQL quality checks 0.24; export 0.95; scan for hidden information 0.31; 5.2 in all. The second generation took 3.8 s, the reader without the private branch 3.3 s, the planted defects 1.0 s and the noise diagnostics 1.1 s.

H1. Both starting points verified, the source at -1.605 and the target at -0.964 1/min against a margin of -0.5, balances closed there to a relative 1e-15; six true trajectories accepted. H2. For all six runs the observations read back from Parquet, those rebuilt from DuckDB and those rebuilt from the exported files equal the generated ones bit for bit, with the same content hashes: `9006d4e1bf9ff9d0...`, `aba65cd6c2817a8a...` and `6d9b8c99a852e794...` on the source, and `c45d6a0957b867e0...`, `9e561bfc61983740...` and `37a3b227eec190ae...` on the target, for excitation seeds 0, 1 and 2. H3. In every run the instants at which the stored inputs change, and the inputs from then on, are the 20 settings of the protocol exactly. H4. The seven quality queries return no row on the database. In the copies of `target.p3.e0.x10.n0`, the untouched copy and the three valid oddities give no finding, and each of the ten defects is reported by the query that exists for it; four of them are also reported by another query, a duplicate by the row count of the cadence check for instance. H5. The second generation gives the same six hashes, reports the data set and the export as already present, and the modification time of no file of the available data sets and exports changed. H6. Every run is reported as already present and the row counts are unchanged. H7. In another process, with a data directory holding only a copy of the data set, the six runs were ingested into a new database, the quality queries found nothing, and the export has the same six hashes; no private directory existed there. H8. The scan read 19 files, the 11 of the data set, the database and the 7 of the export, and found nothing; on a copy of the export with a seed in the manifest it reported 2 findings, with an exact state as a column 2 findings, and none on the untouched copy.

H9. All 102 z-scores are within +-4. The largest is 2.58, the spread of the T errors of `target.p3.e2.x10.n0`, whose root mean square is 0.4737 K for a specified 0.5 K. The mean of the scores is -0.11 and their root mean square 0.96. Over the 12 series the root mean square of the C_A errors runs from 4.896 to 5.167 mol/m^3 and that of the T errors from 0.4737 to 0.4984 K; the target carries 4.90, 5.07 and 5.17 mol/m^3, the absolute noise of D-020 and not 3.8 mol/m^3. One reading of 14 412 lies 4.21 sigma from the truth.

**Interpretation.** For this data set, the path keeps the data intact and the truth out. What the sensors gave is what Parquet holds, what the database returns and what the export delivers, to the last bit; identities, instants and input changes survive; a repetition changes nothing and says so; and a reader needs nothing but the data set. The quality queries are neither blind nor trigger-happy on the cases put to them. The noise drawn from streams derived from run identities behaves as specified and is independent between plants that receive the same inputs, which is what D-021 was meant to give.

These are results about software, as the registration said. They are evidence for the cases exercised: six runs, one protocol, one process, ten defects and two kinds of planted leak, on one machine. They are not a proof that no defect can pass the quality queries or that no leak can pass the scan, which compares names and numbers and would not see a hidden value that had been scaled or rounded. Nothing here says that the data are sufficient for any model.

Two things found while preparing the experiment, before its registration, and corrected then. A defect that exists only between plants, one variable in two units, could not be seen by the quality gate of an ingestion, which looks at one run of one plant; the database as a whole is now checked inside the transaction (`b409f14`). And one of the defects first chosen for H4, a concentration channel in mol/L, turned out to be of that kind and was replaced before registration by one that a single run can show.

Limitations. The run time of ingestion grows with the square of the number of runs, because the whole database is checked after each. The content hashes are those of one machine and one set of library versions. `aligned_series` names the six variables of the CSTR. The data set lives by default under the repository, in a synchronised folder; `PT_DATA_DIR` exists to move it.

### Re-runs at the end of the iteration of 2026-09-22

Selection. Between `7bc3e6a` and `a58afe8` the only module of `simulation/` that changed is `operating_run.py`, together with `measurement/` and the new `sampling_clock.py`, `data/` and `generation/`. Of the experiments, M0-E04 and M0-E05 use them; M0-E01, M0-E02, M0-E03 and M0-E03b use `integration`, `checks`, `balances`, `steady_state`, `envelope`, `excitation` and `protocols`, none of which was touched, so they were not run again.

M0-E04, run `20260921T221811Z_a58afe8`, clean working tree: the four hypotheses hold, the largest of the 64 scores is 2.31 as in the registered run, and the digests are those recorded under `observations/v1`, `a8508ef8a442ba5f...` and `bab51b85b4b95ba6...`.

M0-E05, run `20260921T221817Z_a58afe8`, clean working tree: the nine hypotheses hold and the exit code is 0. The data set `m0-e05` was already there, so this run exercised the path of repetition from its first step: the generator reported the data set, the ingestion and the export as already present, touched no file of the available branch, and gave the six content hashes of the registered run. Run time 14.4 s.

### Re-run of M0-E05 after the three fixes of the review (2026-09-22)

Selection. The three defects reported by the reviewer of `404ae79` were fixed in `b4b1715`, `633b17e` and `968d226`, in `data/export`, `data/parquet_store` and `generation/leak_scan`, and recorded in `b754d14`. Of the experiments, M0-E05 alone uses those modules; M0-E01 to M0-E04 use nothing that changed and were not run again.

**Method.** The generation command and `experiments/05_full_data_path.py`, both unchanged, from the clean commit `b754d14`, in the environment of the registered run: Python 3.13.7, numpy 2.5.3, scipy 1.18.1, pyarrow 25.0.1, duckdb 1.5.5. The data set `m0-e05` of the registered run was on disk, so the path of repetition was exercised from the first step.

**Result.** Generation command: exit code 0; the data set, the ingestion and the export reported as already present; the ten checks passed; 4.2 s. Experiment: run `20260922T092559Z_b754d14`, clean working tree, exit code 0; the nine hypotheses hold; the six content hashes are those of the registered run, in the pipeline, in the reader without the private branch and in the export; the scan read 19 files and found nothing, and reported 2 findings on each planted copy; the largest of the 102 z-scores is 2.58, as registered. Run time 15.7 s. The database file has grown from 6.83 to 9.45 MB over the repeated ingestions that added no row, which is how DuckDB manages its file and not a change of content; the row counts are those of the registered run.

Two attempts refused before that, and why. The same command and script were first run from the same commit under another interpreter of the same machine, with numpy 2.3.5 and scipy 1.16.3 instead of 2.5.3 and 1.18.1. Both stopped at the publication of the data set with a conflict: the six runs generated under those libraries have other content hashes, all six of them, and a published data set is never modified. Nothing on disk was touched. The two attempts are recorded under `private/datasets/m0-e05/` as `20260922T092410Z_b754d14` and `20260922T092415Z_b754d14`, each with its environment. Generated afresh in a scratch directory, the same definition passes the ten checks under those libraries, with its own hashes.

**Interpretation.** The three fixes change nothing of the content or of the verdicts of M0-E05 in the registered environment, which is what a fix of a reader, of an exporter and of a scan must do. The refused attempts are the limitation stated when the digest was defined, seen in practice: a content hash identifies the numbers of one environment, and the same definition under other numerical libraries is other content, bit for bit. They are also a test the policy on repetition passed without being asked: another environment could not replace the registered data set by accident. Continuous integration installs current libraries and will therefore not reproduce the registered hashes; it tests the software, not the numbers of this data set.

### M0-E06 Steady operation with noise only (2026-09-22)

Code: `experiments/06_steady_operation.py`; the reusable parts are `simulation/protocols.py` (`steady_segments`), `data/identifiers.py` (`steady_run_identifier`) and the generator of M0-E05. Definition of the data set: `configs/datasets/m0_e06.yaml`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`, `configs/sensors_cstr.yaml`. Command: `python experiments/06_steady_operation.py`; the data set alone is generated by `python -m process_transfer.generation configs/datasets/m0_e06.yaml`.

Registration. The definition file, the block "Fixed before the first run" of the script and this entry were committed before the script was run on that definition. Known at that point: the nominal steady states and their stability (M0-E01, verified again on every generation); the noise under P3 (M0-E04, M0-E05); the data path on short steady runs in the unit tests, 120 s with another master seed; and one smoke run of this script outside the repository, 10 min per plant with master seed 1, to debug it. The two noise streams of `m0-e06` follow from the identities `source.steady.d7200.n0` and `target.steady.d7200.n0` under master seed 20260923 and had not been drawn. No seed or tolerance is changed after a result, and a criterion that fails is reported as failing.

What this experiment is and is not. It completes the first of the two operating runs of D-010 that were never generated. It is a test of the simulator and of the data path under no excitation, and of the generated noise on top of a constant state. The noise is a hypothesis of the simulator (D-020): nothing here says that a real sensor behaves like it. The run excites nothing, so it says nothing about the dynamics of the plants.

**Hypothesis.** (H1) Started at the nominal steady state, the true states stay there: over two hours, max |x(t) - x(0)| is within 1e-6 of the scale of each state on each plant, 2.5e-4 mol/m^3 and 3.5e-4 K, the balances close and the trajectory is accepted. (H2) The ten mandatory checks of the generator pass. (H3) Each stored run has 1201 rows at exactly 6 s, and every stored input row equals the nominal inputs of its plant exactly. (H4) Every z-score of the noise diagnostics is within +-4: 26 scores, mean, spread, lag-one and two tail fractions of the four error series (20), the correlation of the C_A errors with the T errors within a run (2), and each variable of the source with each of the target (4). (H5) The scan of the available branch finds nothing, the export holds exactly the ten columns of the contract, no description names a seed, and the private record holds the master seed and the streams.

**Method.** Plants: source and target. Protocol: steady operation, the nominal inputs held for 7200 s from the verified nominal steady state, one run per plant; nothing is drawn at random for the inputs and no excitation seed exists. Sensors of D-020, unchanged: readings every 6 s, sigma_CA = 5 mol/m^3, sigma_T = 0.5 K; the noise stream of each run follows from its identity under one private master seed, so the two plants have independent noise. The truth is integrated with LSODA, rtol = atol = 1e-9, and stored every 0.1 s. Steps: the two trajectories on the truth side, with the drift and the acceptance checks; the generator on the definition, with its ten checks; the stored data read back, for cadence, constancy of the inputs and the columns of the export; the runs generated again on the truth side for their exact values, and the stored readings compared with them.

Drift tolerance, fixed from the scales and the numerical precision. The integrator keeps the local error of a step below 1e-9 of the state; at a steady state stable with the margin of D-017 that error is damped by the following steps, so the global error is of the order of a few local errors, not of their sum over 72 000 steps. A margin of one thousand over the local tolerance gives 1e-6 of each state: 2.5e-4 mol/m^3 and 3.5e-4 K, which is 1/20 000 of sigma_CA, 1/1400 of sigma_T, and about 1/150 and 1/14 of the recovery tolerances of P3. The residual of the polished root, below 1e-9 of the feed terms and in practice near 1e-16 of them, can move the state by at most residual / |lambda|, about 5e-7 mol/m^3 even at the allowed limit. Exact equality of the integrated states with the root is not asked for.

The correlation of an error series with the exact state it was added to, one of the diagnostics of M0-E04 and M0-E05, is not computed here: the exact state of a steady run is constant by design, H1 bounds its variation to a millionth of its scale, and the correlation with a constant series is undefined. `noise_statistics.correlation` refuses such a series; it is not given a correlation of zero and no epsilon is added. What the mean score of each error series checks instead is that the readings average to the steady state within the noise.

Expected costs, from the smoke run and from M0-E05. Rows: 2 plants, 16 parameters, 12 channels, 2 runs, 14 412 measurements. Disk: about 0.1 to 0.2 MB for the data set and for the export, 5 to 12 MB for the database. Time: under a minute for the whole script. Expected results, stated before the run: a drift of the order of 1e-8 mol/m^3 and 1e-9 K over 10 min in the smoke run, so of that order or somewhat more over two hours, far inside the tolerance; all 26 scores within +-4, which correct noise fails with a probability of about 0.16 %.

Expected artefacts. `PT_DATA_DIR/available/datasets/m0-e06/`, `PT_DATA_DIR/available/databases/m0-e06.duckdb`, `PT_DATA_DIR/available/exports/m0-e06/`, one directory per generation under `PT_DATA_DIR/private/datasets/m0-e06/`, and `PT_DATA_DIR/experiments/m0_e06/<run id>/` with `summary.json`, `fig1_drift_of_the_true_states.png`, `readings_C_A.png`, `readings_T.png` and copies of the configurations. The exit code is 0 only if every hypothesis holds.

**Result.** First run on the registered definition: run `20260922T130915Z_2f7efd3`, from the commit that registered it, clean working tree, identified in its provenance block. The data set `m0-e06` did not exist before it. All five hypotheses hold and the exit code was 0. Run time 4.6 s in all, of which the generator 2.2 s.

| | Expected before the run | Found |
|---|---|---|
| Rows: plants, parameters, channels, runs, measurements | 2, 16, 12, 2, 14 412 | 2, 16, 12, 2, 14 412 |
| Data set, export, database on disk | 0.1 to 0.2 MB, the same, 5 to 12 MB | 0.122 MB, 0.086 MB, 5.26 MB |
| Drift of the true states over two hours | of the order of 1e-8 mol/m^3 and 1e-9 K | source 4.3e-8 mol/m^3 and 1.2e-9 K; target 1.3e-8 mol/m^3 and 2.1e-9 K |
| Largest of the 26 z-scores | within 4 | 1.83 |

H1. Both trajectories are accepted, 55 and 54 evaluations of the right-hand side for 72 001 samples, the integrator taking long steps on a state that does not move; the balances close to a relative 1e-11. The largest deviation from the starting state is 4.3e-8 mol/m^3 and 2.1e-9 K, five and six orders of magnitude inside the tolerances of 2.5e-4, 1.9e-4 and 3.5e-4, and the final state is within 6e-13 mol/m^3 and 2e-13 K of the start: the excursion is a transient of the first steps, not a drift that grows. The residual of the right-hand side at the start is below 1e-14 in both balances. H2. The ten checks of the generator pass; the seven quality queries return no row; the content hashes are `56c1a4c2bf17d859...` for the source and `685eb8024c972a5d...` for the target. H3. Both runs have 1201 rows, every instant is on the sensor clock and every stored input row equals the nominal inputs of its plant exactly. H4. All 26 scores are within +-4; the largest, 1.83, is the correlation of the C_A errors with the T errors on the source; the mean of the scores is 0.05 and their root mean square 0.94. Over the four series the root mean square of the errors is 4.93 and 5.10 mol/m^3 against a specified 5, and 0.501 and 0.497 K against 0.5; the largest error is 3.6 sigma. On the sensor grid the exact states move by at most 4.3e-8 mol/m^3 and 2.1e-9 K, which is why their correlation with the errors was not computed. H5. The scan read 11 files and found nothing; each export holds exactly the ten columns; no description names a seed; the private record holds the master seed and both streams.

**Interpretation.** Under no excitation the simulator stays where its verified steady state puts it, to within the precision that the tolerance was derived from, and the data path keeps that constant plus the specified noise intact through Parquet, DuckDB and the export. The readings of the two plants are compatible with independent Gaussian noise of the specified levels on constant states, for these two realisations, as the 26 scores were fixed to test. That is a statement about the simulator and its data path: the noise is the hypothesis D-020, and nothing here says that a real analyser or thermometer behaves like it. The run excites nothing and therefore tells nothing about the dynamics of the plants or about what a model could learn from it; it is one of the two operating runs that D-010 promised, now generated and verified, and not a training set.

Limitations. Two runs, two noise streams, one duration. The drift tolerance is a bound derived for this integrator at these tolerances; another integrator or another rtol needs its own derivation. The diagnostics of the noise omit, on purpose, the correlation with the exact state, so a dependence of the noise on a constant state could not be seen here; it is checked under P3 (M0-E04, M0-E05), where the state moves.

### M0-E07 Single-input step tests with ten-minute holds (2026-09-22)

Code: `experiments/07_single_input_steps.py`; the reusable parts are `simulation/protocols.py` (`single_step_segments`), `data/identifiers.py` (`step_run_identifier`) and the generator of M0-E05. Definition of the data set: `configs/datasets/m0_e07.yaml`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`, `configs/sensors_cstr.yaml`. Command: `python experiments/07_single_input_steps.py`; the data set alone is generated by `python -m process_transfer.generation configs/datasets/m0_e07.yaml`, which verifies each trajectory before observing it but does not write the diagnostics of this experiment.

Registration. The definition file, the block "Fixed before the first run" of the script and this entry were committed before the script was run on that definition. Known at that point: single-input steps at the A10 amplitudes from the nominal steady state stayed inside the envelope for 40 min in M0-E02, which is how A10 was chosen (D-018); the return from the stepped state to the nominal inputs had never been simulated; the data path on short step tests in the unit tests, 60 s segments with another master seed; and one smoke run of this script outside the repository with segments of 240 s and master seed 2, to debug it, whose numbers are the expectations below. The sixteen noise streams of `m0-e07` follow from the identities under master seed 20260924 and had not been drawn. No seed or tolerance is changed after a result, and a criterion that fails is reported as failing.

What this experiment is and is not. It completes the second of the two operating runs of D-010 that were never generated. Its first part is physical: whether a hold of 600 s at one input moved by its A10 amplitude, and the return from it, keep both plants inside the envelope, which M0-E02 examined for the step only. Its second part is the data path, run only if the first part accepts all sixteen trajectories. The eight tests of a plant are independent runs from the nominal steady state, each carrying its state across its own three segments; they are not one trajectory of eight steps, because the transitions between tests have not been studied, and the safety of P3 is not extrapolated to these holds.

**Hypothesis.** (H1) All sixteen true trajectories are accepted: finite, physical, inside [335, 380] K over lead, hold and recovery, balances closed. (H2) The ten mandatory checks of the generator pass. (H3) In every stored run the inputs change exactly at 600 s and at 1200 s, to the setting of the test and back to nominal, and nowhere else; each run has 301 rows on the 6 s clock. (H4) Every z-score of the noise diagnostics is within +-4: 240 scores, mean, spread, lag-one and two tails of the 32 error series (160), each error series with the exact state it was added to (32), C_A with T within a run (16), and for each test each variable of the source with each of the target (32). (H5) The scan of the available branch finds nothing and no description names a seed.

Reported and not hypotheses, because no rule of sign or size is imposed on them: the refined peak and the minimum temperature of every test with their times and margins to the envelope; the initial response, the derivative of each state at the switching instant under the stepped inputs, and the direction of its first move; the extreme of each state during the hold, its time, and whether it lies inside the hold, a non-monotone response, or at its end; the state at the end of the hold against the steady state of the stepped inputs and the slowest time constant there; the extreme of the return; and the recovery, |x(1800 s) - x_nominal|, set against the recovery tolerances of P3 for reference only.

**Method.** Plants: source and target. Protocol: for each of q, C_Af, T_f and T_c and each direction, 600 s at the nominal inputs, 600 s with that input moved by its A10 amplitude (+-10 % for q and C_Af, +-5 K for T_f and T_c) while the other three stay nominal, 600 s at the nominal inputs; sixteen runs of 1800 s from the verified nominal steady state, nothing drawn at random for the inputs. Sensors of D-020, unchanged; the noise stream of each run follows from its identity under one private master seed. The truth is integrated with LSODA, rtol = atol = 1e-9, restarted at each input change and stored every 0.1 s. Steps: (1) the sixteen trajectories on the truth side with the acceptance checks of `simulation/checks.py` and the diagnostics above, and one figure per plant of the true responses; the script stops here, with the diagnostics written and the exit code 1, if any trajectory is not accepted; (2) the generator on the definition; (3) the stored data read back for cadence and switching instants; (4) the runs generated again on the truth side for their exact values, and the noise diagnostics; (5) the readings of the export drawn.

Expected costs and results, from the smoke run with segments of 240 s, stated before the run. Rows: 2 plants, 16 parameters, 12 channels, 16 runs, 28 896 measurements. Time: about 30 s. Acceptance of all sixteen is expected: in the smoke run the hottest test was T_c up, 359.3 K on the source and 366.1 K on the target, with the peak about 50 s after the step, so a longer hold should not raise it; the coldest was T_c down, 342.7 and 345.9 K; the return from the cold T_c state overshot to 358.8 K on the target. With holds of 600 s the return starts closer to the stepped steady state, whose slowest time constant is 30 to 80 s, so the overshoots of the return may differ slightly from those of the smoke run; nothing suggests a violation, but that is what part 1 checks. Non-monotone responses are expected: on the target the temperature overshoots under every input, and C_A overshoots under q and C_Af. For correct noise at least one of the 240 scores exceeds 4 with a probability of about 1.5 %.

Expected artefacts. `PT_DATA_DIR/available/datasets/m0-e07/`, `PT_DATA_DIR/available/databases/m0-e07.duckdb`, `PT_DATA_DIR/available/exports/m0-e07/`, one directory per generation under `PT_DATA_DIR/private/datasets/m0-e07/`, and `PT_DATA_DIR/experiments/m0_e07/<run id>/` with `summary.json`, `fig1_true_responses_source.png`, `fig1_true_responses_target.png`, `readings_C_A.png`, `readings_T.png` and copies of the configurations. The exit code is 0 only if every hypothesis holds; if part 1 rejects a trajectory, only the summary and the figures of the truth are written.

**Result.** First run on the registered definition: run `20260922T131708Z_7e04034`, from the commit that registered it, clean working tree, identified in its provenance block. The data set `m0-e07` did not exist before it. All five hypotheses hold and the exit code was 0. Run time 14.9 s: the sixteen trajectories and their diagnostics 1.2 s, the generator 7.0 s, of which the ingestion 2.7 s and the export 1.9 s, the stored data and the noise 1.0 s.

| | Expected before the run | Found |
|---|---|---|
| Accepted trajectories | 16 of 16 | 16 of 16 |
| Hottest test, source and target | T_c up, about 359.3 and 366.1 K | T_c up, 359.30 K at 654 s and 366.11 K at 650 s |
| Coldest test, source and target | T_c down, about 342.7 and 345.9 K | T_c down, 342.71 K at 674 s and 345.89 K at 684 s |
| Hottest return, target | about 358.8 K after T_c down | 358.72 K at 1278 s after T_c down; 362.03 K at the return from T_c up is the lowest point of that recovery seen from above, not a peak |
| Rows: plants, parameters, channels, runs, measurements | 2, 16, 12, 16, 28 896 | the same |
| Largest of the 240 z-scores | within 4 | 3.40 |

H1. Every trajectory is accepted; the balances close to a relative 1e-9 or better. Margins to the envelope: on the source the peak is 20.7 K below 380 K (T_c up) and the minimum 7.7 K above 335 K (T_c down); on the target 13.9 K and 10.9 K. Every peak of the hold occurs 50 to 140 s after the step, so the length of the hold does not raise it, as expected. The return from a hold overshoots on both plants and stays well inside: on the target the return from T_c down reaches 358.72 K, and the return from T_c up dips to 353.25 K; on the source the returns move by at most 7 K. H2. The ten checks pass; the seven quality queries return no row; the scan read 39 files and found nothing. H3. Every run has 301 rows on the sensor clock, its inputs change at 600.0 and 1200.0 s exactly, to the setting of the test and back to the nominal inputs, and nowhere else. H4. All 240 scores are within +-4; the largest, 3.40, is the two-sigma tail fraction of the T errors of the source under T_c down; the mean of the scores is -0.01 and their root mean square 1.02; over the 32 series the root mean square of the errors runs from 4.70 to 5.31 mol/m^3 and from 0.453 to 0.537 K, and the largest error is 3.8 sigma. H5. Nothing hidden; no description names a seed.

Response to each input, from the diagnostics of the truth. The direction of the first move after the step and the extreme during the hold:

| Test | Source: first move of C_A, T; extreme of T during the hold | Target: first move of C_A, T; extreme of T during the hold |
|---|---|---|
| q up | +, +; 350.57 K at 737 s, 0.01 K above the value at the end of the hold | +, -; T first falls to 355.15 K then rises 1.2 K above it |
| q down | -, -; 349.36 K, mirror of q up | -, +; T first rises then falls, mirror of q up |
| C_Af up | +, +; 351.29 K at 735 s, 0.05 K above the end | +, +; 358.45 K at 702 s, 0.79 K above the end |
| C_Af down | -, -; 348.85 K | -, -; 352.76 K, 0.25 K below the end |
| T_f up | -, +; 352.00 K at 663 s, 0.33 K above the end | -, +; 358.35 K at 663 s, 0.97 K above the end |
| T_f down | +, -; 348.08 K, 0.21 K below the end | +, -; 352.13 K, 0.64 K below the end |
| T_c up | -, +; 359.30 K at 654 s, 2.5 K above the end | -, +; 366.11 K at 650 s, 4.1 K above the end |
| T_c down | +, -; 342.71 K at 674 s, 0.39 K below the end | +, -; 345.89 K at 684 s, 0.95 K below the end |

Every extreme of T lies inside the hold, not at its end: every temperature response is non-monotone, on both plants, and on the target the response to a change of flow is of two signs, the temperature moving first against the direction it settles in. The concentration overshoots inside the hold in fifteen of the sixteen tests; under T_c down on the source it approaches its stepped value from below without overshoot. The interaction of reaction and cooling produces these shapes, and no rule of sign was imposed on them. At the end of the hold each plant is within 2.7e-5 mol/m^3 and 1.6e-6 K of the steady state of the stepped inputs on the source, and within 1.7e-2 mol/m^3 and 1.9e-3 K on the target, whose slowest time constants at the stepped points are 47 to 83 s against 31 to 43 s on the source; a hold of 600 s is seven to nineteen of those constants. Each stepped input has one steady state in the scanned range.

Recovery after the final 600 s. On the source the residual is at most 1.5e-5 mol/m^3 and 2.8e-7 K; on the target 1.9e-3 mol/m^3 and 8.1e-4 K, the largest after T_c down. All sixteen are inside the recovery tolerances of P3, 0.038 mol/m^3 and 0.005 K, by factors of 20 and 6 on the target at the worst; this is reported against those tolerances for reference, since they were derived for P3 and the recovery of these tests was not assumed to match it.

**Interpretation.** The second operating run of D-010 is generated and verified. Holds of 600 s at one input moved by its A10 amplitude, and the return from them, keep both plants inside the envelope with margins of at least 7.7 K, for these plants, amplitudes and durations, which is evidence about the sixteen cases simulated and not a guarantee for chained tests, other amplitudes or other plants. The responses are non-monotone in temperature everywhere and of two signs under flow changes on the target, which any model of these plants will have to reproduce and any reading of "the sign of the response" must take into account. The stored data are the responses plus the noise of D-020, kept intact through the path, and the noise behaves as specified for these sixteen realisations; it is a hypothesis of the simulator, not a property of real sensors.

Limitations. Sixteen tests, one hold length, one lead and one recovery; the tests are independent and say nothing about chaining them. The response diagnostics are read off samples every 0.1 s and the refined peak is an estimate, not a bound. The data set is a verification of the path and of the protocol, not a training set for M1.

### M0-E08 The effect of the temperature-dependent conductance alone, an oracle diagnostic (2026-09-22)

Code: `experiments/08_oracle_conductance.py`; it reuses `simulation/protocols.py` for the P3 sequences, `simulation/integration.py`, `simulation/checks.py` and `simulation/cstr_true.py`. Configuration: `configs/source_cstr.yaml`, `configs/target_cstr.yaml`, unchanged. Command: `python experiments/08_oracle_conductance.py`. Outputs under `PT_DATA_DIR/experiments/m0_e08/<run id>` only; nothing of this experiment goes to the available branch.

Registration. The script, with its block "Fixed before the first run", and this entry were committed before the script was run on the registered cases. Known at that point: the ranges of UA(T)/UA_ref under P3 from M0-E03, 0.972 to 1.032 on the source and 0.997 to 1.023 on the target; the three P3 sequences of seeds 0, 1 and 2, accepted in M0-E03 and observed in M0-E05; and one smoke run of this script outside the repository with one excursion of seed 900 on both plants, whose numbers are the expectations below.

What this experiment is and is not. It is an oracle: it uses the hidden physics, and it compares two variants of the truth with everything else known and no noise. It answers how much the temperature dependence of the conductance contributes to the trajectories under the P3 sequences of M0-E05. It does not say whether that contribution can be separated from re-estimated parameters, such as a constant UA fitted to data, or from the noise; that is a question of identifiability and belongs to M1. The comparison with the sigmas of D-020 is a diagnostic of scale, not a detection threshold, and it needs no noise to be computed.

**Hypothesis.** (H1) Anchoring: for each plant the conductance of variant B equals UA_true(T_nominal) exactly, and the right-hand side of B at the nominal state equals that of A bit for bit, so the starting point is a steady state of both variants, closed to the residual of the plant. (H2) Numerical resolution: for every state, plant and sequence, the integration error estimate of each variant, the largest distance between its integration at rtol = atol = 1e-9 and at 1e-11 over the dense grid, is at most 1 % of the largest |B - A| of that state on that sequence. A case that fails is reported as not resolved and its differences are not interpreted.

Reported without a criterion: whether B stays inside the envelope, with its peak and the seconds above the limit if any, a variant that leaves it being kept and never mixed with accepted data; the range of UA_A(T)/UA_B over each trajectory; the differences B - A of C_A and T on the dense grid, largest absolute value and root mean square, over the whole run and over the excursions and the rests separately; the same on the sensor grid in units of the sigmas of D-020; and the fraction of the time at which B is hotter than A.

**Method.** For each plant, variant A is the true plant and variant B has the same true kinetics and a constant conductance equal to UA_A(T_nominal): the parameters of the plant with `ua_ref` replaced by that value and `alpha` by zero, built locally in the script; the official configuration files are not touched. The anchoring at T_nominal keeps the energy balance closed at the starting point, so the variants differ only by the temperature dependence and not by a different conductance at the start; alpha = 0 with UA_ref kept would move the nominal point of the target, whose T_nominal differs from T_ref. Both variants are integrated from the same nominal steady state under the three P3 sequences of M0-E05, seeds 0, 1 and 2 with ten excursions, 2 h each, with the integration of the project, LSODA, rtol = atol = 1e-9, restarted at each input change and stored every 0.1 s, and once more at rtol = atol = 1e-11 to estimate the integration error. Excursions and rests are told apart by the segment structure of P3; the sensor grid is every 6 s. alpha_source = 0.005 1/K and alpha_target = 0.002 1/K are those of the plants and are not changed.

Expected before the run, from the smoke run with one excursion: on the source a largest temperature difference of about 0.46 K and a largest concentration difference of about 3.9 mol/m^3, on the target about 0.37 K and 3.5 mol/m^3, that is below one sigma of each sensor on the sensor grid; root mean squares of a third of those values; differences during the rests as large as during the excursions, because the rest carries the transient that the excursion started; B inside the envelope; an integration error near 1e-7, so H2 by a margin of about 1e4. With ten excursions the largest differences may be somewhat larger than with one, since the corners visited are more. Cost: about 10 s.

Expected artefacts. `PT_DATA_DIR/experiments/m0_e08/<run id>/` with `summary.json`, `fig1_differences_source.png`, `fig1_differences_target.png` and copies of the two plant configurations. The exit code is 0 only if H1 and H2 hold for every case.

**Result.** First run on the registered cases: run `20260922T132213Z_6578f45`, from the commit that registered it, clean working tree, identified in its provenance block. H1 and H2 hold on every case and the exit code was 0. Run time 5.7 s for the 24 integrations.

H1. On the source UA_B = 1666.660 W/K, the conductance of the true plant at its nominal 349.9992 K, against UA_ref = 1666.667 W/K at T_ref = 350 K; on the target UA_B = 1347.116 W/K at 355.169 K, against UA_ref = 1333.333 W/K, so anchoring at T_ref instead would have moved the target's nominal point by the 1 % that separates the two. In both plants the conductances are equal bit for bit at T_nominal, the right-hand sides of A and B at the nominal state are equal bit for bit, and their residual is below 1e-14 in both balances. H2. The integration error estimate, at most 9.4e-7 K and 7.2e-6 mol/m^3 over any case, is between 2e-7 and 8e-7 of the largest difference of the same state, four orders of magnitude inside the criterion of 1 %.

The effect, B minus A. It is not what was expected: the expectation of a few tenths of a kelvin was taken from the ranges of UA(T)/UA_ref of M0-E03, which were computed over the central 90 % of the samples and left out the peaks of the excursions, where the conductance of A departs most from its nominal value. Over the whole trajectories UA_A(T)/UA_B runs from 0.952 to 1.079 on the source and from 0.975 to 1.042 on the target.

| | Source, seeds 0, 1, 2 | Target, seeds 0, 1, 2 |
|---|---|---|
| Largest |T(B) - T(A)|, K | 4.40 in every sequence, during an excursion | 1.79 in every sequence, during an excursion |
| Largest |T(B) - T(A)| during the rests, K | 0.65 | 0.73, 1.02, 0.75 |
| Root mean square of T(B) - T(A) over the run, K | 0.50, 0.40, 0.42 | 0.25, 0.23, 0.23 |
| The same over the excursions and over the rests, K | 1.19, 0.96, 0.99 and 0.11, 0.10, 0.10 | 0.53, 0.44, 0.45 and 0.15, 0.16, 0.15 |
| Largest |C_A(B) - C_A(A)|, mol/m^3 | 33.7 in every sequence | 8.8 |
| Root mean square of C_A(B) - C_A(A) over the run, mol/m^3 | 4.49, 3.56, 3.69 | 1.85, 1.72, 1.67 |
| On the sensor grid, largest |difference| in sigmas of D-020 | 8.8 sigma_T, 6.7 sigma_CA | 3.5 sigma_T, 1.8 sigma_CA |
| On the sensor grid, root mean square in sigmas, whole run | 0.99, 0.80, 0.83 sigma_T; 0.90, 0.71, 0.74 sigma_CA | 0.51, 0.46, 0.45 sigma_T; 0.37, 0.34, 0.33 sigma_CA |
| The same, excursions and rests | 2.4, 1.9, 2.0 and 0.23, 0.22, 0.22 sigma_T | 1.05, 0.87, 0.90 and 0.30, 0.32, 0.30 sigma_T |
| Peak temperature of A and of B, K | 365.75 and 370.12 | 376.19 and 377.97 |
| Fraction of the time with B hotter than A | 0.47, 0.46, 0.48 | 0.47, 0.47, 0.49 |

The same largest values in the three sequences of a plant are those of one corner, the hottest, which every sequence visits: there the temperature of A rises by about 16 K on the source and 21 K on the target, the conductance of A rises by 8 % and 4 %, and B, without that extra cooling, peaks 4.4 K and 1.8 K higher. The effect has a sign that follows the excursion: B is hotter and poorer in A during the hot excursions, colder and richer during the cold ones, and it decays during the rests to a few tenths of a kelvin and a few mol/m^3 within the first minutes, as the figures show. B stays inside the envelope on both plants; on the target its peak, 377.97 K, is 2.0 K below the limit, against 3.8 K for the true plant: the temperature dependence of the conductance is part of what keeps the target inside the envelope under P3.

**Interpretation.** Under the P3 sequences of M0-E05, and with everything else known, the temperature dependence of the conductance changes the trajectories by up to 4.4 K and 34 mol/m^3 on the source and 1.8 K and 8.8 mol/m^3 on the target, several sigmas of the sensors of D-020 at the peaks of the hot excursions, and by about one sigma in root mean square over a run on the source and half a sigma on the target, most of it in the excursions, which take 17 % of the time. The alpha of each plant, 0.005 and 0.002 1/K, is therefore not a negligible part of the physics under this excitation, and the source, with the larger alpha, is where it shows most. This is a statement about the size of the effect when the rest of the physics is known exactly. It says nothing about whether a modeller who re-estimates a constant UA, and sees the readings through the noise, could separate this effect from the rest, which is the identifiability question of M1; nor does it say that a difference of several sigmas at a peak is detectable, since detection depends on how many readings carry it and on what else is being fitted. The comparison with the sigmas is a scale, not a threshold. The values of alpha are kept as they are for this version.

Limitations. Three sequences of one protocol, two plants, one excitation amplitude. The effect under other excitations, the steady run and the single-input steps in particular, was not computed. The two variants are both truths; neither is the modeller's model, and the difference between them is not a model error.

### M0-E08, correction of the verdict and re-run (2026-09-22, D-027)

**Hypothesis.** The script's exit code and its printed verdicts were meant to speak for the physical and numerical validity of every case, not only for the anchoring (H1) and the numerical resolution (H2) of the comparison. Codex, reviewing `915bb42`, reproduced a gap: `check_trajectory` already computed `A.accepted` and the individual physical checks of B, but `main()` never read them into the verdict that decides the exit code. Substituting `check_trajectory`'s return so that `balances_close=False` on both variants left `A.accepted` and `B.accepted` false while H1 and H2, the only things the exit code depended on, still held.

**Method.** `TrajectoryCheck.physically_valid` (`simulation/checks.py`) is the criterion independent of the temperature envelope: finite values, physical states, closed balances, without requiring `inside_envelope`. `comparison_is_valid(primary, secondary)` requires the primary (A) fully `accepted` and the secondary (B) `physically_valid`; B leaving the envelope stays a reported diagnostic, never a rejection by itself, which is the property M0-E08's variant B needs, since it exists to probe how far the physics moves the plant. `experiments/08_oracle_conductance.py` now aggregates a third verdict, `H3_valid`, from `comparison_is_valid` on every case into the same `all(verdicts.values())` the exit code is drawn from, and tags an invalid case in its printed line. Nothing of the physics, the limits, the tolerances, alpha, the noise or the excitation was touched. Regression tests in `tests/test_checks.py` cover `physically_valid` ignoring the envelope on the real M0-E03 counterexample trajectory, each of its three other criteria rejecting alone, `comparison_is_valid` rejecting an unaccepted primary and a secondary with open balances, accepting a secondary that only leaves the envelope, and an invalid case flipping the same `all(...)` aggregate the exit code is drawn from.

**Result.** Fix committed at `b6dccd0` (716 tests plus 9 new, 725, ruff clean); the reference-environment work of D-028 committed at `352c9aa`. Re-run from the clean commit `352c9aa`: run `20260922T152009Z_352c9aa`, exit code 0. H1_anchoring, H2_resolved and H3_valid all hold, on every one of the six cases (`case["valid"]` true in every one, `A.accepted` and `B.physically_valid` both true). Every number reported for the effect, B minus A, in the table above is reproduced exactly: source largest |dT| 4.40 K, |dC_A| 33.7 mol/m^3, peaks 365.75 K and 370.12 K; target 1.79 K, 8.8 mol/m^3, peaks 376.19 K and 377.97 K.

**Interpretation.** The verdict now speaks for what it always reported alongside it: a case can only be H3-valid, and therefore reported as science, when A is fully accepted and B is physically valid regardless of its own envelope. Because the registered cases were already accepted and physically valid, in fact, before the fix, the correction changes what the exit code and the verdicts can prove, not any number of the registered run: M0-E08's characterisation of the effect of UA(T), 4.4 K and 34 mol/m^3 on the source and 1.8 K and 8.8 mol/m^3 on the target, stands. Whether that effect can be separated from a re-estimated constant UA and from the noise remains an identifiability question reserved for M1; this correction is about the mandatory validity of a comparison, not about what the comparison can be used to conclude. alpha_source = 0.005 1/K and alpha_target = 0.002 1/K are unchanged. M0 remains a candidate for closure, not closed by this fix; see `docs/m0_audit.md`.

### Re-run of M0-E05 after the two other protocols were added (2026-09-22)

Selection. Between `4bc056c` and `6578f45` the generator, the data set definitions and the identifiers changed to admit steady operation and single-input steps (`9aaf178`), and M0-E05 uses all of them; M0-E01 to M0-E04 use none of what changed and were not run again. M0-E06 and M0-E07 were run for the first time in this interval and are their own evidence.

**Method.** `experiments/05_full_data_path.py`, unchanged, from the clean commit `6578f45`, in the environment of the registered run, with the data set `m0-e05` of the registered run on disk.

**Result.** Run `20260922T132222Z_6578f45`, exit code 0, the nine hypotheses hold. The data set, the ingestion and the export are reported as already present, no file of the available branch was touched, and the six content hashes are those of the registered run: `9006d4e1bf9ff9d0...`, `aba65cd6c2817a8a...`, `6d9b8c99a852e794...` on the source and `c45d6a0957b867e0...`, `9e561bfc61983740...`, `37a3b227eec190ae...` on the target. The largest of the 102 z-scores is 2.58, as registered. Run time 15.6 s.

**Interpretation.** The identities, the segments and the content of P3 are what they were: the generalisation of the definitions to three protocols changed nothing of the first one, which is what the re-run was for.

## M1

The plan of M1 is `docs/m1_plan.md`. Development runs made while implementing an iteration are recorded here as such: they are not experiments, nothing in them was registered, and none of their numbers is a result of M1.

### Development runs of I1 (2026-09-25)

Two runs, made after the tests of I1 passed, from clean commits, with PT_DATA_DIR set to a new directory outside the repository and outside OneDrive. Environment: Python 3.13.7, numpy 2.5.3, scipy 1.18.1, the versions of the reference environment of M0.

#### Smoke run on the exports of M0

Code: `experiments/m1_i1_smoke_run.py` at commit `bfb69ab`, clean working tree; run `20260924T232014Z_bfb69ab` under `PT_DATA_DIR/development/m1-i1-smoke/`. Data: the exports `m0-e05` and `m0-e07`, read only, from the data directory of the repository. A fingerprint of every file under `available/exports` was the same before and after the run.

**Hypothesis.** None about the plants. About the software: the code of I1 runs end to end on real exports; the windows, budgets and parts are those the index contract gives on runs without a lead; every declared start of MR and MR_F ends on a declared criterion; the structured models give no integration failure, no violation of the validity bounds and no implied term that the physics forbids, since they satisfy them by construction; the reference integration stays below 1 % of sigma on real windows.

**Method.** The three target P3 runs of `m0-e05`, each taken as a replicate with a budget of its nine windows, F the first eight and V the last. MR fitted on the nine windows and MR_F on F, from the five declared starts (D-032). MN, MR_F and MR evaluated on the windows of their replicate, on the windows of the two other runs and on the eight target steps of `m0-e07`, the last two held out from every model; MR_F also on V alone, which it never saw. Validity bounds and implied terms on every completed window. The reference integration compared with LSODA and DOP853 at rtol = 1e-12 on the nine windows of the first run and the eight steps, for MN and for the MR of the first run. Everything is in `summary.json` of the run.

**Result.** In each run the first excursion, at tick 0, forms no window, for lack of a context; excursions 2 to 10 form nine, F holds 880 scored readings, and the prefix ends at tick 1200. The fitting parts visit 6, 8 and 6 distinct corners, with sign vectors of rank 4 in each. All 30 starts converged, in 9 to 17 evaluations of the residuals; within each fit the five endpoints differ by at most 0.011 K in E/R and 4e-7 relative in k_350 and UA. The singular values of the Jacobian of the normalised residuals at the selected endpoints lie between 2308 and 2519, 485 and 525, and 17.7 and 19.8.

| Run | Model | k_350, 1/s | E/R, K | UA, W/K | J, own windows | J, other P3 runs | J, steps | J, V |
|---|---|---|---|---|---|---|---|---|
| e0 | MN | 0.016666 | 8750 | 1666.7 | 11.46 | 11.13 | 11.22 | |
| e0 | MR | 0.017716 | 9598.1 | 1329.8 | 2.241 | 2.535 | 1.788 | |
| e0 | MR_F | 0.017682 | 9684.9 | 1331.1 | 2.244 | 2.546 | 1.754 | 2.475 |
| e1 | MN | | | | 11.01 | 11.36 | 11.22 | |
| e1 | MR | 0.017437 | 9436.2 | 1311.9 | 2.665 | 2.333 | 1.912 | |
| e1 | MR_F | 0.017364 | 9446.7 | 1309.4 | 2.665 | 2.340 | 1.926 | 1.695 |
| e2 | MN | | | | 11.25 | 11.23 | 11.22 | |
| e2 | MR | 0.017622 | 9568.3 | 1320.4 | 2.361 | 2.478 | 1.799 | |
| e2 | MR_F | 0.017626 | 9627.8 | 1321.9 | 2.363 | 2.486 | 1.770 | 2.464 |

The "own windows" of MR_F are its eight windows of F and the window of V together, and are labelled as fitting data, so no excess over the noise is computed for them. No evaluation had an integration failure, a violation of a validity bound, a negative implied rate or an implied heat flow incompatible with a non-negative conductance. The error of the reference integration against the two tighter ones was at most 2.2e-5 sigma for MN and 2.5e-5 sigma for MR. The run took 62.9 s.

**Interpretation.** The code of I1 runs end to end on real exports and does there what its tests say it does. The numbers are not results: they come from the data of M0, which is not the benchmark, three runs and one budget, and nothing was registered before they were seen. Nothing here compares models, or says how any of them would score on the benchmark of M1, and the estimates of E/R are not read as evidence about E/R, which is the question of I2. What the run does say about the software: on these data the fit reaches one endpoint from every start; the reference integration is about four hundred times more accurate on real windows than the plan requires; and MN's score of about 11 is of the size that section 2 of the plan leads one to expect, its steady state lying 12 sigma_CA and 10 sigma_T from the target's.

#### Calibration of the sandwich standard errors

Code: `experiments/m1_i1_sandwich_calibration.py` at commit `f71deea`, clean working tree; run `20260924T232253Z_f71deea` under `PT_DATA_DIR/development/m1-i1-sandwich-calibration/`; noise seeds 2000 to 2199, fixed before the run.

**Hypothesis.** When the model is right, the sandwich standard errors of MR describe the spread of its estimates: z = (estimate - true value) / standard error is standard normal for each of ln k_350, E/R and ln UA.

**Method.** The data of the recovery test of I1: the modeller's own equations at k_350 = 1.3 times the textbook value, E/R = 9200 K and UA = 0.85 times the textbook value, simulated on the truth side from their steady state, with a lead of 60 s and eight P3 excursions whose corners come from seed 2026; readings with the noise of D-020, one seed per replicate. MR fitted on the eight windows from the textbook start; its covariance computed at the estimate with the error of the context mean (the sandwich) and without it. The spread of z over the 200 replicates, with a bootstrap interval of its standard deviation.

**Result.** 200 fits, no training failure. Means of z: -0.001, -0.061 and -0.043. Standard deviations: 1.014, 1.004 and 1.011, with 95 % bootstrap intervals of 0.93 to 1.09, 0.91 to 1.10 and 0.90 to 1.11. Largest |z|: 2.34, 2.91 and 3.15; fraction beyond 2: 0.025, 0.060 and 0.055, against 0.046 for a standard normal. With the initial state taken as exact, the standard deviations are 1.03, 1.10 and 1.04. Mean sandwich standard errors: 0.0021 in ln k_350, 30.0 K in E/R and 0.0013 in ln UA. 201 s.

A first pass over 40 seeds, 1000 to 1039, with a script outside the repository and the same fit, had given standard deviations of 1.23, 1.12 and 1.09. The first is about two standard errors of such an estimate above 1, which is what prompted this run over 200 seeds; its seeds were fixed before it was run, and it is the run recorded here.

**Interpretation.** With the right structure and these eight windows, the sandwich standard errors describe the spread of the estimates of MR to within about 10 %, the width of the bootstrap intervals. Leaving out the error of the initial state understates the spread of E/R by about 10 %, and those of ln k_350 and ln UA by a few per cent. This supports the criterion of four standard errors in the recovery test of I1, and the use of the sandwich in part 1 of M1-E01. It says nothing about a model whose structure is wrong, which is the case of MR on the target.

### Development runs after Codex's audit of I1 (2026-09-25)

Codex audited `124c377` to `a8905dc` and reported six findings, corrected in `d785135` to `b5c2838`, with a last fix of the covariance in `08ebcf6` after these runs (`docs/numerical_robustness.md`, twelfth review). These runs check that the corrections change nothing at the scales of the plants and that the tables of the smoke run keep their denominators. They are development, like the runs above, and none of their numbers is a result of M1. All were made from the clean commit `b5c2838`, with PT_DATA_DIR set to a new directory outside the repository and outside OneDrive, and the exports of M0 read only: a fingerprint of every file under `available/exports` was the same before and after.

#### The smoke run, repeated

Run `20260925T170124Z_b5c2838`, `experiments/m1_i1_smoke_run.py` as in `20260924T232014Z_bfb69ab`.

**Hypothesis.** The corrections change no fit, and no score beyond the rounding of the new way of forming squares: the fits of the audited run are reproduced bit for bit and its scores to within a few units in the last place.

**Method.** The same exports, replicates, budget, starts and settings. Every number of the two summaries compared, path by path, apart from times and the provenance of the attempt.

**Result.** 11 332 numbers compared; 4 492 are equal bit for bit, among them every parameter, objective, singular value, covariance, count of evaluations and excitation, and the error of the reference integration, 2.53e-5 sigma at most as before. The other 6 840 are metrics: MSE, MSE / sigma^2, RMSE and J differ by at most 3.7e-15 relative, and MSE - sigma^2 by at most 1.2e-12 relative, which is 4e-16 absolute on a difference of nearly equal numbers. All 30 starts converged, as before. The summary now also holds the tables of the windows of each replicate and of V. 68.2 s.

**Interpretation.** At the scales of the plants the corrections are neutral: the fits are the same numbers, and the scores move in their last bits because squares are now formed on scaled values. What the audit found lay outside these scales.

#### Training failures, reproduced on purpose

Run `20260925T170305Z_b5c2838`, the same with `--max-evaluations 1`.

**Hypothesis.** With a limit of one evaluation no start can converge, MR and MR_F are training failures in every replicate, and every table still counts the three replicates.

**Method.** The smoke run with every start limited to one evaluation of the loss; the failure tables of every model and view.

**Result.** Every start of MR and MR_F ended with its budget exhausted, and both fits are training failures in the three replicates. Each of their tables, on the windows of the replicate, the other runs, the steps and, for MR_F, V, holds three replicates, none scored and three training failures. MN, which is never trained, is scored in all three on each of its views.

**Interpretation.** A replicate whose model was not trained stays in the denominator as a failure of that model, which is what section 9.3 of the plan asks and what the audited version did not do.

#### The calibration of the sandwich errors, repeated

Run `20260925T170324Z_b5c2838`, `experiments/m1_i1_sandwich_calibration.py` over the noise seeds 2000 to 2199, as in `20260924T232253Z_f71deea`.

**Hypothesis.** The fixes to the fit and to the covariance add checks and change no number where the numbers are doubles, so the calibration repeats bit for bit.

**Method.** The same data, fit and seeds; the 200 rows of the two summaries compared.

**Result.** The 200 rows, each an estimate, its z-scores and its standard errors, are equal bit for bit, and so is every statistic: standard deviations of z of 1.014, 1.004 and 1.011, with the same bootstrap intervals. 191 s.

**Interpretation.** The corrected fit and covariance give the numbers they gave before the audit on these data; the calibration recorded above stands as it was. The last fix of the covariance, `08ebcf6`, came after this run and changes only how overflow is reported for noise levels of 1e200.
