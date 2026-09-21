# Experiment log

One entry per experiment or data-generation run, in order. Each entry separates hypothesis, method, result and interpretation, and records the code version, configuration and seeds. Results are not written before they exist.

## M0

No data set has been stored yet. M0-E01 to M0-E03b are deterministic verifications of the virtual plants; M0-E04 generates observations in memory, seeded, and checks them.

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

**Result.** Not run yet at the commit that registers this definition.
