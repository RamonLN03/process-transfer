# Experiment log

One entry per experiment or data-generation run, in order. Each entry separates hypothesis, method, result and interpretation, and records the code version, configuration and seeds. Results are not written before they exist.

## M0

No plant data have been generated yet. The entries below are deterministic verifications of the virtual plants.

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
