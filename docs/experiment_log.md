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

**Interpretation.** H1 holds. H2 holds for the source and is rejected for the target, which leaves the envelope by 5.3 K. The narrower envelope quoted in the first version of `docs/assumptions.md` was wrong: the scratch calculation simulated four cases and took the low-flow corner to be the hot one, whereas more flow of a richer feed brings more reactant and more heat. No multiplicity and no loss of stability appears anywhere on the input box, so the violation is a matter of amplitude and weak damping, not of a bifurcation. The 380 K limit is a documented convention, not a property of the fluid. Remedies are evaluated in M0-E02.

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

The source stays inside the envelope under every option (339.67 to 372.44 K at most). Under B the source keeps its nominal point with eigenvalues -2.651 +- 0.737i 1/min. No option produces multiple steady states in any input case.

**Interpretation.** A at +-10 % and B restore the envelope; A at +-15 % misses by 0.6 K and C does not help. A at +-10 % costs little visibility of the kinetic mismatch, because the temperature inputs drive most of the concentration excursion. B gives the strongest damping but changes the accepted plant design and brings the target's operating point closer to the source's. The choice modifies accepted decisions (D-010 or D-005) and is recorded as the open decision D-018.
