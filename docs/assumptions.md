# Physical assumptions of the M0 virtual plant

Every simplification that affects what the simulator can and cannot represent is listed here. Later milestones that relax an assumption update this file.

## Process

* A single liquid-phase irreversible reaction A -> B in a continuous stirred-tank reactor.
* Perfect mixing: reactor contents are uniform, and the outlet has the composition and temperature of the contents.
* Constant volume: inflow equals outflow, no level dynamics.
* Constant density and heat capacity, independent of composition and temperature.
* Constant heat of reaction.
* Heat removal proportional to (T - Tc), with the coolant temperature Tc as a manipulated input; no jacket energy balance, no coolant dynamics.
* No heat loss to the ambient.
* No catalyst deactivation, no fouling drift, no side reactions (deferred difficulty levels).

## Equations

True plant (simulation truth):

    dC_A/dt = (q/V) (C_Af - C_A) - r_true(C_A, T)
    dT/dt   = (q/V) (T_f - T) + (-dH / (rho Cp)) r_true(C_A, T) - UA_true(T) / (V rho Cp) (T - Tc)

    r_true(C_A, T) = k0_true exp(-E/(R T)) C_A / (1 + K_sat C_A)      saturating kinetics
    UA_true(T)     = UA_ref [1 + alpha (T - T_ref)]                     temperature-dependent conductance

Modeller's equations (available knowledge):

    dC_A/dt = (q/V) (C_Af - C_A) - k0 exp(-E/(R T)) C_A
    dT/dt   = (q/V) (T_f - T) + (-dH / (rho Cp)) k0 exp(-E/(R T)) C_A - UA / (V rho Cp) (T - Tc)

The modeller knows the reaction, that it is exothermic, the Arrhenius form, the physical properties, the design and the measured inputs. The modeller does not know the saturating form of the rate or the temperature dependence of UA.

## Parameter table

| Symbol | Meaning | Source (true) | Target (true) | Modeller | Shared or plant-specific |
|---|---|---|---|---|---|
| V | volume | 100 L | 100 L | known | plant-specific (equal in M0) |
| q | feed flow | 100 L/min | 100 L/min | known input | operation |
| C_Af | feed concentration | 0.5 mol/L | 0.5 mol/L | known input | operation |
| T_f | feed temperature | 350 K | 350 K | known input | operation |
| Tc | coolant temperature | 337.5 K | 337.5 K | known input | operation |
| rho | density | 1000 g/L | 1000 g/L | known | shared |
| Cp | heat capacity | 0.239 J/(g K) | 0.239 J/(g K) | known | shared |
| dH | heat of reaction | -5.0e4 J/mol | -5.0e4 J/mol | known | shared |
| E/R | activation temperature | 8750 K | 8750 K | estimable | shared |
| k0 (first order) | modeller pre-exponential | n/a | n/a | estimable, nominal 7.2e10 1/min | shared |
| k0_true | true pre-exponential | 1.44e11 1/min | 1.44e11 1/min | hidden | shared |
| K_sat | saturation constant | 4 L/mol | 4 L/mol | hidden form | shared |
| UA | modeller conductance | n/a | n/a | estimable, nominal 1.0e5 J/(min K) | plant-specific |
| UA_ref | true conductance at T_ref | 1.0e5 J/(min K) | 0.8e5 J/(min K) | hidden | plant-specific |
| alpha | conductance slope | 0.005 1/K | 0.002 1/K | hidden form | plant-specific |
| T_ref | reference temperature | 350 K | 350 K | hidden | shared |

Whether E/R counts as known or estimable for the modeller is open and only matters from M1.

## Operating points (M0-E01)

Computed by `experiments/00_verify_operating_points.py`; see `docs/experiment_log.md`, M0-E01.

| | Source | Target |
|---|---|---|
| Steady states for nominal inputs, 280 to 480 K | 1 | 1 |
| C_A | 250.02 mol/m^3 (0.25002 mol/L) | 189.67 mol/m^3 (0.18967 mol/L) |
| T | 350.00 K | 355.17 K |
| Conversion | 50.0 % | 62.1 % |
| Eigenvalues | -1.605 +- 1.362i 1/min | -0.964 +- 1.804i 1/min |
| Stable with the 0.5 1/min margin (D-017) | yes | yes |

The source is designed for 250 mol/m^3 at 350 K (D-005). The book's k0 and E/R give k(350 K) = 0.99993 1/min rather than exactly 1, so the computed point differs from the design values by 0.02 mol/m^3 and 0.8 mK.

## Operating envelope

Documented envelope for M0: T in [335, 380] K, C_A in (0, 1.2 C_Af]. The temperature limits are a convention of this study, not a property of the fluid. They have not been changed to admit any result.

Three kinds of evidence exist, and they must not be confused.

### Steps from the nominal steady state, original D-010 amplitudes (historical, M0-E01)

q and C_Af +-20 %, T_f and T_c +-5 K; 8 single-input excursions and the 16 corners of the input box, each applied as a 40 min step from the nominal steady state.

| | Source | Target |
|---|---|---|
| Temperature range over the 24 cases | 339.67 to 372.44 K | 341.19 to 385.29 K |
| Concentration range | 110 to 440 mol/m^3 | 46 to 392 mol/m^3 |
| Least stable case, max real part | -0.896 1/min | -0.381 1/min |
| Inside the documented envelope | yes | no, by 5.3 K on the all-plus corner |

Each of the 24 tested input cases has one steady state in the scanned range, locally stable, on both plants, and every step response converges to it. The hottest case is the all-plus corner, with q, C_Af, T_f and T_c all high: more flow of a richer feed brings more reactant and therefore more heat. The first version of this file quoted a narrower range from a scratch calculation that simulated four cases and assumed the hot corner had low flow; that assumption was wrong. These conditions are no longer the planned ones, and are kept under test only so that M0-E01 stays reproducible.

### Steps from the nominal steady state, A10 amplitudes (current, D-018)

q and C_Af +-10 %, T_f and T_c +-5 K; the same 24 cases.

| | Source | Target |
|---|---|---|
| Temperature range over the 24 cases | 340.42 to 365.75 K | 342.37 to 376.19 K |
| Inside the documented envelope | yes | yes |

This is all that A10 has been shown to satisfy: single steps, from the nominal steady state, at 24 input cases. It says nothing about the interior of the input box, about the rest of its boundary, or about what happens when one change follows another.

### Chained input changes (M0-E03): the requirement is open

When changes are chained, the state at each change depends on the history. From the nominal steady state of the target, 120 s at q = 110 L/min, C_Af = 0.55 mol/L, T_f = 345 K and T_c = 332.5 K raise C_A from 190 to 347 mol/m^3; raising T_f to 355 K and T_c to 342.5 K then takes the reactor to 395.63 K, 42.6 s later. With random binary levels at the A10 amplitudes on a 120 s clock, every one of 20 seeded sequences of 2 h leaves the envelope on the target, with peaks between 381.5 and 396.6 K. The source stays inside in every case tested.

Four alternative protocols passed every case that was simulated (D-019). That is evidence about the sequences tested, not a guarantee over all the sequences a protocol can generate. The requirement that open-loop excitation stays inside the envelope (D-009) is therefore not satisfied for sequences at present, the excitation protocol is an open decision, and no plant data are generated until it is taken.

## Measurement

* Measured variables: C_A and T. Inputs q, C_Af, T_f and Tc are known exactly (set or measured without error).
* Sampling every 0.1 min for all variables.
* Additive, independent, zero-mean Gaussian noise: sigma_T = 0.5 K, sigma_CA = 0.005 mol/L. No bias, no drift, no delay, no missing values (all deferred).
* Open ambiguity (D-020). D-010 states sigma_CA as 2 % of the nominal concentration, which equals 0.005 mol/L on the source (nominal 0.2500 mol/L) but 0.0038 mol/L on the target (nominal 0.1897 mol/L). Whether the noise is an absolute property of the analyser, the same on both plants, or relative to each plant's nominal concentration is not decided. No sensor is implemented yet.

## Sources

Seborg, D. E., Edgar, T. F., Mellichamp, D. A., Doyle, F. J. Process Dynamics and Control, Example 2.5 (exothermic CSTR), for the chemistry and physical properties.
