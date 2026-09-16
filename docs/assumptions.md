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

## Operating point and envelope (preliminary)

A scratch calculation outside the repository, to be reproduced by repository code before these numbers are quoted anywhere else, gave: source nominal point C_A = 0.25 mol/L, T = 350 K, unique steady state, eigenvalues -1.60 +- 1.36i 1/min; target C_A = 0.190 mol/L, T = 355.2 K, unique, eigenvalues -0.96 +- 1.80i 1/min. Under the planned excitation amplitudes the source stayed within 341 to 363 K and the target within 343 to 372 K. Documented envelope for M0: T in [335, 380] K, C_A in (0, 1.2 C_Af].

## Measurement

* Measured variables: C_A and T. Inputs q, C_Af, T_f and Tc are known exactly (set or measured without error).
* Sampling every 0.1 min for all variables.
* Additive, independent, zero-mean Gaussian noise: sigma_T = 0.5 K, sigma_CA = 0.005 mol/L. No bias, no drift, no delay, no missing values (all deferred).

## Sources

Seborg, D. E., Edgar, T. F., Mellichamp, D. A., Doyle, F. J. Process Dynamics and Control, Example 2.5 (exothermic CSTR), for the chemistry and physical properties.
