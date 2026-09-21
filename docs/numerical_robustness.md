# Numerical robustness

The rule itself lives in `AGENTS.md`, under Numerical Robustness. This file is the record: what has been examined, which failures were reproduced, how they were resolved and what is still open. A new review adds a dated section; it does not rewrite the earlier ones.

One mechanism enforces part of the rule on every test run: `pyproject.toml` turns every `RuntimeWarning` into an error, so a division by zero or an invalid operation inside numpy, which would otherwise only print a warning and carry on with `inf` or `NaN`, fails the test that triggers it.

## Review of 2026-09-22: the M0 modules as they stood at `50dc509`

Scope. A bounded review of the interfaces that exist today: `units`, `config`, `simulation/cstr_true`, `modeller/cstr_first_order`, `simulation/steady_state`, `simulation/integration`, `simulation/envelope`, `simulation/excitation`, `simulation/balances`, `simulation/checks` and `data/paths`. For each: denominators, exponentials, interpolation, empty sets, grid generation, and the result of conversions and products, not only their inputs. It is not an enumeration of every possible number; it looks at inputs that the current interfaces can receive and at operations that fail without saying so. Every candidate was run against the code before anything was changed, and only reproduced failures were fixed.

### Failures reproduced and fixed

Two were found by the reviewer of `50dc509`; the rest by probing the interfaces.

| Module | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `integration.refined_peak` | a higher maximum between the samples of another segment, or around another local maximum | returned 379.9 K where the second segment peaks near 380.988 K | yes | every triple of every segment is a candidate; no parabola across an input change; uneven spacing, plateaus, repeated instants and fewer than three samples handled explicitly | `41695aa` |
| `balances` | endothermic reaction | the residual was divided by a signed scale; a last sample moved by 1 K gave -0.0228 and passed | yes | scale by the traffic through the balance; acceptance written as a product | `e9ccfb2` |
| `balances` | zero enthalpy | `ZeroDivisionError` | no | the same; zero traffic is resolved explicitly | `e9ccfb2` |
| `config` | 1e306 mol/L; a volume of 1e-322 L | SI value infinite; a positive volume became exactly 0 m^3, by which the balances divide | yes | the SI value is checked: finite, and non-zero when the written value is non-zero | `338de24` |
| model parameters built directly | volume 0; negative rate constant | right-hand side returned `inf` and `NaN` with a numpy warning; the reaction ran backwards | yes | validated once, at construction, including the products V rho cp and dH / (rho cp) | `e1ba2d9` |
| `steady_state.find_steady_states` | a right-hand side returning NaN | returned an empty list, read as "no steady state", because a NaN never changes sign | yes | an error naming the temperature | `a4e929e` |
| `steady_state.find_steady_states` | range starting at 0 K; reversed range; one grid point; concentration bound of zero | division by zero inside the rate law with a warning; accepted; accepted; scipy's message | partly | arguments validated, with messages that say what to change | `a4e929e` |
| `steady_state.numerical_jacobian` | step of zero; right-hand side returning infinity | Jacobian of NaN | yes | rejected; finiteness checked before subtracting | `a4e929e` |
| `envelope.simulate_envelope` | a right-hand side returning NaN | reported a temperature range of (350, 350) | yes | runs through `simulate_piecewise`, which raises on non-finite states | `35a0067` |
| `envelope.input_cases` | a negative deviation | the case labelled `q+` lowered q | yes | deviations must be finite and not negative; zero is valid | `35a0067` |
| `excitation.levels_to_segments` | a level of 7 | the input moved by seven amplitudes | yes | levels must lie between -1 and +1 | `35a0067` |
| `integration.simulate_piecewise` | `rtol = 0` | scipy substituted its own tolerance and warned | partly | tolerances must be positive and finite | `35a0067` |
| `integration.sample_times` | 1e6 s at 1e-9 s; 1e308 s at 1e-308 s | `MemoryError` for 7 PiB; `OverflowError` | no | refused above ten million samples per segment, with advice | `35a0067` |
| `integration.Trajectory` | no segments | `IndexError` on first use | no | rejected at construction | `35a0067` |
| `checks` | negative feed flow | trajectory reported as physical | yes | the four inputs must be positive | `01c55ea` |
| `checks` | temperature below T_ref - 1/alpha | the linear conductance turned negative and moved heat the wrong way | yes | a trajectory reaching that region is not physical; zero conductance stays valid | `01c55ea` |

### Valid limits, resolved explicitly

These are not errors, and are handled as the physics says:

* a reaction enthalpy of either sign, or zero (D-015);
* no reaction at all, or a negligible one: the mass balance is judged on the flow terms;
* no traffic at all through a balance: nothing may accumulate, and nothing is divided;
* a heat effect too small to change a stored temperature: compared with the floating-point resolution of the accumulation, which is not a tunable number;
* zero conductance (adiabatic), zero saturation constant, zero activation temperature;
* zero deviation in the input cases; a sampling period longer than the duration;
* a plateau of equal samples, which has no interior maximum to refine.

Several of these use parameter values that the configuration schema does not admit, such as a rate constant of zero. They are built directly in tests to exercise the numerical functions at their limits. The schema was not widened.

### Examined and left as they are

* The rate law at T <= 0 or 1 + K_sat C_A <= 0 is infinite or undefined, and at exactly T = 0 raises `ZeroDivisionError`. It is not guarded, on purpose: it runs inside the integrator, millions of times. Those states are reachable only if an integration leaves the physical domain, and then the states stop being finite, which `simulate_piecewise` raises on, or stop being physical, which `checks` reports.
* exp(-(E/R)/T) for T > 0 and E/R >= 0 lies in (0, 1]; it cannot overflow.
* The step of the numerical Jacobian is `rel_step * max(|x|, 1)`, never zero for a valid `rel_step`.
* Sampling instants are multiples of the period, not a running sum, so they do not drift.
* Simpson's rule on two samples is the trapezoid; a segment always has its two ends.
* Every unit conversion is a pure factor; there are no offsets, since only absolute temperatures are accepted.
* `data/paths` has nothing numerical in it.

### Open limitations

* The refined peak is an estimate, not a bound. A peak much narrower than the sampling period leaves no trace in the samples. Critical cases are recomputed with finer sampling and a second integrator; that is a practice, not a guarantee.
* The steady-state scan cannot see tangent roots, two roots in one grid cell, or anything outside its range; a count of one is not a proof of uniqueness (`simulation/steady_state.py`).
* The physical rules describe a reactor that only consumes A, and the temperature rule exists for the exothermic case only. The symmetric rule for an endothermic reaction, T never above the hottest stream, is not implemented; no plant of M0 is endothermic.
* The limit on the sampling grid is per segment. The total over many segments is not bounded.
* Overflow of the states themselves during an integration is not checked at every step; it is caught afterwards, as non-finite states.
* Parameters and arguments are validated where they enter. Code that mutates an array after handing it over can still defeat that; arrays are not copied defensively everywhere.
