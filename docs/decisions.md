# Decision log

Short records of decisions that a reader months from now would otherwise have to reconstruct. Newest at the bottom. Status is one of proposed, accepted, superseded.

## D-001 Names (2026-09-16, accepted)

Product and project: ProcessTransfer. Python package: `process_transfer`. Local working folder: `plant-transfer`, kept as is. The three names differ on purpose and are not to be reconciled by renaming the folder.

## D-002 Language (2026-09-16, accepted)

Conversation with agents may be in Spanish; everything in the repository is English.

## D-003 Write ownership (2026-09-16, accepted)

Claude Code implements and commits; Codex reviews read-only unless ownership is explicitly transferred. See AGENTS.md, Multi-Agent Coordination.

## D-004 First hidden-physics design (2026-09-16, accepted)

Shared hidden physics: saturating kinetics more complex than the modeller's first-order law. Plant-specific hidden physics: temperature-dependent heat-transfer conductance, while the modeller uses a constant UA per plant. No secondary reaction, no sensor bias, no structural reaction-network mismatch yet. Continuous-time modelling direction; open-loop operation at a verified stable point.

Alternatives considered for the kinetics: non-integer reaction order (kept as a later difficulty level), modified Arrhenius temperature dependence, reversible reaction (rejected as a structural change), product inhibition (adds a state). For heat transfer: flow-dependent U (poorly motivated when coolant temperature is the input), fouling drift (non-autonomous, later), jacket dynamics (hidden state, later), ambient heat loss (absorbed by a re-estimated UA).

Why: the kinetic mismatch must be a functional form that no parameter re-fit can remove, concentration-dependent so that it appears under excitation and is invisible at the nominal steady state, and shared across plants so that it is transferable. The heat-transfer mismatch must be state-dependent so that a static learned correction can represent it, plant-specific in slope as well as level, and free of hidden states.

## D-005 CSTR parameterisation (2026-09-16, accepted)

Chemistry and physical properties from the exothermic CSTR of Seborg, Edgar, Mellichamp and Doyle, Process Dynamics and Control, Example 2.5: k0 = 7.2e10 1/min for the first-order law, E/R = 8750 K, dH = -5.0e4 J/mol, rho = 1000 g/L, Cp = 0.239 J/(g K), V = 100 L, q = 100 L/min, T_f = 350 K. Plant design chosen for this study: feed concentration 0.5 mol/L (book: 1.0), heat-transfer conductance 1.0e5 J/(min K) (book: 5.0e4), coolant temperature 337.5 K (book: 300 K), giving a nominal point at C_A = 0.25 mol/L, T = 350 K, 50 % conversion.

Why: the book's nominal point is an open-loop saddle and its only stable branch sits at 12 % conversion, where the kinetics is nearly invisible. A more dilute feed halves the adiabatic temperature rise, and a larger cooling conductance (a coil in addition to the jacket) makes the 50 % point stable and unique. The chemistry stays published and checkable; feed and cooling are exactly the plant-design quantities that vary between plants. The alternative of tripling UA gives more damping at the cost of a less plausible heat-transfer area. Preliminary numbers come from a scratch calculation and must be reproduced by repository code before they are quoted anywhere.

## D-006 Naming of the hidden kinetics (2026-09-16, accepted)

The true rate law r = k(T) C_A / (1 + K_sat C_A) is described as saturating, Langmuir-Hinshelwood-like kinetics. No specific Langmuir-Hinshelwood mechanism is claimed, because none is being derived. K_sat = 4 L/mol so that K_sat C_A = 1 at the nominal point, and k0_true = 1.44e11 1/min so that the nominal rate equals that of the modeller's first-order law with the book's k0: a modeller who fits at the nominal point recovers the textbook parameters and is wrong away from it.

## D-007 Heat-transfer mismatch parameters (2026-09-16, accepted, values provisional)

UA_true(T) = UA_ref [1 + alpha (T - 350 K)]. Source: UA_ref = 1.0e5 J/(min K), alpha = 0.005 1/K. Target: UA_ref = 0.8e5 J/(min K), alpha = 0.002 1/K. The magnitude of alpha is the least certain choice; M0 reports the size of the mismatch under the planned excitation so that alpha can be confirmed or raised (0.01 1/K would make it unmistakable) before M1.

## D-008 Target plant in M0 (2026-09-16, accepted)

Same nominal inputs as the source; the target differs only in hidden heat-transfer behaviour. Known-parameter shifts (volume, flow) belong to the M3 domain-shift study. Same inputs mean the target settles at a different operating point, which is reported rather than corrected.

## D-009 Stability verification (2026-09-16, accepted)

All steady states of the true plant for nominal inputs are found by continuation in temperature with Newton refinement; the Jacobian is evaluated by finite differences and checked against the analytical Jacobian of the first-order case; eigenvalues must have real part below a margin; the steady state must be unique in the scanned range; every corner of the excitation range is simulated from the nominal point and must stay inside the documented envelope.

Update, 2026-09-21. Two limits of this criterion, both found after it was accepted. The steady-state search reports the roots it can see in the scanned range, under the limitations stated in `simulation/steady_state.py`; a count of one is not a proof of uniqueness. And single steps from the nominal point are a necessary test of an excitation, not a sufficient one: M0-E03 showed that chained changes reach temperatures that no step from nominal reaches. The check on sequences is `simulation/checks.py`; the protocol it has to be applied to is open (D-019).

## D-010 Excitation (2026-09-16, accepted)

Open loop, zero-order-hold inputs. Amplitudes: Tc +-5 K, T_f +-5 K, q +-20 %, C_Af +-20 %. Three operating runs per plant: steady operation with noise only; single-input step tests with 10 min holds; multi-level random binary sequences on all four inputs with a 2 min clock. Sampling every 0.1 min. Gaussian measurement noise, sigma_T = 0.5 K, sigma_CA = 2 % of the nominal concentration, independent, no bias, no drift. Steps of +-10 K were rejected after simulated peaks of 380 to 400 K.

Update, 2026-09-21. The amplitudes of q and C_Af are superseded by D-018 and are now +-10 %. The protocol for chained changes is open again (D-019): M0-E03 showed that random binary levels on a 2 min clock take the target far above 380 K even at the reduced amplitudes. The noise level of the C_A sensor is ambiguous between this entry and `docs/assumptions.md` (D-020). The text above is kept as it was accepted.

Update, 2026-09-21, after D-020. The noise of the C_A sensor is sigma_CA = 0.005 mol/L, 5 mol/m^3, on both plants; the "2 % of the nominal concentration" above is superseded and the percentage is dropped. Both sigmas are standard deviations of additive zero-mean Gaussian noise, not limits of the error. The measurement specification in force is D-020 and `configs/sensors_cstr.yaml`.

## D-011 Target data budget lives in the experimental layer (2026-09-16, accepted)

The target plant generates truth and evaluation data comparable in amount to the source. The simulator is never restricted. Adaptation budgets (initially about 2 h; later 30 min, 1 h, 2 h, 4 h, 8 h) are applied by the experiment code on top of the same target record and the same fixed evaluation set.

## D-012 Reproducibility criteria (2026-09-16, accepted)

Same seed: identical numerical trajectories, identical schema, identical canonical content where appropriate. No test depends on byte-for-byte identity of Parquet files or on a particular PyArrow version's serialisation.

## D-013 Units (2026-09-16, accepted)

A small explicit conversion table at the configuration boundary, not Pint. Unknown units are an error. SI floats inside the simulator. Units recorded in the `sensors` table.

## D-014 Data location and repository visibility (2026-09-16, accepted)

`PT_DATA_DIR` defaults to `data/` inside the repository, ignored by git; it can point outside synced folders. The repository is private for now (see project_scope.md, commercialisation).

## D-015 Sign of the reaction enthalpy is not constrained (2026-09-16, accepted)

`reaction_enthalpy` is a plain `Quantity`, not a positive-only or negative-only field. The M0 CSTR configuration uses the negative value of an exothermic reaction, but the configuration models must be able to represent endothermic reactions in later process families (M7), so no global sign constraint is added to the plant configuration. Sign expectations belong to a specific plant's documentation, not to the schema.

## D-016 No AI co-author trailers in commit messages (2026-09-16, accepted)

Commit messages carry no `Co-Authored-By` trailers naming AI agents. The four initial commits were rewritten on 2026-09-16, before any remote existed, to remove such trailers; content, authorship and dates were unchanged (identical tree hashes). Authorship of the repository is the user's; agent roles are recorded in AGENTS.md, Multi-Agent Coordination.

## D-017 Stability margin at the nominal point (2026-09-18, accepted)

At the nominal inputs every eigenvalue must have real part below -0.5 1/min (-8.33e-3 1/s), so that disturbances decay with a time constant of at most two minutes. The value was part of the M0 design proposal accepted on 2026-09-16 and is recorded here because D-009 left it unstated. At each tested input case, 8 single-input excursions and the 16 corners of the input box, the requirement is a unique, locally stable steady state in the scanned range, without a margin, in addition to the envelope of D-009. This is a check at 24 points; it is not a statement about the rest of the input box.

## D-018 Target leaves the temperature envelope under the D-010 amplitudes (2026-09-18; A10 chosen 2026-09-21; sequence safety not settled)

M0-E01 found that the target peaks at 385.3 K on the all-plus corner of the input box, 5.3 K above the documented limit, while each of the 24 tested input cases keeps a unique stable steady state. M0-E02 evaluated the remedies (target plant):

| Option | Change | T range, K | Least stable case, 1/min | r_true / r_model | Envelope |
|---|---|---|---|---|---|
| Baseline | none | 341.2 to 385.3 | -0.381 | 1.69 to 0.78 | violated |
| A15 | q and C_Af +-15 % | 341.8 to 380.6 | -0.610 | 1.63 to 0.80 | violated by 0.6 K |
| A10 | q and C_Af +-10 % | 342.4 to 376.2 | -0.723 | 1.58 to 0.83 | met |
| B | UA_ref x1.5 on both plants, T_c = 341.667 K | 342.8 to 372.5 | -1.176 | 1.42 to 0.76 | met |
| C | target UA_ref = 0.9 x source | 340.2 to 382.3 | -0.632 | 1.63 to 0.74 | violated |

Considerations. A10 changes only the excitation (D-010) and keeps the accepted plant design and the source-target shift; it costs little visibility of the kinetic mismatch, because the temperature inputs drive most of the concentration excursion. B changes the plant design (D-005), moves the target's operating point closer to the source's (216 mol/m^3 and 352.9 K instead of 190 mol/m^3 and 355.2 K) and implies the larger heat-transfer area that D-005 already judged less plausible, in exchange for the strongest damping. C does not solve the problem. Raising the 380 K limit is possible, since the limit is a convention, but it would fit the criterion to the result.

Recommendation of the implementation agent: A10. The decision belongs to the project owner. Until it is taken, the target envelope test is a strict expected failure and no data are generated.

Decision, 2026-09-21. The project owner chose A10: q and C_Af +-10 %, T_f and T_c +-5 K. The strict expected failure was removed; the original +-20 % conditions stay under test as a historical regression of M0-E01.

Limitation, found in the review of `f6c5781` and confirmed by M0-E03. Every option above, A10 included, was judged on single steps from the nominal steady state at 24 input cases. That is what A10 satisfies, and nothing more. When input changes are chained the state at each change depends on the history: a cold stage of 120 s at the A10 levels followed by a hot one takes the target to 395.6 K, and with random binary levels every one of 20 seeded sequences leaves the envelope. A10 therefore fixes the amplitudes for steps from nominal. It does not satisfy the requirement that open-loop excitation stays inside the envelope (D-009) for sequences, and that requirement is not closed by the change of amplitudes. It continues as D-019. No data are generated until D-019 is decided.

## D-019 Protocol for chained input changes (2026-09-21, P3 accepted as the initial protocol)

Open requirement: open-loop excitation must keep both plants inside the documented envelope (D-009) when input changes are chained, not only for steps from the nominal steady state. D-018 does not settle it. M0-E03 evaluated five protocols on a 120 s clock, with 20 seeds of 2 h each, adversarial ramps and, for binary protocols, all two-stage corner transitions. Target plant, the binding one:

| Protocol | Worst peak found, K | Margin to 380 K | r_true / r_model excited | Verdict on tested cases |
|---|---|---|---|---|
| P0: A10, binary levels | 396.57 | none, 20 of 20 seeds rejected | 1.49 to 0.87 | fails |
| P1: thermal inputs +-2.5 K, binary levels | 378.06 | 1.9 K | 1.35 to 0.97 | passes |
| P2: A10, three levels, one level per tick | 376.99 | 3.0 K | 1.40 to 0.90 | passes |
| P3: A10, 120 s excursions separated by 600 s at nominal | 376.19 | 3.8 K | 1.33 to 0.99 | passes |
| P4: thermal +-2.5 K, three levels, one level per tick | 369.14 | 10.9 K | 1.29 to 1.00 | passes |

What the evidence supports. P0 must not be used. P1 to P4 passed every case that was simulated, which is evidence about those cases and not a guarantee over all sequences. For P1 and P2 the worst seeded peak exceeded the designed worst case, by 0.75 K and 4.4 K, so histories longer than two stages matter and their margins, 1.9 K and 3.0 K, are of the same size as that effect. P3 has a structural argument on top of the tested sequences: after a rest of about ten time constants each excursion starts practically at the nominal steady state, its responses reduce to the 16 corner steps, which are enumerated exhaustively, and its worst peak equals the step reference. P4 has by far the largest margin and the narrowest excitation.

Recommendation of the implementation agent, not a decision. Use P3 as the default sequential protocol. Its safety rests on the enumerated steps and not on a sample of random sequences, and its excursions are countable tests, which is the unit in which D-011 already measures the adaptation budget. Where continuous excitation is wanted, use P4 and accept the narrower range. Treat P1 and P2 as not sufficiently supported until a wider study, with more seeds or a search for worst-case histories, shows otherwise. Do not enlarge the 380 K limit or change the physical design to admit a protocol.

Still open after this entry: the protocol itself, the length of the rest in P3 (600 s was tried, nothing else), whether single-input excursions are added to the corner excursions, and whether the 2 min clock of D-010 stands.

Decision, 2026-09-21. The project owner accepts P3 as the initial excitation protocol: A10 amplitudes, excursions of 120 s to the corners of the input box, 600 s of recovery at the nominal inputs, the state never being reset. It is defined once, in `src/process_transfer/simulation/protocols.py`.

Scope of the acceptance. P3 stands for the plants and conditions that were verified: the present source and target, these amplitudes, this hold and this rest. M0-E03b measured its premise instead of assuming it: after the rest the target is within 1.3 mK and 0.0065 mol/m^3 of its nominal steady state, the state carried into the next excursion changes its peak by less than a millikelvin, and over all 256 ordered pairs of excursions the largest peak is 376.19 K on the target and 365.75 K on the source, the peaks of the hottest corner from the exact steady state. That is evidence about what was simulated, not a guarantee. Another plant, such as a new target in M3, or another amplitude, hold or rest, needs the same verification before data are generated with it.

Not decided here: the rest time has been measured, not optimised (on the target 600 s meets the recovery tolerance by a factor of about four); single-input excursions are not part of P3; P1, P2 and P4 remain unsupported or unused; the noise level of the C_A sensor is still open (D-020).

Update, 2026-09-21. D-020 was accepted later the same day, with sigma_CA = 5 mol/m^3. The recovery tolerances of P3 were derived while both readings were open, from the stricter one, and they stay as they were verified: 0.005 K and 0.038 mol/m^3. The second is now 1/132 of sigma_CA instead of 1/100. The decision is not a reason to relax them. They are a practical criterion, not a statistical guarantee: see the note on the tolerances in `simulation/protocols.py`.

## D-020 Measurement noise; two readings of the noise level of the C_A sensor (2026-09-21, reading 1 accepted)

D-010 gives sigma_CA as 2 % of the nominal concentration. `docs/assumptions.md` gives 0.005 mol/L. The two agree for the source, whose nominal C_A is 0.2500 mol/L, and not for the target, whose nominal C_A is 0.1897 mol/L: 2 % of that is 0.0038 mol/L. No sensor is implemented yet, so nothing depends on the choice so far.

Reading 1, absolute. sigma_CA = 0.005 mol/L on both plants. The noise is a property of the analyser, which is the same instrument on both plants in M0. Relative to its own nominal concentration the target is then noisier, 2.6 % against 2.0 %.

Reading 2, relative to each plant. sigma_CA = 2 % of that plant's nominal C_A: 0.0050 mol/L on the source and 0.0038 mol/L on the target. The signal-to-noise ratio at the nominal point is equal on both plants.

Recommendation, not a decision: reading 1. Sensor noise belongs to the instrument and not to the operating point, and differences between sensors are deferred in M0, so both plants should carry the same analyser. Reading 2 would also make a sensor specification depend on the target's nominal steady state, which is fixed by hidden physics: ground truth would leak into something the modeller is supposed to know. If reading 1 is accepted, D-010 should say 0.005 mol/L and drop the percentage.

Decision, 2026-09-21. The project owner chose reading 1: the same absolute noise on source and target.

* sigma_CA = 0.005 mol/L = 5 mol/m^3.
* sigma_T = 0.5 K.
* Additive Gaussian noise of zero mean, independent between the two channels and between samples.
* No bias, no drift, no delay and no missing values.
* The inputs q, C_Af, T_f and T_c remain known without error.

Reason: one instrument specification common to both plants. In M0 source and target carry the same analyser and the same thermometer, and differences between sensors are deferred. The specification is written once, in `configs/sensors_cstr.yaml`; there is no per-plant sensor file.

sigma is a standard deviation, not a limit of the error. For Gaussian noise about 32 % of the readings lie further than one sigma from the true value, 4.6 % further than two and 0.27 % further than three. Readings are never clipped or truncated to stay within +-sigma, or inside any physical range; a reading is what the sensor said, and no criterion written for true states (positivity, closed balances) is applied to it.

Correction to the recommendation above. It argued that reading 2 would leak ground truth, because the nominal steady state of the target is fixed by hidden physics. That was overstated and is withdrawn as a reason. The concentration at which a plant normally runs is something its engineers observe, and an error stated as a percentage of reading or of span is an ordinary instrument specification. A relative noise level is not by itself a leak of ground truth; whether it would be one depends on where the number comes from and who is given it. The decision rests on the common instrument specification and on nothing else.

Consequences. Relative to its own nominal concentration the target is noisier than the source, 2.6 % against 2.0 %, which is accepted as part of the scenario. Why two readings existed is kept above: D-010 wrote the noise as a percentage, `docs/assumptions.md` as an absolute value, and the two agree on the source only. The recovery tolerances of P3 are unchanged (D-019).

## D-021 Identity of stored runs, and noise streams derived from it (2026-09-21, decided by the implementation agent, technical and reversible)

Taken under the authorisation of the project owner to settle reversible details of storage, identifiers, schema and queries. The full text is `docs/data_contract.md`.

Three things are kept apart: the logical identity of a run, `run_id`, built from plant, protocol, excitation seed, number of excursions and noise realisation; the generation attempt, UTC time and commit; and the content hash, a SHA-256 of a versioned canonical encoding of the observations. The same identity with the same content is accepted and changes nothing; the same identity with different content is a conflict and an error. A published data set is never modified.

The noise stream of a run is the first 128 bits of the SHA-256 of its `run_id`, as four 32-bit words, under one private master seed per data set. A repetition has the same identity and replays the same noise; a new realisation has a new `n<k>` and a new stream; two runs with one identity or one stream are refused when a data set is defined.

Alternatives. A registry of small integers per plant and run, as M0-E04 used, (plant index, run index): simple, but the same pair means different runs in different data sets, so merging two data sets generated with one seed would share noise without a word. The content hash as the identity: it cannot be known before generating and does not say what a run is. A random identifier per attempt: it makes repetition undetectable. Python's `hash()`: salted per process, so not reproducible.

M0-E04 keeps its own registered streams and is not affected.

## D-022 Process time is relative seconds and an integer tick (2026-09-21, decided by the implementation agent, technical and reversible)

`time_s` is process time in seconds on the clock of the run, a 64-bit float, identical to the instants of the simulation. `sample_index` is the integer tick of the sensor clock and is what joins, lags and gap detection use. No absolute timestamp is stored, and no table holds a time of creation: the virtual plants have no calendar, and a wall-clock column would make two generations of the same data differ. The time of generation is in the manifest of the data set.

This refines the sketch of `docs/architecture.md`, which listed `timestamp`, `created_at`, `start_time` and `end_time`. Alternative set aside: timestamps from an arbitrary origin such as 2000-01-01 UTC, which would add a time zone and a conversion, and would invite confusing the date of a file with the time of the process.

## D-023 Known inputs are channels of the long table; the quality flag describes a record (2026-09-21, decided by the implementation agent, technical and reversible)

The four inputs are stored with their full history in `measurements`, and `sensors` describes channels, with `channel_kind` equal to `measured` or `input`. An input has null noise fields, not zero ones, and is never presented as a noisy sensor. This widens the meaning of `sensors` and keeps the five entities. Alternatives set aside: input columns on `operating_runs`, which lose the history; extra tables; storing only the switching instants, which turns every alignment into a range join and hides a missing row.

`quality_flag` is 0 for a record whose value is present and finite, whose instant is finite and on the sampling clock, and whose channel is declared with a known unit. The writer refuses anything else, so version 1 of the contract holds no other value. It is distinct from the acceptance of the simulated truth, which is decided before anything is observed and whose diagnostics stay private, and from the selection of data, which is a query. No physical criterion is applied to readings.

## D-024 What is available to a model, and where it lives (2026-09-21, decided by the implementation agent, technical and reversible)

Everything a model may read is under `PT_DATA_DIR/available/`: immutable Parquet data sets, the DuckDB databases derived from them, and the exports derived from those. What is needed to regenerate and diagnose, seeds, noise streams, full configurations and the checks of the truth, is under `PT_DATA_DIR/private/`. The branches are siblings, no reader crosses from one to the other, and reading and exporting work without `private/`. It is a separation of content and code paths, not a permission of the operating system.

Known plant parameters are exported through an explicit list of fields of `PlantSpec`: volume, density, heat capacity, heat of reaction and the four nominal inputs. Configuration files are never copied to the available branch. The nominal values of the modeller's simplified model, k0, E/R and UA, are not known plant parameters and are not exported as such; whether E/R is known remains open and belongs to M1. Initial states and the nominal steady state, which come from the true model, are not exported.

DuckDB is a derived store: it is built from published data sets and can be rebuilt. Alternative set aside: DuckDB as the store of record, which would make the database file, inside a synchronised folder by default, the only copy of the data.

## D-026 The two other runs of D-010: definitions, identities and independence (2026-09-22, decided by the implementation agent, technical and reversible)

D-010 promised three operating runs per plant. P3 (D-019) is the third; the first two, steady operation with noise only and single-input step tests with 10 min holds, had never been generated. They are now defined in `simulation/protocols.py` next to P3, generated by the same path, and verified in M0-E06 and M0-E07.

Definitions. Steady operation: the nominal inputs held for 7200 s from the verified nominal steady state, one run per plant. Single-input steps: for each of q, C_Af, T_f and T_c and each direction, 600 s at the nominal inputs, 600 s with that input moved by its A10 amplitude while the other three stay nominal, 600 s at the nominal inputs; eight independent runs per plant, each started at the nominal steady state, the state carried across the three segments of a run and never across runs. They are not chained into one trajectory of eight steps, because the transitions between tests have not been studied and the safety of P3 does not transfer to them. What the holds and the returns do to the plants was verified in M0-E07 on the sixteen runs, not inferred from P3.

Identities. Neither protocol draws anything at random, so no excitation seed exists and none is invented: `<plant>.steady.d<seconds>.n<k>` and `<plant>.step.<input>-<direction>.l<lead>.h<hold>.r<recovery>.n<k>`, the input written as q, caf, tf or tc, the direction as up or down, the durations in whole seconds. The noise stream still follows from the identity (D-021), so the plants have independent noise, and the definition part still pairs the plants, so the source-target query and the figures needed no change. A data set definition names its protocol and carries the settings of that protocol only. Contract version 1 is unchanged: no table, column or encoding changed; the grammar of `run_id` is extended and recorded in `docs/data_contract.md`.

Alternatives set aside. One continuous trajectory of eight steps, compact but unverified in its transitions. A generic framework of protocols, which nothing in M0 needs. A single token for the three durations, which would not tell a change of the lead from a change of the hold.

## D-025 Minimum DuckDB version 1.3 (2026-09-22, decided by the implementation agent, technical and reversible)

`database.connect` turns DuckDB's external file cache off, after the stale read found by the first CI runs, and the setting that does it exists since DuckDB 1.3.0; `pyproject.toml` admitted any DuckDB from 1.0. The minimum is now 1.3. It was set from evidence and not from the setting alone: the whole test suite, which holds the storage and SQL path, Parquet, staging, the seven quality queries, the views, the analyses, the export and the scan, was run on a clean worktree of `6578f45` in one isolated virtual environment per version, with that DuckDB pinned and nothing of the registered environment touched.

| DuckDB | Result |
|---|---|
| 1.2.2 | refuses the setting, "unrecognized configuration parameter", and with it every test that opens a database fails, 40 or more |
| 1.3.0 | all 716 tests pass |
| 1.3.2 | all 716 tests pass |
| 1.4.5 | all 716 tests pass |
| 1.5.0 | all 716 tests pass |
| 1.5.5 | the registered environment, and CI |

The cache stays off in every version, and the regression tests of the stale read stay. Alternatives set aside: keeping `>=1.0` and skipping the setting where it is unknown, which would leave the ingestion exposed on those versions to the very read that was found; pinning 1.5.5, which would refuse versions shown to work.

## D-027 M0-E08's verdict must depend on the physical validity of the pair, not only anchoring and resolution (2026-09-22, decided by the implementation agent, technical and reversible)

`check_trajectory` already computed `A.accepted` and the individual physical checks of variant B, but `experiments/08_oracle_conductance.py`'s verdict and exit code read only H1 (anchoring) and H2 (numerical resolution). Codex reproduced the gap by forcing `balances_close=False`: `A.accepted` and `B.accepted` went false while H1 and H2, the only things gating the exit code, still held.

`TrajectoryCheck.physically_valid` (`simulation/checks.py`) is `accepted` without the envelope requirement: finite values, physical states, closed balances. `comparison_is_valid(primary, secondary)` requires the primary fully `accepted` and the secondary `physically_valid`, deliberately exempting the secondary from the envelope, since M0-E08's variant B exists to probe how far a change in the physics moves the plant and leaving [335, 380] K is a diagnostic result to report, not a validity failure. The script aggregates a third verdict, `H3_valid`, from this function on every case into the same `all(verdicts.values())` the exit code already used.

Alternative set aside: rejecting any case where B leaves the envelope, which would discard exactly the diagnostic M0-E08 was registered to produce (docs/experiment_log.md's expectation that a differently-behaved B might leave the envelope at the hot corner). The chosen criterion keeps that report while still rejecting a pair that is invalid on the grounds that have nothing to do with the envelope: non-finite values, unphysical states, or balances that do not close.

## D-028 A pinned reference environment, separate from CI's rolling dependencies (2026-09-22, decided by the implementation agent, technical and reversible)

`docs/m0_audit.md` already records that exact content reproduction (D-012) does not hold across numpy/scipy versions, and CI (`.github/workflows/ci.yml`) installs current dependencies on every run, which is right for testing the software but cannot reconstruct the registered numbers. Nothing pinned the environment the registered results were generated in.

`requirements-reference.lock.txt` pins the exact versions `pip freeze` reported for the registered environment (Python 3.13.7, Windows); `docs/reference_environment.md` states its scope, the two-step install (`pip install -r` the lock file, then `pip install -e . --no-deps` so the editable install does not re-resolve pinned versions against `pyproject.toml`'s looser bounds), the validation commands, and keeps the two claims apart: exact content reproduction, shown elsewhere not to hold across library versions, against numerical agreement across environments, which D-025's DuckDB matrix already demonstrates for one dependency. `pyproject.toml` is unchanged; this file does not replace it and CI does not install it.

Validated once, on the machine that captured it: installed into a clean virtual environment outside the repository and outside any synchronised folder, `pip freeze` there matched the lock file exactly aside from the local editable install, `ruff check .` and the 725 tests passed, and `experiments/08_oracle_conductance.py` exited 0 with H1, H2 and H3 holding, numbers identical to the registered run. No claim is made about a system, architecture or BLAS backend this has not actually been run on.

Alternatives set aside: freezing every package of a personal machine indiscriminately, which would pin unrelated tools and make the file harder to trust as a description of this project's environment; upgrading `pyproject.toml`'s bounds to the newest versions, which would change the dependency contract rather than record what was actually used.

## D-029 M1 begins: its question, comparators and exclusions (2026-09-22, accepted)

Decided by the project owner when M1 was started, the day M0 was closed.

The question: does a continuous-time hybrid model improve data efficiency or extrapolation over simple alternatives, when every model is fitted on data of the target plant only?

The comparators: the nominal mechanistic model, as the starting reference; the mechanistic model with its parameters re-estimated on the target; a black-box model trained on the target only; a hybrid model trained on the target only; and an oracle with the complete true physics, a separate diagnostic reference that is never a candidate.

Out of scope: pretraining on the source, fine-tuning between plants, transfer policies and systematic studies of domain shift, which belong to M2 and later.

Conditions. The hybrid is not assumed to win, and a well-supported negative result is a valid result. The registered values of alpha and the physical configurations of M0 are kept, and the difficulty is not adjusted to favour any method. Ordinary models never contain alpha, the saturating law or the true form of UA(T); a study that uses them is declared an oracle and kept apart. The data sets of M0 do not become the benchmark of M1 by default.

The benchmark itself, its task, data, budgets, metrics and the choices still open, is proposed in `docs/m1_plan.md` and waits for the owner's answers to the questions it lists. Nothing in that file is accepted by this entry beyond what is written here.

## D-030 M1: the owner's answers to questions Q1 to Q5 of the plan (2026-09-24, accepted)

The owner accepted the recommendation of each question of revision 1 of `docs/m1_plan.md`, section 15:

* Q1. E/R is estimated together with the other parameters of MR and of the hybrids. E/R held at 8750 K is a secondary analysis, for MR and HK at two budgets.
* Q2. Extrapolation is tested by training at half amplitude, A5, and evaluating at A10, once P3 at A5 has passed the verification of section 6.2 of the plan. The single-input steps are a secondary evaluation of a change of protocol.
* Q3. Budgets of 2, 5, 10, 20 and 40 excursions. Ten replicates as the initial proposal, to be confirmed from the cost of a fit before the benchmark is registered. Validation is a temporal hold-out inside the budget, there is no refit after selection, the hybrids start from MR_F, and MR and BL are fitted on the whole budget.
* Q4. The kinetic correction, HK, is the main hybrid; HU and HKU are variants for comparison.
* Q5. The primary evaluation is the rollout over windows with 110 scored readings (section 4.3 of the plan); rollouts over whole runs are a secondary evaluation. In a paired comparison, a failure loses to a completed evaluation and two failures tie.

What this entry does not fix, and later iterations must: the two budgets of the secondary analysis with E/R held fixed, and the number of replicates, both in the registration of the benchmark (I5), the second from the cost of a fit measured in I3; the training framework (I3); the grids of the learned models, the thresholds of the hypotheses and the rules for reading them (I5); P3 at A5, which exists only once its verification (I4) has passed.

## D-031 A Linux container for the data path of M0 (2026-09-24, decided by the implementation agent, technical and reversible)

Asked for by the owner, as a technical iteration before I1: the data path of M0 must build and run in a Linux container on the CPU, with explicit inputs and with results that outlive the container, without anything of M1. The details, the commands and what was checked are in `docs/docker.md`.

* The base image is `python:3.13.7-slim-trixie`, pinned by digest: the interpreter of the registered environment of M0, on a Debian whose C library, glibc, has a wheel of every package of the lock.
* The image has its own lock, `docker/requirements.lock.txt`: the runtime packages at the versions of `requirements-reference.lock.txt`, which is not modified, plus setuptools, each pinned to the hash of its Linux wheel for CPython 3.13 and installed with `--require-hashes --no-deps`.
* The checkout is copied through a whitelist, `.dockerignore`, and installed in editable mode, because the SQL and the configurations are read next to `pyproject.toml`.
* `PT_DATA_DIR` is `/data`, and the entry point refuses to run, with exit code 3, unless it is a mounted directory, so that a data set is never written into the container and deleted with it.
* A user without administrator rights runs the container; the code is read-only for it.
* Neither `.git` nor git is in the image. Runs made there record that the code could not be identified, as the provenance of M0 does whenever git cannot answer, and no commit is written into their results.

Alternatives set aside. The Windows reference lock as the dependencies of the image: it was never tested on Linux, and it holds development tools and a package only Windows uses. A wheel installed without the checkout: the database would not find its SQL, and that way of installing has not been built or tested. `.git` and git in the image: the files of the image differ from the commit on purpose, so git would report changes, and its ownership check refuses a repository owned by another user. A `VOLUME` in the Dockerfile: `docker run --rm` deletes the anonymous volume with the container. Alpine as the base: PyPI has no wheel of DuckDB 1.5.5 for its C library, musl, so DuckDB would have to be compiled from source (checked on 2026-09-24; the other packages of the lock do have musl wheels). Docker Compose: there is one process and nothing to compose.

Checked on 2026-09-24 (`docs/docker.md`): the ten checks of the generator pass in the container on the definitions of M0-E06, and experiments 05 and 06 hold all their hypotheses there. Content hashes are not equal to those registered on Windows in seven of eight runs, with the same library versions; the readings agree to the precision of the integrator. A data set generated in the container is therefore not a copy of a registered one, and the policy on repetition refuses to put it in its place.

Clarification, 2026-09-25, after Codex's independent audit of `01c6422` on 2026-09-24. The fourth point above overstates what the entry point does. `mountpoint` only asks whether something is mounted on `PT_DATA_DIR`, so the check refuses a run with nothing mounted there and no more: it cannot tell a mount that keeps the results from one that does not. Codex reproduced two runs that pass it and lose the results. On a tmpfs mount the pipeline ran, and its results were temporary. On an anonymous volume (`-v /data`) with `--rm` the run generated the data set and ended with exit code 0; Docker then removed the volume, and Codex verified that it was gone. Results are kept by a folder of the host or by a named volume; `docs/docker.md`, "Keeping the results", gives both with the permissions each needs, and says why tmpfs and anonymous volumes must not hold results to keep. The comments and messages of `docker/entrypoint.sh`, a comment of the `Dockerfile` and `docs/docker.md` now describe the check as a guard against a forgotten mount. The check, its exit code 3 and the data path are unchanged, and the text of this entry above is kept as it was decided. The same audit, recorded in `docs/docker.md`, confirmed the other points of this entry that it checked: the hashes of the lock, the files copied into the image, the user without administrator rights, and the provenance recorded as unknown.

## D-032 I1 of M1: how the evaluation contract and the mechanistic models are built (2026-09-25, decided by the implementation agent, technical and reversible)

The choices made while implementing I1 (`docs/m1_plan.md`, sections 13 and 14) where the plan left the how open. None changes the design of the plan. The one the owner may want to decide otherwise is marked.

* The rollout is written on the model side. `simulation.integration.simulate_piecewise` restarts the integrator at every change of the inputs, which is what a rollout needs, but `models` may not import `simulation`. Its loop is written again in `models/rollout.py` instead of being moved to a neutral module: a rollout must return failures as records instead of raising, limit the evaluations of the right-hand side, sample only the sensor instants and integrate sensitivities, and the simulation needs none of that; moving it would have changed audited code of M0 for a caller that uses little of it. The cost is two loops of about thirty lines doing the same thing. A test checks that they agree on the same equation.
* The reference integration of the evaluation is LSODA with rtol = 1e-8 and atol = 1e-6 in the SI unit of each state and sensitivity, with a guard of 100 000 evaluations of the right-hand side per window. Its error against LSODA and DOP853 at rtol = 1e-12 is below 3e-5 sigma, on the 16 corners of P3 in the tests and on real windows of M0 in the smoke run, against the 1 % the plan asks.
* Windows are found from the inputs, with the layout of the protocol that the export states as the operating mode of the run. Inputs are compared with the nominal inputs exactly, since they are the same floating-point numbers. A run that breaks its layout is refused, not read with other phases. A budget is counted in windows: on the P3 runs of M0, which have no lead, b windows span b + 1 excursions.
* The validity bounds use the inputs applied before each scored instant, since the row at a scored tick acts only after it. The temperature bound is implemented for dH <= 0 and refused for an endothermic reaction, which no plant of the project has.
* The sign of the implied rate and heat flow is judged only beyond the rounding bound of the expressions that form them: gamma_n times the traffic through each balance, n = 4 and 12 (Higham's bound for sums of products). T = T_c is recognised exactly, and a residue of rounding there is not counted as a heat flow.
* MSE - sigma^2 is computed only for readings declared held out from the model. Every evaluation states what its readings were to the model: fitting, selection or held out.
* The Jacobian of a fit comes from forward sensitivities integrated with the states, not from finite differences, which with an adaptive integrator would difference its error along with the model. The optimiser works on (ln k_350, (E/R) / 350 K, ln UA), with E/R >= 0 as a bound, `x_scale="jac"`, ftol = xtol = gtol = 1e-10, 100 evaluations per start, and every option of `least_squares` given explicitly. The starts are the textbook values, then k_350 x 2 with UA / 2, k_350 / 2 with UA x 2, and E/R -2000 K and +2000 K; with E/R fixed, the first three.
* A failed rollout ends its start. The TRF method happens to treat a trial point whose residuals are not finite as a rejected step, but that is not documented, and the plan asks that a failure the procedure does not treat end the fit.
* For the owner: only starts that converged, status 1 to 4 of `least_squares`, can be selected. A start that exhausted its budget is recorded with its endpoint and not selected, and if no start converged the fit is a training failure. The plan keeps the lowest objective among the declared starts and does not say what to do with an endpoint that met no criterion; selecting among every finite endpoint instead would let a fit be reported at a point that is not a solution. In the smoke run every start converged.
* The covariance of an estimate is the sandwich of section 7.4 of the plan. It is not computed when the Jacobian is rank deficient by numpy's convention for the numerical rank; a Jacobian that is deficient only to the accuracy of the integration gives very large standard errors, which its singular values show.

Checked on 2026-09-25: the tests of I1, the smoke run on the exports of M0 and the check of the calibration of the sandwich errors (`docs/experiment_log.md`, development runs of I1).

Clarification, 2026-09-25, after Codex's audit of `124c377` to `a8905dc`. Two points of this entry were stated without their limits, and the audit showed what that let through (`docs/numerical_robustness.md`, twelfth review). The domain of a fit: what makes the loss undefined for any model is refused before any start, with a ValueError, now including scored readings whose values divided by the noise levels are not doubles; whatever depends on the point a start has reached, including residuals, Jacobian, loss or gradient that are not doubles, is a numerical failure of that start, recorded with its cause, while the other starts go on. Before, the first case escaped from least_squares as an exception, and readings of 1e160 let every start end as converged with an infinite objective. And the implied terms and the metrics are judged or returned only when they are doubles: squares and means are formed on values scaled by their largest magnitude, and what is not representable is refused, never returned as inf or as zero and never counted as compatible with the physics. The selection among converged starts, the loss, the definitions of the metrics and the rounding bounds are unchanged.

Clarification, later on 2026-09-25, after Codex's review of `9ea9a76`. The optimiser's own arithmetic can overflow on quantities that are all doubles: with `x_scale="jac"` it squares the elements of the Jacobian to scale its steps, and an infinite norm made a scale zero and a start converge by its tolerance on the step without moving. `least_squares` now runs with numpy's floating-point errors raised, and such an error ends the start as a numerical failure, like any other point the loss cannot be computed at; the loss also checks the norms of the columns of the Jacobian. The method, its scales and tolerances, the starts and the selection among converged starts are unchanged (`docs/numerical_robustness.md`, thirteenth review).

## D-033 I1 of M1 accepted and closed (2026-09-27, accepted by the owner)

The owner accepted I1 of M1, the evaluation contract and the mechanistic models MN, MR and MR_F, and authorised its closure on 2026-09-27, once its criteria of acceptance (`docs/m1_plan.md`, section 13) had been checked against the repository. Its code is that of `96945a6`, the last commit that changed it, and its tests are those of `d8bc372`, which corrected one test found by the checks below; the commit that records this entry changes documentation only. Closing I1 does not close M1. I2 is the next iteration.

Who did what, kept apart.

* The owner accepted I1 and authorised its closure. During the corrective iteration the owner had asked that the rule of D-032 marked for the owner be kept: only starts that converged are selectable. It stands.
* Codex, as read-only reviewer, made three reviews:
  * an audit of `124c377` to `a8905dc`, which reported six findings (`docs/numerical_robustness.md`, twelfth review);
  * a review of `9ea9a76`, which reported two points (thirteenth review);
  * a last review, focused on the two corrections of `96945a6`. It inspected them, ran the tests of the fit and of the metrics, which passed, and found no new reason to extend the corrective iteration.

  That last review was not a new audit of the whole of I1. Codex did not check the continuous integration of `96945a6` itself.
* Claude Code, as implementation agent, did the following:
  * It reproduced every finding and corrected it with a regression test.
  * It ran the suite on the code of `96945a6`. On Windows 855 tests passed. In a container of the image on Linux 851 passed, and 4 were skipped because the image has no git.
  * It repeated the smoke run of I1 after the last correction, from the working tree before the commit. The 11 444 numbers of its summary equal those of the run before it bit for bit.
  * It read the continuous integration of `96945a6` on the Actions page of the repository, run 36171775881: test (3.12) and test (3.13) passed 855 tests each, and the docker job succeeded.
  * On 2026-09-27, before this entry, it checked each criterion of section 13 against the tests again, with read-only sub-agents of its own, and ran the suite at `96945a6`: 855 passed, and ruff was clean. What that check found is below; one test was corrected in `d8bc372`, and the suite passed again after it.

Where each criterion is met:

* The worked example of section 5.7 and the counts of section 5.4 at every budget: `tests/test_evaluation_budgets.py` and `tests/test_evaluation_windows.py`.
* The metrics against values computed by hand: `tests/test_evaluation_metrics.py`.
* The error of the rollout below 1 % of sigma, against LSODA and DOP853 at rtol = 1e-12: `tests/test_models_rollout.py`. The largest error found there is below 4e-5 sigma, and 2.5e-5 sigma on real windows in the smoke run.
* The recovery of the parameters: without noise from an exact initial state, and with the noise of D-020 and the rule of the context within four sandwich standard errors for five declared seeds. Both are in `tests/test_models_fitting.py`. The calibration of those standard errors over 200 seeds was last repeated at `b5c2838`; the two commits of I1 after it change only what happens when a number is not representable.
* The import graph of `models` and `evaluation`: `tests/test_boundaries.py`. That a fit of MR_F does not read V: `tests/test_models_fitting.py`.

What the checks of 2026-09-27 found beyond the criteria. None of it is a defect in the scientific results of I1. The first item was corrected before the closure. The others do not block it and are recorded so that later work can strengthen the tests where it touches them. The addendum of 2026-09-27 to `docs/numerical_robustness.md` has the details.

* A test compared the rounding bound of the implied heat flow, about 1.2e-13, below the absolute tolerance that `pytest.approx` allows by default, so a bound of zero or eight times too large passed. It is the mechanism of the second point of Codex's review of `9ea9a76`. It was corrected in `d8bc372`, and a run of the suite that recorded every such comparison found no other.
* The noise-free recovery test asks 1e-5 of theta and of J. The optimiser's tolerances are 1e-10, and the accuracy reached, 2e-7 in (E/R) / 350 K, is set by the integrations, not by the optimiser; "to the tolerance of the optimiser" in the criterion and in the name of the test says more than the test asks.
* The noisy recovery test fits from the textbook start alone, not from the five declared starts.
* Several paths are not reached by any test:
  * the integration failures "integrator" and "non-finite state";
  * a physical violation carried beside a score in the tables of failures;
  * per-window and per-phase scores whose windows or phases differ, and the excess over the noise inside them.
* The test of the import graph reads only import statements that name a module of the simulation or of the generator, so it would not see `from process_transfer import simulation`. No such import exists.
* With an integrator other than LSODA, which D-032 fixes for the evaluation, a rollout whose right-hand side is huge but finite can raise instead of returning a record.
* The thirteenth review gives the objective and the endpoint of the silent convergence with numbers that the window of its regression test does not reproduce; the defect and its correction are as described. An open limitation of the eleventh review had been made obsolete by the twelfth.

Carried forward, documented and not blocking:

* the open limitations of the eleventh to thirteenth reviews of `docs/numerical_robustness.md`, among them:
  * a `LinAlgError` of LAPACK is not caught;
  * a floating-point error inside `solve_ivp` is reported differently inside and outside a fit;
  * LSODA's `UserWarning` is printed as SciPy prints it;
  * a subnormal result keeps only subnormal precision;
  * I3 has to decide how a window whose implied terms cannot be computed is counted;
* the difference between the platforms in the test of a steady state that leaves one direction undetermined, recorded in the twelfth review;
* the test of the exit code of M0-E08, carried from M0 (`docs/roadmap.md`).

The numbers of the development runs of I1 remain development numbers and are not results of M1.

## D-034 I2 of M1: how M1-E01 is computed (2026-09-27, decided by the implementation agent, technical and reversible)

These are the choices made in implementing the diagnostic of section 7.4 of the plan, where the plan leaves the how open. The registration of M1-E01 in `docs/experiment_log.md` states them in full; this entry says why, and what was set aside. None of them changes the design of the plan, and none reopens Q1 (D-030). Two read-only reviews by agents of the implementation, before the registration, shaped several of them; the registration lists what those reviews computed.

* The oracle has a script of its own, `experiments/m1_e01_oracle.py`, where the plan asked for one script. With two scripts, a test on the import graph can show that parts 1 and 2 never import the simulation, the generator or the private branch. In one script only a reading could show it.
* A third covariance joins the two the plan names. Beside the covariance with the initial state exact and the sandwich of the estimator used, part 1 reports the Cramer-Rao bound with the initial state known through its ten context readings: the inverse of the Schur complement of the joint information of the parameters and the initial state, (S^T Sigma^-1 S)^-1.
  * The two quantities of the plan alone could not say whether the cost of the rule for the initial state is information lost or the price of the estimator's rule; the bound splits the two.
  * It is the bound for estimators that leave each initial state free, not for one that ties it to the model's steady state.
  * It needs nothing new: the same sensitivities, and a 2 x 2 solve per window.
* The point of evaluation. The steady state is the mean of the steady run of `m0-e06`, which M0 generated to verify steady operation and which serves this purpose with 1201 readings; the mean of the contexts of `m0-e05` is a cross-check. E/R is the modeller's textbook value, and k_350 and UA follow from the two balances in closed form. The E/R of the pooled free fit of part 2 gives a second point, because information is local and one point alone would hide how much it depends on the point. Set aside:
  * the mean of the contexts as the primary estimate: 270 readings against 1201, each taken at the end of a rest with a residue of the excursion before it;
  * the estimates of the smoke run of I1 as the primary point: they are development numbers of one run each, and the plan asks for the steady state.
* The designs of part 1 count the windows a fit receives: b for MR, and b - n_V for MR_F. For each count there are two designs:
  * the expected design, n / 16 windows at each corner, whose information is the mean information of n windows drawn by P3. For the covariance with the initial state exact and for the context bound it is therefore a lower reference for the mean over the draws, by Jensen's inequality for the matrix inverse; for the sandwich it is a reference with no guaranteed order;
  * 20 000 draws of the corners, with replacement and a declared seed, which give the spread and are what small budgets are read from.

  Beside them are the designs of the corners the runs of part 2 actually visited, at 9 and 27 windows. The windows of part 1 start exactly at the steady state; the residue an excursion leaves after a rest is of millikelvins (M0-E03b). Set aside: an exact enumeration of the draws. It is possible for small n, but not for n = 40, and the 20 000 draws are close enough for the quantiles reported, about 0.15 points in the probability level of a 5 % quantile.
* The resolution of part 1. The standard errors of every corner window at the settings of the evaluation must agree with those of LSODA at rtol = 1e-12 to 1e-3 relative. The sensitivities at those settings are bounded by the tests at 1e-5 of their scale (`tests/test_models_rollout.py`), and measured at 4e-8 to 8e-8, so 1e-3 leaves a wide margin while keeping the reported values to about three significant digits.
* The grid of the profile has a wide part, 6000 to 13000 K in steps of 500 K, for the shape, and a fine part, 9000 to 10000 K in steps of 25 K, for the minimum.
  * The fine part was placed where the smoke run of I1 had found the estimates of MR on the same data. The registration says so, and the minimum is located by the free fit, not by the grid.
  * The fits held at a point of the grid are those of MR (D-032) with its three starts that do not move E/R, so that the profile is the objective of MR and nothing else.
  * The sharpness of the minimum is read from brackets that hold under convexity alone, not from the spacing of the grid (the registration, B2).

  Set aside:
  * a fine grid centred on each free estimate by a declared rule, which would resolve every minimum the same way, but whose points no one could name before the run;
  * a single uniform grid of 25 K over the whole range: 281 points instead of 53, about five times the cost for no gain in shape;
  * starting each held fit from its neighbour on the grid, which would make each point depend on the path through the grid.
* The rule for reading the spread of E/R between runs divides each difference by the combined sandwich standard errors at the two estimates, the estimator's own. The a priori standard errors of each run's design are reported beside them as context. A pair is read only if both runs' profiles agree with the linearised curvature, since the sandwich rests on it.
* The oracle's kinetic variant keeps the true E/R and the true conductance, and takes k0 / (1 + K_sat C_A,nominal) with K_sat = 0.
  * Its rate then equals the true rate at the nominal steady state, and the steady state is shared: the anchoring M0-E08 used for the conductance.
  * Variant B of M0-E08 is recomputed on the same 16 windows, so that the two mismatches are compared on the same windows.
  * Set aside: anchoring the kinetic variant by fitting it to the trajectories, which would make it a second pseudo-true fit instead of a mismatch alone.
* A precision on section 7.4 of the plan. It says the pseudo-true value is not the large-budget limit of the benchmark partly because the benchmark's corners are drawn with replacement and its fits use prefixes of runs. Neither acts in that limit, since uniform draws tend to equal weights and prefixes do not change a limit. What does separate the two is the error of the context means, the state an excursion carries into the next, and, at finite budgets, the bias and variance of the estimator. The registration uses this reading. The plan's text is kept and the precision recorded here.
* The fits of part 2 run in separate processes, one fit per task, each built from the exports themselves. A fit depends on nothing but its windows, its E/R and the declared settings.
* Both scripts refuse to run when PT_DATA_DIR resolves inside the repository, whose data directory is the default. The outputs of the oracle hold hidden parameters, and the outputs of both belong outside git and outside OneDrive.

Clarification of 2026-09-27, after Codex's review of `129063c`. Two points of the implementation above are corrected (`docs/numerical_robustness.md`, fifteenth review). The choices themselves stand, and the experiment log records that the corrected script reproduces the registered results.

* The context bound is the same quantity, the inverse of the Schur complement. It is no longer computed by Woodbury's identity with a 2 x 2 solve, whose matrix Sigma_0^-1 + G^T G could overflow while each term was finite and drop the cost of the initial state. The initial state is counted in standard deviations of its context mean, and the Schur complement is read from a QR factorisation of the joint problem.
* "Brackets that hold under convexity alone" is kept, with what it implies made explicit. The points of a profile cannot show that it is convex between them; slopes that do not decrease are consistent with convexity and do not demonstrate it. A bracket is reported only where the sampled slopes over its span do not contradict convexity, and is labelled as conditional on it. Elsewhere it is not evaluable, B2 gives that profile no verdict, and B4 reads no pair that holds its run.

## D-035 The training framework of M1: JAX (2026-09-27, decided by the implementation agent, technical and reversible)

The plan leaves the framework to I3, chosen from a short comparison of two candidates on development data (section 8.6). The owner authorised I3 and asked that the choice follow the evidence on these criteria:

* the correctness of gradients;
* the treatment of changes of the inputs;
* reproducibility on a CPU;
* compatibility with Windows, Docker and Linux, and the Pythons of CI;
* the cost of installing, integrating and training;
* the clarity of the equations.

It was not to be chosen for popularity, and a GPU was not to be assumed to help.

Compared: JAX 0.11.2 with diffrax 0.7.2, and PyTorch 2.14.0 on the CPU with torchdiffeq 0.2.5. `experiments/m1_i3_framework_comparison.py` ran both at `28e0936`, on the nine windows of one target run of `m0-e05`; the experiment log has the method and the numbers ("Development runs of I3").

Decision: JAX, with a fixed-step fourth-order Runge-Kutta scheme written in the repository (`models.training`) for the rollouts of training. diffrax is not a dependency. The dependency is `jax`, with jaxlib, ml_dtypes and opt_einsum, as the optional extra `learning`.

Why:

* Correctness does not separate them. Both computed the same losses to 1e-15, gradients that agree with central differences to 3e-7 or better at a generic point, and the same trained weights bit for bit twice and across processes.
* Cost does. On one CPU thread, a loss and its gradient of HK took 7 ms in JAX and 540 ms in PyTorch, and of BN 23 ms and 184 ms. JAX compiles the loop of a rollout once; PyTorch dispatches each of its many small operations. A benchmark of about a thousand trainings of thousands of steps each makes that ratio decisive.
* Installation: 292 MB against 652 MB with PyTorch's CPU index. On Linux, PyTorch from PyPI's default index would also bring the CUDA libraries unless CI named the CPU index.
* Both write the equations as numpy does, and `models.learned` writes them once for numpy and JAX alike, so the model trained and the model evaluated are the same equation.
* Windows with Python 3.13 was measured here; Linux with Python 3.12 and 3.13 is CI's job. A jaxlib wheel exists for each.

Why a fixed step and not an adaptive solver in training:

* At MR on real windows the fixed-step scheme, two steps per row, differs from the reference rollout by 5e-4 sigma, and diffrax's adaptive Tsit5 at 1e-8 by 2e-5 sigma, at five times the cost for HK.
* Each row is integrated with its own inputs, so no step crosses a change of the inputs, and the gradient is the exact derivative of the loss computed.
* What is selected and evaluated is the continuous-time equation, rolled out by the reference integration of `models.rollout`, whatever scheme trained it. Every checkpoint records the largest difference of the two rollouts on V, so that a model whose training scheme drifts from its equation is seen.

Set aside:

* PyTorch, for the cost above.
* CasADi, not measured. It would write each network as a symbolic expression and estimate with IPOPT. The equations would then be written in its symbolic language, and a rollout of 110 rows of several steps becomes a large expression graph for every window.
* SciPy with sensitivities written by hand, not measured for networks. It serves MR and BL, whose parameters are few. For a network it integrates two extra states per parameter, several thousand for BN, or needs an adjoint written by hand.
* A GPU, not tried. The batches hold at most 32 windows of two states, and the steps of a rollout run in sequence.

Consequences:

* Importing `models.training` sets JAX to double precision for the process.
* JAX changes quickly between versions. `pyproject.toml` states a lower bound, as it does for the other dependencies, and CI tests the current versions. The reference environment adds the versions measured here to its lock file.
* The image of `docs/docker.md` runs the data path and does not install the extra. A run of the suite in a container adds it, as it adds pytest.

## D-036 I3 of M1: how the learned models are built and trained (2026-09-27, decided by the implementation agent, technical and reversible)

These are the choices made in implementing BL, BN, HK, HU and HKU, where the plan leaves the how open. The rate of learning, the number of steps and the lists of configurations are not decided here: the development runs of I3 inform them (experiment log), and they are fixed before the benchmark (section 5.5). None of these choices changes the design of the plan.

The equations (`models.learned`):

* They are written once, on an array module that is numpy or `jax.numpy`. The model trained with JAX and the model evaluated by the reference rollout are the same equation computed by the same operations, not two codings to be kept in step.
* Each factor of a hybrid is exp(N), N a network with tanh hidden layers and a linear last layer. It is positive whatever the weights and one when the last layer is zero, which is how every network starts. Set aside:
  * 2 sigmoid(N), bounded in (0, 2), which would put a ceiling on the correction that nothing in the physics sets;
  * softplus, which is positive but not one at a natural point without an offset.
* An exponential that overflows is a failure, never a clipped value. In the reference rollout it is a failed right-hand side; in training, a loss that is not finite.
* The hidden layers start from Glorot's uniform law with zero biases, the last layer from zero. The gradient then reaches the last layer from the first step, and the hidden layers from the second; a test checks both. BN also starts with a zero last layer, as a model whose state does not move.
* The arguments of the factors are the plan's (section 8.2): C_A and T for the kinetic factor, T and T_c for the thermal one, each centred and scaled by F. BN reads the six variables so scaled, and its output is scaled by D = s_x / 60 s. The plan asks for D computed from F; s_x is, and 60 s is a declared constant, the order of the plant's slowest time constant. It sets only the size of an output of order one.
* The mechanistic parameters of a hybrid are MR's, in MR's coordinates, trained with the networks from MR_F. E/R can be held, for the secondary analysis of D-030.

The penalty (`models.training`):

* lambda times the sum of the squares of every weight and bias of the networks, added to J_F^2. The mechanistic parameters are not penalised, as MR's are not.
* The penalty is zero only where every weight is zero, where each factor of a hybrid is one and BN does not move. For a hybrid it therefore pulls toward the mechanistic model it started from. The last layer bounds the departure from the identity everywhere, not only where the data are: |ln g| <= sum |W_L| + |b_L|, since tanh lies in [-1, 1].
* Set aside: a penalty on ln g at the states of F only. It says nothing about extrapolation, which is where the hypotheses of the plan look.

The training (`models.training`):

* Rollout: a fixed-step fourth-order Runge-Kutta scheme, two steps per row, each row with its own inputs, as D-035 explains.
* Optimiser: Adam, written out in the module with the usual constants, beta1 = 0.9, beta2 = 0.999 and epsilon = 1e-8, and a constant rate. Nothing clips a gradient.
* Checkpoints: the step 0 and every `validation_every` steps to `max_steps`, with no early stop. At each checkpoint:
  * the criterion is J on V by the reference rollout, as the plan asks (section 5.5);
  * the largest difference between the rollouts of the training scheme and of the reference on V is recorded.
* Selection of a checkpoint: the lowest criterion among checkpoints whose rollouts of V all completed, the earliest on a tie. Selection among configurations: the lowest selected criterion among those that did not fail, the first in the declared order on a tie. The first checkpoint of a hybrid is MR_F with its factors at one, so a hybrid that does not improve on V ends as MR_F (section 5.5).
* Failures, following section 8.7, which lists a loss that is not finite as a training failure:
  * a loss or gradient that is not finite at any step ends the training as a failure of that configuration, and its earlier checkpoints are not used;
  * a scale of F that is zero refuses the training before any step;
  * if every checkpoint has a failed rollout of V, the training is a failure.

  The registration may choose otherwise for the first case, keeping the checkpoints before the failure; it would then be a rule declared before any benchmark fit, not a remedy.
* Seeds: the initial weights of a configuration on a replicate come from `numpy.random.SeedSequence([base, replicate, configuration])`. They are the same at every budget of the replicate, so its budgets start from the same weights. The base is fixed in the registration.
* `hold_mechanistic` trains the networks of a hybrid with its mechanistic parameters held. It exists for the test of recovery of a planted correction (section 13); the procedure of M1 trains them together.

BL (`models.linear`):

* B is estimated on the span of the levels of the inputs its data excite, and is zero on the complement (section 8.7).
  * The level of an input is its deviation from nominal over its amplitude in the data of the fit, the largest deviation seen, so that P3's levels are the signs of its corners.
  * The rank and a basis of the span are computed exactly on the rational levels; the basis is then made orthonormal in floating point.
* The fit is MR's: `least_squares` with the sensitivities of the reference rollout, on all b windows, from declared starts, keeping the converged endpoint with the lowest objective.
  * The starts are x_e at the mean of the initial states and A = -I / tau for tau of 60, 20 and 180 s, with C = 0.
  * The coordinates are made without units: states in noise levels, time in units of 60 s.
  * The trust region is not rescaled by the Jacobian. On noise-free data of a linear plant the rescaled region left the basin of the true values for a stiff local minimum, and the unscaled one reached them in 11 evaluations (`tests/test_models_linear.py`).
* `fitting.WindowLoss`, the loss of MR, takes the model at theta from a builder, so that BL uses the same loss; MR's fits are unchanged bit for bit.

Clarification of 2026-09-28, after the pilot of I3 (experiment log, "Development runs of I3"). The pilot ran with two steps of the training scheme per row. On a synthetic run hotter than the target, the selected checkpoints of HK and HKU differed from the reference on V by about 0.05 sigma, at windows where the model's fastest eigenvalue is 0.48 times a step. Four steps per row keep those windows within 3e-3 sigma. The default of `TrainingSettings` stays at two, as the pilot ran; four is proposed for the registration of the benchmark, which fixes it. Four steps double the time of the steps, and the estimate of the benchmark includes them.

Clarification of 2026-10-01, after Codex's audit of `1a9fad4` (`docs/numerical_robustness.md`, seventeenth review). The choices above stand, with these corrections and precisions:

* The domain of E/R. Adam could take E/R below zero, a model MR's fit never reaches, since its bound keeps E/R >= 0. Each step is now projected onto E/R >= 0, and `learned.LearnedModel` refuses parameters outside the family's domain, so training, evaluation and storage share it. The projection is the constrained method of the declared domain, not a remedy: it reaches the valid limit E/R = 0 exactly, where a rate does not depend on temperature, and inside the domain it changes nothing. Set aside:
  * a change of coordinates, E/R as exp or softplus of a free coordinate, which cannot reach the limit, or as a square, whose gradient vanishes there. Either would also change the path of every training of a hybrid, the pilot's included, far from the bound;
  * a negative E/R admitted for the hybrids, a change of the scientific domain from MR's that nothing in the plan asks for.
* The state of Adam and the criterion on V. Failures of the optimiser's state end a training like a loss that is not finite. The criterion is J as the evaluation computes it, and only a finite criterion is selected.
* The penalty, stated more precisely. Penalising the networks pulls each factor toward one, and bounds its distance from one everywhere through the last layer. It does not anchor the hybrid to MR_F: the mechanistic parameters are trained with the networks and are not penalised, and they can move away from MR_F's values as far as the data pull them. In the development runs that set the tolerances of `tests/test_hybrid_recovery.py`, a joint fit of HK on planted data moved k_350 by 7 % from the value that simulated them while it recovered the total rate.
* Selected models are stored by `models/persistence.py`: one JSON file with everything needed to rebuild the model and say how it was obtained, and a SHA-256 of its content checked on reading. This is the record of one model, for the freeze of I6, not a registry.
* Four steps per row, checked by complete trainings (experiment log, after the audit).
  * On the case of M0, the selected checkpoint was the same. The difference from the reference fell from 3.6e-4 to 2.9e-5 sigma on F and stayed near 2e-5 on V.
  * On the fast synthetic case it fell from 0.059 to 0.0028 sigma on F, and from 0.055 to 0.0026 on V.
  * A training took about 1.7 times as long.

  Four steps remain the proposal for the registration; the default of `TrainingSettings` stays at two, as the pilot ran.
* A training that fails at a step still discards its earlier checkpoints. Any other policy is for the registration, before any benchmark fit.
