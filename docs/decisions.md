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
