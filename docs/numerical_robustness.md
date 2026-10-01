# Numerical robustness

The rule itself lives in `AGENTS.md`, under Numerical Robustness. This file is the record: what has been examined, which failures were reproduced, how they were resolved and what is still open. A new review adds a dated section; it does not rewrite the earlier ones.

One mechanism enforces part of the rule on every test run: `pyproject.toml` turns every `RuntimeWarning` into an error, so a division by zero or an invalid operation inside numpy, which would otherwise only print a warning and carry on with `inf` or `NaN`, fails the test that triggers it.

## Review of 2026-09-21: the M0 modules as they stood at `50dc509`

Scope. A bounded review of the interfaces that exist today: `units`, `config`, `simulation/cstr_true`, `modeller/cstr_first_order`, `simulation/steady_state`, `simulation/integration`, `simulation/envelope`, `simulation/excitation`, `simulation/balances`, `simulation/checks` and `data/paths`. For each: denominators, exponentials, interpolation, empty sets, grid generation, and the result of conversions and products, not only their inputs. It is not an enumeration of every possible number; it looks at inputs that the current interfaces can receive and at operations that fail without saying so. Every candidate was run against the code before anything was changed, and only reproduced failures were fixed.

### Failures reproduced and fixed

Two were found by the reviewer of `50dc509`, one while testing protocol P3, the rest by probing the interfaces.

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
| `integration.simulate_piecewise` | any nonlinear system | the first stored sample of a segment came from the solver's interpolant at t = 0 and differed from the last sample of the previous segment in the last bit, so a switching instant was stored with two states | yes | the first sample of a segment is its initial condition, by definition | `07bcdc1` |
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

## Review of 2026-09-21, second: structure of a trajectory and identification of the code

Scope. The two defects found in the review of `325cb93`, and the rest of the two interfaces they belong to: how a `Trajectory` is built, and how `data/provenance.py` reads git. Every row below was reproduced against the code at `325cb93` before it was changed.

### Failures reproduced and fixed

| Module | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `integration.Trajectory` | two pieces simulated apart, each at rest at its own steady state, placed side by side | the state jumped by -67.09 mol/m^3 and +6.86 K where they met, and `check_trajectory` accepted it, because the balances are closed inside each segment | yes | the state at the start of a segment must equal the state at the end of the previous one, checked when the trajectory is built | `875e83a` |
| `integration.Trajectory` | a second segment moved 30 s later, or 30 s earlier | accepted; with the overlap `Trajectory.times` was not even monotonic | yes | consecutive segments must share their switching instant | `875e83a` |
| `integration.SegmentTrajectory` | sampling instants in reverse order, or repeated | accepted, with time running backwards | yes | instants must be finite and strictly increasing | `875e83a` |
| `integration.SegmentTrajectory` | a matrix of inputs; segments with different numbers of states or inputs | accepted, and failed later with a shape error or not at all | partly | one constant input vector per segment, the same sizes in every segment | `875e83a` |
| `integration.Trajectory.duration` | a run of segments cut out of a longer trajectory | returned the final instant: 120 s for a piece lasting 60 s | yes | the time between the first and the last sample | `875e83a` |
| `provenance.git_state` | `git status` failing with an empty output | read as a clean working tree: `dirty = False`, `code_identified = True` | yes | exit codes checked; the state of the tree is recorded as unknown and the commit is kept | `9603ade` |
| `provenance.git_state` | `git diff` failing | the SHA-256 of an empty output was recorded as the fingerprint of the changes | yes | no fingerprint, and the reason says so | `9603ade` |
| `provenance.git_state` | `git rev-parse HEAD` answering with something that is not an object name | recorded as the commit | yes | a commit is 40 or 64 hexadecimal digits, or it is not recorded | `9603ade` |
| `provenance.new_run_directory` | a commit with an unknown working tree | the run id carried no mark and looked like a clean run | yes | the mark `-unverified` | `9603ade` |

### Exact comparison at a junction, on purpose

The rule of this file forbids an arbitrary epsilon, and here none is needed. The switching instant is one instant and the state there is one state, stored twice: as the last sample of a segment and as the first of the next. `Trajectory.times` and `Trajectory.states` keep a single copy, which is only right if the two are the same number. `simulate_piecewise` makes them so by construction: the first sample of a segment is its initial condition, which is the last sample of the previous segment, and both instants are the same floating-point sum. A tolerance would need a scale for times and another for each state, and would mean choosing between two different values without saying so. A difference of one unit in the last place is therefore rejected, and a test pins that. A trajectory assembled from pieces is made valid by starting each piece from the final state of the previous one, not by loosening the comparison.

### Valid cases, kept valid

* inputs that change discontinuously at a junction, which is what a junction is for; the sample taken there carries the inputs applied from that instant on;
* consecutive segments with equal inputs;
* a trajectory that does not start at t = 0, including negative instants, and unevenly spaced samples.

### Limitation closed

The first review listed arrays that a caller can mutate after handing them over. For trajectories that is closed: a segment stores read-only copies of its arrays, so what was validated is what every later reader sees. It still holds for `InputSegment`, whose input vector is the caller's array until `simulate_piecewise` copies it.

### Open limitations

* A non-finite state inside a segment is still accepted when a trajectory is built, and reported afterwards by `check_trajectory` and by the peak functions. At a junction it is rejected, because it cannot be shown to be continuous.
* `git_state` does not limit how long git may take; a git that hangs would hang the run.
* The fingerprint of local changes covers tracked files only. Untracked files are listed by name and not hashed.

## Review of 2026-09-21, third: the measurement interfaces

Scope. The numerical interfaces added for M0-E04: `measurement/sensors.py`, `measurement/noise_statistics.py`, `measurement/observations.py` and `simulation/operating_run.py`. They were written with the rule in hand, so this section records domains and limits, not defects found in committed code. Two problems were caught by their own tests before the code was committed and are listed for the record.

### Domain and limits of each interface

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `SensorSpec`, `MeasurementSpec` | noise level finite and not negative; sampling period finite and positive; one sensor per variable | a noise level of zero is an exact sensor | negative or non-finite noise, a period of zero, no sensor, two sensors for one variable |
| `measure` | finite values, shape (n, sensors); seed and stream explicit non-negative integers | zero noise returns the value bit for bit and draws nothing; no rows returns no rows | non-finite values; a reading that overflows; a missing, negative, fractional or boolean seed or stream element, also when every sensor is exact and nothing is drawn |
| `noise_statistics` | at least 100 finite errors, none beyond 1e100; sigma finite and positive | errors all exactly zero: the lag-one score is set to zero without dividing, and the spread and tail scores report the defect | sigma of zero, which is a valid sensor but has no distribution to test; too few samples; an error more than 1e100 times sigma, whose square would overflow |
| `correlation` | two finite series of equal length, at least 100 samples | none | a constant series, which has no correlation rather than a correlation of zero; unequal lengths |
| `Observations` | finite values, instants strictly increasing, one row per instant and one column per name | none | duplicates or disorder in the instants, non-finite values, shapes that do not match the names |
| `sensor_sample_indices` | a sensor period that is a whole multiple of the sampling of the truth; input changes on sensor instants | a trajectory that does not start at zero; a tail shorter than one period, which simply has no reading | a sensor instant that is not a stored sample, never interpolated; two stored samples on one sensor instant; an input change between two readings |

### Caught before commit

* `measure` checked the readings for finiteness after adding the noise, but numpy warned about the overflow first, and the test suite turns that warning into an error of its own. The addition now runs under `numpy.errstate(over="ignore")` because the outcome is checked on the next line and raised as a `ValueError` that names the cause. This is the one place where a numpy warning is silenced, and it is silenced only to be replaced by an error.
* The stream was validated only where a generator was built, so a wrong stream passed unnoticed when every sensor was exact. Seed and stream are now validated on entry.

### One resolution that is not an arbitrary epsilon, stated as such

A sensor instant and a stored sample are the same instant computed by two routes, `start + period * k` on the sensor clock and `start + h * j` on the grid of the simulation. Each is a product and a sum, each rounded to half a unit in the last place, so they can differ in the last bits: 0.1 * 3 is 0.30000000000000004. They are recognised as the same instant when they agree to four units in the last place of their magnitude. That is the resolution of the arithmetic, of the order of 1e-13 s at two hours, against a spacing of 0.1 s between stored samples; it cannot make a wrong sample pass for the right one. The instant stored in the observations is the stored sample of the truth, so that truth and observations carry identical instants. This differs from the junctions of a trajectory, where the comparison is exact, because there the two numbers are copies of one another and not two computations.

### Open limitations

* The z-scores use large-sample approximations: a normal law for the mean and the correlations, Wilson-Hilferty for the spread, a binomial normal law for the tail fractions. They were checked to be standard normal at n = 1201 over thousands of replicates, not at small n, and the functions refuse fewer than 100 samples.
* Nothing prevents two runs from being given the same seed and stream, in which case they share their noise sample for sample. The convention is one stream per run; the storage layer will need to enforce it.
* `RunTruth` keeps the whole true trajectory in memory, about 1.2 MB per two-hour run.

## Review of 2026-09-21, fourth: noise keys and units where sensors meet the states

Scope. Two defects found in the review of `7bc3e6a`, both reproduced through `observe_trajectory` before anything was changed, and two more of the same kind found while fixing them.

### Failures reproduced and fixed

| Module | Input | What happened | Silent | Resolution |
|---|---|---|---|---|
| `operating_run.observe_trajectory` | a stream of `(0.9, 0.9)`, `(False, False)`, `("0", "0")` or `(-0.9, 0)` | `int()` was applied to every element before the strict rule saw it, so all four were taken for `(0, 0)` and replayed its noise, also with exact sensors | yes | seed and stream are validated as given, first, by the one rule of `measurement.sensors`; nothing is rounded or converted into a valid key |
| `measurement.sensors` | a stream element of 2**32 or more | numpy splits such a key into several 32-bit words, so the stream `(2**32,)` became the words of `(0, 1)` and replayed its noise exactly | yes | stream elements and channels must be below 2**32, one word each; the seed is a plain integer of any size and is not affected |
| `operating_run.observe_trajectory` | `SensorSpec("C_A", "mol/L", 0.005)` | 0.005 was added to states held in mol/m^3, a noise a thousand times too small, and the observations were labelled mol/m^3 | yes | a `SensorSpec` is in SI or it is not built; the configuration loader remains the one place that converts |
| `operating_run.observe_trajectory` | a sensor of T with a unit of concentration | accepted, and the observations came back labelled in kelvin | yes | where a generic sensor meets the states of the CSTR, its unit must be the SI unit in which the simulation holds that state; the generic noise component knows nothing of the CSTR |
| `measurement.sensors.measure` | T measured alone, or listed before C_A | the noise channel was the position of the sensor, so T measured alone received the noise that C_A has when both are measured: two different channels of one run sharing one noise | yes | the channel of a sensor is the index of its state; `measure` takes explicit, distinct channels. Unchanged for C_A then T, so the digests of M0-E04 do not move |

### The one rule for noise keys

`noise_key` and `noise_stream` in `measurement/sensors.py`, used by `measure`, `channel_generator` and `observe_trajectory`. A key is a non-negative integer, Python or numpy, given explicitly. Refused: fractions, including a float that happens to be whole, booleans of Python and of numpy, strings, `None`, negative numbers, and for stream elements and channels anything that does not fit 32 bits. A stream must be a sequence, and a string is not one for this purpose. The rule runs on the values as given, before any conversion and whether or not a sensor draws anything.

### Valid cases, kept valid

* numpy integers of any width, stored afterwards as plain integers;
* the largest stream element, 2**32 - 1, and a seed of any size;
* a subset of the sensors, and any order of them, each variable keeping its own noise;
* the configuration file in engineering units, converted by the loader as before;
* an exact sensor, which still has its keys and its unit checked.

## Review of 2026-09-21, fifth: identifiers, persistence and the database

Scope. The interfaces added for storage: `canonical`, `sampling_clock`, `data/identifiers`, `data/schema`, `data/records`, `data/parquet_store`, `data/private_store`, `data/database` and the SQL under `sql/`. What is examined here is less about arithmetic than the earlier reviews and more about identities, paths, duplicates, conflicts and failures half way.

### Defects found while building them

| Where | Input | What happened | Silent | Resolution |
|---|---|---|---|---|
| `Observations.content_digest` | two sets of different shapes whose numbers line up, one label holding the separator character | the same digest: one row with four inputs against two rows with one input. Labels were joined with a separator and arrays appended without their shapes | yes | a versioned canonical encoding, `observations/v1`, with a kind and explicit lengths for every value; a test rebuilds it by hand, byte for byte |
| Arrow, through `records.table_from_rows` | a null in a column declared not nullable | the table was built without complaint: Arrow records nullability and does not enforce it | yes | nullability is checked explicitly, on writing and on reading; Parquet would also refuse it on writing, later and less clearly |
| `database.ingest_run` | a successful ingestion | the staging schema kept the rows of the last run after the commit | no, found by its test | staging is emptied inside the same transaction |

### Domain and limits

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `sampling_clock.nearest_ticks` | finite instants, a finite start, a positive period | no instants gives no ticks; a clock that starts anywhere, negative instants included | a tick number beyond 2**53, which a float cannot hold exactly, whether it comes from a tiny period, a distant instant or an overflow of the difference |
| `identifiers.path_identifier` | lower-case ASCII, letter or digit first, up to 100 characters | the limit of 100 itself | upper case, separators, a leading dot, a trailing dot, Windows device names such as `nul.p3`, non-ASCII, anything that is not a string |
| `identifiers.run_identifier` | non-negative integer seeds and realisation, at least one excursion, no dot inside plant or protocol | numpy integers | floats, booleans, strings, negatives, zero excursions |
| `records.validate_observations` | identifiers usable in a path, distinct non-empty channel names, SI units, every row on the sensor clock, no tick missing | a run that does not start at zero keeps its instants and gets ticks from 0 | a channel named twice across readings and inputs, a unit that is not SI, a row off the clock, a gap: contract version 1 stores complete runs only |
| `parquet_store.DatasetWriter` | at least one plant, each once; runs of those plants; one run per identity; channels of a plant equal across its runs | the same data set written again: accepted, and not a byte nor a timestamp changes | a run of an unknown plant or of another data set, a run added twice, channels that differ between runs, a plant without runs, an empty data set, an existing data set with other content |
| `parquet_store.open_dataset_directory` | a directory holding exactly the files of its manifest | none | no manifest, which is also what an interrupted writing leaves; another contract, schema version or encoding; a missing or unexpected file; a schema that differs; a null in a required column; row counts or content hashes that differ |
| `database.ingest_run` | a run of a published data set | the same run with the same content from the same data set: accepted, nothing changes | defective staged rows, content that differs from the recorded hash, a plant described differently, a run already present with other content or from another data set |
| `database.block_average` | a whole number of ticks, 1 or more | one tick per block gives the series; more ticks than the run gives one block; the last block of a run is shorter and says so | zero, negatives, fractions, booleans, strings |

### Failures half way

* Writing a data set. Files go to a staging directory, the whole is read back and verified, and the directory is renamed into place. An interrupted process leaves a `.staging-` directory that no reader takes for a data set; a failure inside `publish` removes it. A test kills the writing at two different points and then writes the data set successfully.
* Two writers of one identity. The rename fails for the second, which then examines what is there instead of replacing it.
* Ingesting a run. One transaction; a test makes the insertion fail after the run row is in, and finds no plant, channel, run or measurement of it afterwards, an empty staging schema, and a connection back on the main schema. Files altered after a data set was verified are refused by the quality queries, or by the content hash when every record is individually valid.

### Open limitations

* Atomic publication relies on the rename of a directory on one volume. It protects against an interrupted process, not against a power cut in the middle of the rename, and it does not coordinate several machines. Files are not flushed to the disk explicitly.
* A stale `.staging-` directory left by a killed process is not removed automatically; it is harmless and can be deleted by hand.
* The SQL files are read from the repository, next to `pyproject.toml`. An installed package without a checkout cannot open a database; the error says so.
* `aligned_series` names the six variables of the CSTR. Another process family needs its own view.
* The database has no protection against two processes writing at once beyond what DuckDB itself gives, which is a single writer per file.
* `PT_DATA_DIR` defaults to a directory inside the repository, which may be a synchronised folder. A database file there can be corrupted by the synchronisation client while it is open; the variable exists to point elsewhere.

## Review of 2026-09-22, sixth: generation, export and the scan for hidden information

Scope. `generation/plants`, `generation/pipeline`, `generation/leak_scan`, `data/export`, and one defect of `data/database` found while preparing M0-E05.

### Defect found and fixed

| Where | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `database.ingest_run` | a second plant whose C_A channel is in another SI unit, in a data set that is valid on its own | it was ingested: the quality gate looks at the staged rows of one run of one plant, where a defect that exists only between plants cannot be seen | yes | the quality queries are run once more inside the transaction, on the database as a whole with the run in it; the writer applies the same rule within a data set | `b409f14` |

It was found because a defect chosen for M0-E05, a concentration channel in mol/L, could not be detected in the staging of one run. That defect was replaced, before the experiment was registered, by one that a single run can show.

### Domain and limits

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `plants.load_virtual_plant` | a plant whose nominal inputs give one steady state in the scanned range, stable with the margin of D-017, with the balances closed there | none | no steady state or several; a largest real part not below -0.5 1/min; a residual above 1e-9 of the feed terms. The residuals found are of the order of 1e-16 of them, so the criterion separates a polished root from a point that is not a steady state and sits seven orders of magnitude from what it judges. Two configurations that must be refused were found by computation and are tested |
| `config.DatasetDefinitionConfig` | one or more distinct plants and excitation seeds, at least one excursion, a non-negative master seed, a simulation period that is a time | none | duplicates, empty lists, negative seeds, another protocol than P3, unknown fields |
| `pipeline.run_pipeline` | a definition whose files exist | a data set that is already there: every stage reports it as such and touches nothing | a truth that is not accepted, which raises and leaves a report and no data set |
| `export.export_dataset` | a data set the database holds, complete | an export that is already there with the same content | a run whose aligned rows are not all complete, never trimmed; content that differs from the recorded hash; another export under the same name |
| `leak_scan.scan_available` | hidden values that are finite and not zero | nulls in a column hold no value and are skipped | a hidden value of zero, which cannot be searched for; nothing to scan; a file that is neither Parquet, JSON nor the database is itself a finding |

### What the scan cannot do, stated in the code and here

It compares names and numbers. A hidden value that was scaled, rounded or combined with another would not be found, nor would information carried by the order or the presence of rows. A hidden value that equals a known one is left out on purpose: T_ref is 350 K, which is also the known feed temperature. It is a check against accidents, and the separation it checks is built into the interfaces, where the writers are never handed the truth.

## Review of 2026-09-22, seventh: three defects reported by the reviewer of `404ae79`

Scope. `data/export`, `data/parquet_store` and `generation/leak_scan`, following the review of `404ae79`. Each defect was reproduced on the code as it stood before anything was changed, on a small data set of synthetic observations and, for the scan, on the stored data of M0-E05.

### Failures reproduced and fixed

| Where | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `export.export_dataset` | an export already on disk whose run file held a changed reading, or was missing, or sat next to a stray file | reported as already present: only the manifest on disk was compared with the one just written | yes | `open_export_directory` verifies an export as `open_dataset_directory` verifies a data set; the exporter reads its staging directory back through it before the rename, and examines an export already in place through it before comparing identities. A damaged export is an integrity error that says nothing was overwritten | `b4b1715` |
| `export.export_dataset` | an export directory without `export.json` | a bare `FileNotFoundError` | no | the same: not an export, or its writing was interrupted | `b4b1715` |
| `parquet_store.open_dataset_directory` | a run whose every `quality_flag` was 7 | opened, and read as good data with its content hash intact: the hash does not cover the flag | yes | every row of a run must carry the flag of the contract; `observations_from_long` checks it for Parquet and DuckDB alike | `633b17e` |
| `parquet_store.open_dataset_directory` | a manifest whose `dataset_id` was `Bad Data Set`; `operating_runs` naming another data set, or a plant absent from `plants`, with the table digest recomputed | opened; the absent plant was refused only by accident, with a message about row counts | yes | identifiers, the relations between the tables, the channel metadata, the extent of a run and the rows of a run are verified on reading, whatever wrote the files, and a data set copied under another name is refused | `633b17e` |
| `leak_scan.scan_available` | a master seed of 0, 1 or 1201, on the stored data of M0-E05 | 24, 20 and 4 findings, all of them in ticks, positions, flags, versions and counts, so a data set generated with such a seed could never pass its last check | no, but a false failure of a valid configuration | the fields whose values the contract fixes are not searched for hidden numbers, and their names still are; the readers verify those fields against the contract instead | `968d226` |

### Domain and limits

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `export.open_export_directory` | a directory holding exactly the files of its manifest, every run with the schema of the channels of its plant | none | no manifest; another format, version or encoding; an identifier that is not one; a run listed under another file name, listed twice, of a plant the manifest does not describe, or with an impossible extent; a channel of an unknown kind or with the noise fields of the other kind; a schema, row count, identifier column, tick sequence, extent or content hash that differs from the manifest |
| `parquet_store.open_dataset_directory`, in addition to the fifth review | the identifiers and relations that the writer enforces | none | a manifest that lists other tables than the four, a run under another file name or twice; a plant defined twice, without channels or without a run; a parameter or channel of an unknown plant, or in a unit that is not SI; a channel not named after its plant and variable, of an unknown kind, with the noise fields of the other kind, or at a position that does not follow from 0; a run of another data set, without a SHA-256 hash, with an impossible extent, or whose rows name another run or plant, use a channel of another plant, carry another quality flag, or do not span the extent recorded |
| `leak_scan.scan_available` | as in the sixth review | a hidden number that appears in a structural field is not a finding | as in the sixth review |

### What was decided for the scan, and what was set aside

A threshold on the size of a seed, below which it would not be searched for, is an arbitrary number of the kind the rule forbids, and it would still fail for a hidden parameter equal to an instant of the clock. Searching documents and metadata only, and not columns, would miss a seed written as a column under an innocent name, which the test of planted leaks plants. Excluding fields by what the contract says they hold keeps the search where a number could be hidden by accident, and leaves the structural fields to the readers, which verify them. The cost is stated in the code: a number written on purpose into a field named `rows` or `version` is not found by the scan; it is refused by a reader when it breaks the contract, and not otherwise.

### Open limitations

* Verification on reading is by rules, not by signature. A data set edited consistently, tables and manifest digests together, is accepted when it follows the contract. The content hashes protect the numbers of a run, not the metadata around them.
* `open_export_directory` rebuilds every run to check its hash, as `open_dataset_directory` does, so opening an export reads all of its rows. Nothing at the sizes of M0.
* Content hashes across library versions, seen in practice while re-running M0-E05: under numpy 2.3.5 and scipy 1.16.3 all six runs have other hashes than under 2.5.3 and 1.18.1, on the same machine and commit, and the generation was refused as a conflict with the published data set. The hash identifies the numbers of one environment; the environment is recorded with every attempt.

## Review of 2026-09-22, eighth: the first runs of continuous integration

Scope. The history was pushed to the private remote on 2026-09-22 and CI ran for the first time, on Ubuntu with Python 3.12 and 3.13 and with the same numpy, scipy, pyarrow and duckdb as the project's environment on Windows. One test failed on both, `test_files_that_changed_after_verification_do_not_get_in` of `tests/test_database.py`; it passes on Windows.

### Failure reproduced and fixed

| Where | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `database.stage_run`, through DuckDB's `read_parquet` | a Parquet file rewritten with the same size within the same second, on Linux | the rows staged were those of the previous file, or a mixture of cached bytes and new ones that DuckDB reports as "Out of buffer". DuckDB 1.5.5 keeps an in-memory cache of the external files it reads and validates an entry by the modification time as its own local file system reads it, `st_mtime` in whole seconds on Unix; the file system underneath had finer timestamps, nanoseconds on the runner, which that validation path does not see. On Windows the same sequence reads correctly | yes, whenever the mixture parses | every connection made by `database.connect` turns the cache off, `enable_external_file_cache = false`, the read-only ones included; a data set is read once, its files are small, and what is staged must be what is on the disk at that moment | `9a16e01` |

How it was pinned down. The sequence of the failing test was run on the CI runners, on a temporary branch of the remote that was deleted afterwards: eight versions of one file written one after another, each read through DuckDB and through pyarrow, with the cache on, with the cache off, and through `stage_run`. The file has 126 rows; its truncated version, 125 rows, and its version with one changed value have the same size as the original, 3144 bytes; the version with a duplicated row and the version with a NaN have other sizes.

| Version written | Cache on | Cache on, 11 s pause before the truncated write | Cache off | pyarrow |
|---|---|---|---|---|
| duplicated row, 3148 bytes | 127 rows | 127 | 127 | 127 |
| NaN, 3132 bytes | 126, NaN read | 126 | 126 | 126 |
| truncated, 3144 bytes, same second as the previous write | 126 rows reported, 20 rows of one channel, "Out of buffer" on a third query | 125, right, since the mtime changed | 125 | 125 |
| original again, 3144 bytes, same second | 126 | 125, the truncated content | 126 | 126 |
| one value changed, 3144 bytes, same second | 126, value read | 125 rows with the new value | 126, value read | 126 |

`stage_run` behaved as `read_parquet` did: with the cache on, the truncated version failed with "Out of buffer"; with the cache off, every version was staged as written. Setting `parquet_metadata_cache`, already off by default, changed nothing.

### What this changes and what it does not

Nothing of the content: the fix touches how a connection is opened. Published data sets are immutable, so the cache could only have mattered for a file altered after a data set was verified, which is the case the failing test exists for, and it is exactly there that the wrong bytes appeared. The test that failed is the behavioural regression test, on Linux; a second test pins the setting on every kind of connection, because on Windows the sequence does not reproduce the failure and a behavioural test there would prove nothing.

## Review of 2026-09-22, ninth: the two other runs of D-010, their diagnostics, the oracle and the DuckDB versions

Scope. `simulation/protocols.py` (`steady_segments`, `single_step_segments`), `data/identifiers.py` (`steady_run_identifier`, `step_run_identifier`, `run_definition`), the definitions of data sets in `config.py`, `generation/pipeline.py` (`define_runs`, `run_segments`), the diagnostics of `experiments/06`, `07` and `08`, and the minimum version of DuckDB.

### Domain and limits

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `protocols.steady_segments` | finite nominal inputs, a finite positive duration | none | a duration that is zero, negative or not finite, through `levels_to_segments` |
| `protocols.single_step_segments` | one of the four inputs, one of the two directions, three finite positive durations | none | an unknown input or direction, a duration that is not finite and positive |
| `identifiers.steady_run_identifier`, `step_run_identifier` | whole seconds of 1 or more, a plant identifier without a dot, an input token of letters and digits, a direction of `up` or `down`, a non-negative realisation | none | fractions of a second, zero or negative durations, booleans, an input written as the variable name (`T_c`), another direction, a plant with a dot or upper case |
| `config.SteadyDatasetDefinition`, `StepDatasetDefinition` | durations that are whole numbers of seconds, distinct inputs and directions | none | a fraction of a second, refused rather than rounded; duplicated or empty inputs or directions; an unknown protocol tag |
| Drift of the steady run (M0-E06) | the largest deviation of the true states from the starting state over the run, against a tolerance of 1e-6 of each state | the tolerance is derived from the local error control of the integrator, 1e-9 of the state per step, damped at a stable point, with a margin of one thousand; it is 1/20 000 of sigma_CA and 1/1400 of sigma_T | exact equality with the root is not asked for; the observed drift is 4.3e-8 mol/m^3 and 2.1e-9 K at most |
| Correlation of an error series with a constant state (M0-E06) | undefined | the score is not computed for the steady run, and the reason is recorded in the summary; `noise_statistics.correlation` refuses a constant series | a correlation of zero, or an epsilon, would state something about a quantity that does not exist |
| Direction of the first move after a step (M0-E07) | the sign of the first sampled change, 0.1 s after the switching instant | when that change is exactly zero, the sign of the largest deviation from the starting value over the segment | the derivative at the switching instant is reported but not used for the direction: for a state on which the stepped input has no direct effect it is the residual of the steady state, of the order of 1e-15, whose sign says nothing |
| Extreme during a hold and its position (M0-E07) | the extreme in the direction of the first move; inside the segment means strictly between its ends | an extreme at the last sample is a monotone approach | none |
| Distance to the steady state of the stepped inputs (M0-E07) | one steady state in the scanned range | reported with the count of steady states found, which is one in the sixteen cases | none; a count other than one leaves the distance unreported |
| Phase metrics of the oracle (M0-E08) | a phase with at least one sample | none | a phase without samples has no metrics and raises; the segment structure of P3 gives 12 000 excursion samples and 60 001 rest samples |
| Numerical resolution of the oracle (M0-E08) | the integration error estimate, the distance between the integrations at 1e-9 and 1e-11, against 1 % of the largest difference of the same state | none | a case that fails is reported as not resolved and its differences are not interpreted; the errors found are 2e-7 to 8e-7 of the largest difference |
| Anchoring of variant B (M0-E08) | the conductance of A at T_nominal, equal bit for bit to that of B, and the right-hand sides equal bit for bit at the nominal state | none | an anchoring that moved the nominal point would be reported as a failure of H1 |

### An expectation that was wrong, and why

M0-E08 was registered with the expectation of temperature differences of a few tenths of a kelvin, taken from the ranges of UA(T)/UA_ref of M0-E03. Those ranges were computed over the central 90 % of the samples of the sequences, which leaves out the peaks of the excursions, where the conductance of the true plant departs most from its nominal value. The differences found are up to 4.4 K on the source and 1.8 K on the target, at those peaks. The expectation is corrected in the entry; nothing was changed to meet it.

### The DuckDB versions

The test suite was run on a clean worktree of `6578f45`, in one isolated virtual environment per version (D-025). 1.2.2 refuses `enable_external_file_cache` and with it every test that opens a database fails; 1.3.0, 1.3.2, 1.4.5 and 1.5.0 pass all 716 tests; 1.5.5 is the registered environment. The earlier runs of the same script against the live working tree, while it was being edited, were discarded as evidence.

### Open limitations

* The steady and step protocols are verified for the present plants, amplitudes and durations only, like P3; the tests of M0-E07 are independent and say nothing about chaining them.
* The drift tolerance of M0-E06 is derived for LSODA at rtol = atol = 1e-9; another integrator or tolerance needs its own derivation.
* The oracle of M0-E08 measures the effect of UA(T) under the P3 sequences of M0-E05 only, with everything else known and without noise. It is a scale, not a detection threshold or an identifiability result.

## Review of 2026-09-22, tenth: M0-E08's verdict did not read the physical validity it already computed

Scope. `experiments/08_oracle_conductance.py`'s verdict aggregation, and `simulation/checks.py`.

### Domain and limits

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `TrajectoryCheck.physically_valid` | any `TrajectoryCheck` | none; it is a projection of three existing boolean fields, `values_finite`, `states_physical`, `balances_close`, deliberately excluding `inside_envelope` | not applicable, a pure combination of already-validated fields |
| `comparison_is_valid(primary, secondary)` | two `TrajectoryCheck` instances describing a primary/secondary comparison | the secondary is exempt from its own envelope on purpose, since a diagnostic variant leaving it is a reported result | not applicable |

### Not a numerical defect: a verdict that silently dropped information it already had

This is not a case of a denominator, a root or a grid; it is the same class of silence `AGENTS.md`'s Numerical Robustness section warns against at the level of a verdict rather than an arithmetic operation. `check_trajectory` computed `TrajectoryCheck.accepted` for A and every physical field for B, and `one_case` stored both in the case dictionary, but `main()`'s `verdicts` dictionary, the one thing the exit code is drawn from, only ever read `H1_anchoring` and `H2_resolved`. Codex reproduced the consequence directly: substituting `check_trajectory`'s return so both variants had `balances_close=False` left `A.accepted` and `B.accepted` false while the two verdicts gating the exit code stayed true. The registered cases were never affected, since they were always accepted and physically valid; the gap was in what the verdict could catch, not in what it had caught so far.

### Fix

`TrajectoryCheck.physically_valid` and `checks.comparison_is_valid` (D-027) give the script a third verdict, `H3_valid`, folded into the same `all(verdicts.values())` the exit code already used, so a future run with an invalid pair fails loudly instead of reporting H1 and H2 as if they were the whole story.

### Open limitations

* `comparison_is_valid` is written for exactly the primary/accepted, secondary/envelope-exempt shape M0-E08 needs. A future oracle-style comparison with a different exemption would need its own criterion, not a reuse of this one under a different name.

## Review of 2026-09-25, eleventh: the evaluation contract and the mechanistic models of I1

Scope. The interfaces added in I1 of M1 (D-032): `evaluation/plant`, `evaluation/windows`, `evaluation/budgets`, `evaluation/metrics`, `evaluation/outcomes`, `evaluation/physics`, `models/rollout`, `models/mechanistic` and `models/fitting`. They were written with the rule in hand, so this section records their domains, limits and scales, and what was caught before the code was committed. No module of M0 was changed.

### Domain and limits of each interface

| Interface | Valid domain | Limits resolved explicitly | Rejected at the boundary |
|---|---|---|---|
| `plant.read_known_plant`, `KnownPlant` | the eight known parameters of the contract, each once, in its SI unit, as a number | none | a parameter missing, repeated, unknown, in another unit, given as a string or a boolean, or not finite; a volume, density or heat capacity that is not positive; a product V rho cp that overflows or underflows; a nominal input that is not positive |
| `windows.find_windows` | a run whose rows are the consecutive ticks of the 6 s clock, with the channels of the CSTR in their order and SI units | a run without excursions has no window; an excursion without ten rows at the nominal inputs before it, or that the run ends inside, forms no window and is recorded with its reason | another period, a gap, channels in another order or unit; nominal inputs that no row carries, which would otherwise read as one excursion at tick 0; a hold of another length than the layout's; inputs that change inside a window; a context that reaches the scored readings of the window before |
| `windows.WindowData` | the shapes of its layout, finite values, noise levels not negative | the initial state is the mean of the ten context readings, computed there and nowhere else | wrong shapes, values that are not finite |
| `budgets.split_budget`, `BudgetSplit` | two or more windows of one run, in the order of time | the prefix ends at the next onset, or at the end of the run | a budget below 2, a fraction, a boolean; more windows than the run has; parts that share a row, windows out of order, rows beyond the prefix |
| `budgets.excitation` | the inputs at the onsets | the sign of a difference of two floating-point numbers is exact; the rank is computed in rational arithmetic | none |
| `metrics.score`, `metrics.evaluate` | finite errors, one row per scored reading; sigmas finite and positive; one layout and one set of noise levels per evaluation | a negative excess over the noise is reported as it comes; for readings that are not held out the excess is absent, not zero | a sigma of zero, since an exact sensor has no normalised score; errors that are not finite, which are integration failures; squared errors that overflow and sigmas whose squares underflow, both checked on the outcome |
| `outcomes` | the results of every window of a set, or a training failure | a failure loses a paired comparison and two failures tie | a primary score beside a failure; results for a model that was not trained; a score that is not finite |
| `physics.validity_violations` | dH <= 0 | each bound follows the inputs applied before the instant it judges | an endothermic reaction |
| `physics.implied_terms`, `check_implied_terms` | finite derivatives, states and inputs, one row per point | T = T_c recognised exactly; a sign judged only beyond the rounding bound of its term | values that are not finite, rows that do not match |
| `rollout.rollout` | a finite initial state in the domain of the model, finite inputs, a positive period | one row is a segment of its own; sensitivities to the parameters and to the initial state | wrong shapes, values that are not finite, tolerances that are not positive, a guard that is not a positive integer, sensitivities of a model that states no Jacobians |
| `mechanistic` | k0, E/R and UA finite and not negative; a positive temperature at the start | k0 = 0 is no reaction and UA = 0 an adiabatic reactor; a negative estimate of C_A starts the model and is judged by the validity bounds | an exponential that overflows in the coordinates of the optimiser (`OverflowError`); a start at T <= 0, where the rate law divides by the temperature |
| `fitting.fit_mechanistic` | windows that share positive noise levels, each given once | a start that fails or exhausts its budget is recorded, with its endpoint when it has one | no window, a window given twice, a sigma of zero, repeated labels of starts, a start that moves E/R when it is fixed, a start that puts E/R at or below zero |

### Failures reproduced and resolved before the code was committed

| Where | Input | What happened | Silent | Resolution |
|---|---|---|---|---|
| `scipy.integrate.solve_ivp`, LSODA | a right-hand side that returns infinity; dx/dt = x^2, which grows without bound in finite time | LSODA did not stop by itself: more than 100 000 evaluations on each, until a counter stopped them | yes, a run that hangs | the rollout checks every derivative for finiteness and counts evaluations; either ends the rollout with a record of its cause |
| `evaluation.plant.KnownPlant` | two plants compared with `==` | `ValueError`: the dataclass compares its fields as a tuple, and a numpy array has no single truth value | no | the nominal inputs are a tuple of floats (`7f624b8`) |
| `tests/test_models_rollout.py`, sensitivities | central differences of rollouts at rtol = 1e-12 with a step of 1e-5 | 1 % disagreement with the sensitivity to (E/R) / T_ref | no | not a defect of the sensitivities: the disagreement shrank as the step grew, and was the same with LSODA, DOP853 and Radau. The reference was dominated by the error of the integration divided by the step, about 350e-12 / h, on a sensitivity that is small, a factor 1 - T_ref / T of about 0.014 below that to ln k_350. Each step now balances truncation against that error |
| `tests/test_models_rollout.py`, inputs | the states before a change of the inputs, compared bit for bit with those of a run without the change | they differed by about 1e-10 relative | no | not a defect: with a change at 300 s the first piece of the integration ends there, and the adaptive integrator takes other steps. The test compares them to the accuracy of the integration, and checks that the first state after the change moves by more than a tenth of a sigma |

### Scales and resolutions, stated as such

* The reference integration: LSODA, rtol = 1e-8, atol = 1e-6 in the SI unit of each component. Its error against LSODA and DOP853 at rtol = 1e-12 was at most 2.5e-5 sigma on the target windows of M0 in the smoke run, and is asserted below 1 % of sigma on the 16 corners in the tests.
* The guard of 100 000 evaluations of the right-hand side per window is a guard, not a tolerance: a window of 660 s takes a few hundred.
* The rounding bounds of the implied terms are gamma_4 and gamma_12, about 4.4e-16 and 1.3e-15, times the traffic through each balance. A test compares the computed terms with their exact values in rational arithmetic, for the same floating-point arguments, on points that include large derivatives that cancel and T = T_c.
* The numerical rank of the Jacobian, for the covariance, follows numpy's convention: the largest singular value times the larger dimension times the machine epsilon.

### Open limitations

* The rollout of a window depends on where the later changes of its inputs fall, at the level of the error of the integration, since the integration restarts at each change. That is not a reading of later measurements: the inputs over the horizon are known and given to every model (section 4.1 of the plan).
* A prediction within the error of the integration of a validity bound could be flagged. No state of the plants of M1 comes near one.
* The temperature bound of an endothermic reaction is not implemented; the function refuses such a plant.
* Squared errors beyond about 1e154 are refused, not scored.
* The sandwich assumes the model is right. For MR on the target it is not, and the covariance the smoke run reports for it describes the conditioning of the fit, not an uncertainty of anything.
* A Jacobian that is deficient only to the accuracy of the integration, about 1e-8 of its largest singular value, passes the rank test and gives standard errors that are very large and mean nothing; the singular values are reported beside them for that reason.

## Review of 2026-09-25, twelfth: Codex's audit of I1, 124c377 to a8905dc

Scope. Six findings of Codex's audit of the seven commits of I1, each reproduced on `a8905dc` before anything was changed, and two defects of the same kind found while reproducing them. Every fix has a regression test that fails on `a8905dc` and passes after it. The equations, configurations, data contracts and scientific behaviour of M0 are unchanged: no module of M0 was touched.

What the extreme cases show and what they do not. Most of these inputs lie far outside the scales of the plants, readings and errors of 1e154 to 1e308. No result of I1 at ordinary scales was affected: the smoke run repeated after the fixes gave every fit bit for bit and every score to within 1e-12 of its relative value (below). The defects were nonetheless real. The code gave wrong or misleading answers without saying so, J = 0 or J = inf, a heat flow counted as compatible, a fit with an infinite objective reported as converged, and it would have done so for any model whose predictions or derivatives reach such values, which the learned models of I3 can.

### Failures reproduced and fixed

| Where | Input | What happened | Silent | Resolution | Commit |
|---|---|---|---|---|---|
| `tests/test_models_rollout.py`, finding A | central differences of LSODA rollouts at rtol = 1e-12, in the initial temperature, step 1e-3 K | on Linux 3 values beyond rtol = 1e-4, atol = 1e-6, by up to 1.09e-6: 827 passed, 1 failed; on Windows it passed | no | not a defect of the sensitivities (study below): the reference is now central differences of a fixed-step Runge-Kutta integration | `c03ed41` |
| `experiments/m1_i1_smoke_run.py`, finding B | every start limited to one evaluation | MR and MR_F failed to train, were left out of the scores, and their failure tables were null: failed replicates disappeared | yes | every declared model has a record on every view of every replicate, a training failure when its fit selected no start; the tables count every replicate | `b5c2838` |
| `evaluation.physics.implied_terms`, finding C | derivatives [[-1e308, 0]] with V = 0.1, rho = 1000, cp = 100, dH = -5e5 | the heat flow and its bound were both inf, and the point was counted as within its bound, compatible with a non-negative conductance | yes | the rate, the heat flow and both bounds must be doubles or the point is refused; `ImpliedTerms` checks what the check reads | `33bab6a` |
| `evaluation.metrics.score`, finding D | errors of 1, sigmas of 1e200, held out | J = 0 and MSE - sigma^2 = -inf: sigma^2 overflowed | yes | squares and means formed on values scaled by their largest magnitude; what is not representable is refused | `2674cbc` |
| `evaluation.metrics.score`, finding D | errors of 1e154, sigmas of 1 | J = inf, although J is 1e154 | yes | the same | `2674cbc` |
| `models.fitting.fit_mechanistic`, finding E | readings of 1e308, sigmas of 0.1 | `ValueError: Residuals are not finite in the initial point` escaped from least_squares | no | refused before any start, with a message that says why: the loss is not defined on these data for any model | `572ba65` |
| `models.fitting.fit_mechanistic`, found while reproducing E | readings of 1e160 to 1e300, sigmas of 5 and 0.5 | the normalised residuals were finite and the sum of their squares overflowed: every start was recorded as converged with an objective of inf, one was selected, and there was no training failure | yes | residuals, Jacobian, loss and gradient are checked at every point a start reaches; if one is not a double, that start is a numerical failure and the others go on | `572ba65` |
| `evaluation.windows.WindowData`, finding F | sample_period = inf; a context of ten readings of 1e308 | accepted; the initial state was [inf, inf], although the mean is a double | yes | the period must be finite and positive, the onset time finite, the last scored instant a double; a column whose sum overflows is averaged after scaling | `d785135` |
| `models.rollout.rollout`, found with F | a period of 1e307 s, whose grid overflows | reported as an integration failure, "Unexpected istate in LSODA": an invalid argument taken for a failure of the integrator | no | refused where it enters | `d785135` |
| `models.fitting.covariance`, found with E | noise levels of 1e-306, or of 1e200 | the normalised sensitivities, the variance of a context mean or the inverse of the squared singular values overflowed or divided by an underflowed zero before any check: a RuntimeWarning, which the tests turn into an error | no | every such product is formed under the check of its result, and a covariance that is not a double is given as the reason | `572ba65`, `08ebcf6` |

### Finding A: the study behind the new test

The question was whether the sensitivities or their reference were wrong. On the window of the test, MR-like parameters and the corner (+, -, +, +):

| Comparison | Largest difference |
|---|---|
| sensitivities to the initial state, LSODA at 1e-12 against LSODA at 1e-13, DOP853 at 1e-13 and Radau at 1e-12 | 3.3e-11 |
| sensitivities to the parameters, the same | 1.6e-9 |
| central differences of LSODA at 1e-12 in the initial temperature, against the sensitivity, steps 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 and 1e-1 K | 1.4e-5, 8.3e-6, 2.0e-6, 1.1e-6, 1.9e-6, 1.6e-5, 1.8e-4 |
| central differences of fixed-step Runge-Kutta, 0.1 s, the same steps | 3.1e-8, 7.4e-9, 2.0e-8, 1.6e-7, 1.8e-6, 1.6e-5, 1.8e-4 |

The sensitivities agree among four integrations to far below anything the test asked. The adaptive differences have a floor of about 1e-6 at their best step, set by the steps LSODA chooses differently from nearby points and divided by the step of the difference: the tolerance of the old test sat on that floor, and the last bits of a platform decided the outcome. The fixed-step differences have no such floor and converge as the square of the step until rounding takes over near 1e-8. The new test compares with them, at steps chosen from this study for each direction, and asks 1e-7 of the scale of each sensitivity for the tight rollout, which at every element is stricter than the old tolerance, and 1e-5 for the rollout at the settings of the evaluation. The measured errors are about 1e-9 and 4e-8 to 8e-8 of the scale, on Windows and on Linux.

### Domain and limits, as they now stand

| Interface | Refused at the boundary | Recorded as a failure | Computed as before |
|---|---|---|---|
| `WindowData` | a period that is not finite and positive; an onset time that is not finite; a last scored instant beyond the largest double | none | the mean of the context by numpy, bit for bit, unless its sum overflows |
| `rollout` | a period whose grid of instants is not representable | as in the eleventh review | the integration |
| `metrics.score` | normalised errors, an MSE or an MSE / sigma^2 that are not doubles; for held-out readings a sigma^2 that overflows | none | the definitions; ordinary values differ in their last bits at most |
| `physics.implied_terms` | a rate, heat flow or rounding bound that is not a double, naming the point | none | the rounding bounds and the exact recognition of T = T_c |
| `ImpliedTerms` | values that are not finite, bounds that are negative, signs other than -1, 0 and +1, lengths that differ | none | none |
| `fitting.fit_mechanistic` | scored readings whose values divided by the noise levels are not doubles, besides the earlier checks | at a point a start reaches: residuals, Jacobian, loss or gradient that are not doubles, a failed rollout, parameters that overflow; the start ends, the others go on, and with no selectable start the fit is a training failure | the selection among converged starts |
| `fitting.covariance` | none | normalised sensitivities or a covariance that are not doubles, given as the reason | the sandwich |

### Evidence on Windows and on Linux

* Windows, the reference environment of the project (Python 3.13.7, numpy 2.5.3, scipy 1.18.1): 853 passed, none skipped, with ruff clean, at `b5c2838` and again at `08ebcf6`.
* Linux, in a throwaway container of the image of each of those commits (the numerical libraries of `docker/requirements.lock.txt`, the same versions), with pytest 9.1.1 installed apart from the image, the committed tree extracted from `git archive`, and the user without administrator rights of the image: 849 passed and 4 skipped, both times. The four skipped tests need the git executable, which the image does not have: `tests/test_paths.py:55`, `tests/test_provenance.py:46` and `:67`, `tests/test_provenance_git_failures.py:145`. The same run on `a8905dc`: 827 passed, 1 failed (finding A), 4 skipped.
* A difference between the platforms that is not a defect: in the test of a steady state that leaves one direction undetermined, the start "E/R + 2000 K" converges on Windows and on Linux ends as a numerical failure, at a trial point where the optimiser had gone far along that direction, E/R about 164 500 K, and LSODA reported repeated failures of its error test. Along a direction the data do not determine, the path of the optimiser depends on the last bits of the arithmetic. The failure is recorded as designed, the other four starts converge on both platforms, and the test asks for at least two.

### Results of the checks of this iteration

* The images of `b5c2838` and `08ebcf6` build; they hold no test tool. The data set of M0-E06, generated with each in a new folder, passes the ten checks of the generator, and its two content hashes equal those of the earlier generation in the container on 2026-09-24, and one of the two registered on Windows, as D-031 recorded. No module, configuration, SQL file or container file of M0 changed in this iteration or in I1.
* The smoke run of I1, repeated from `b5c2838` in a new data directory with the exports of M0 read only: every start converged, the parameters, objectives, singular values and covariances equal those of the audited run `20260924T232014Z_bfb69ab` bit for bit, the scores differ from them by at most 3.7e-15 relative (1.2e-12 for MSE - sigma^2, a difference of nearly equal numbers, 4e-16 absolute), and the largest error of the reference integration is again 2.53e-5 sigma. A fingerprint of the exports was the same before and after.
* The same run with every start limited to one evaluation: MR and MR_F fail to train in the three replicates, and every one of their tables holds three replicates, none scored and three training failures; MN is scored in all three.
* The check of the calibration of the sandwich errors, repeated from `b5c2838` over the same 200 seeds: every estimate, z-score and standard error equal to those of `20260924T232253Z_f71deea` bit for bit. The fixes to the fit and to the covariance add checks and change no number where the numbers are doubles, which is why the check was repeated rather than assumed.

### Open limitations

* LSODA prints a `UserWarning` when it fails internally, as in the Linux case above. The failure is recorded; the warning is printed as scipy prints it.
* A result that is representable only as a subnormal number keeps the precision of a subnormal: a J of 1e-200 is computed, an MSE / sigma^2 of 1e-400 is zero.
* What is refused as not representable is refused with a ValueError at the boundary of each function. The learned models of I3 will need a decision on whether a window whose implied terms cannot be computed counts as an integration failure or as an undetermined point of the check; it is not taken here.

## Review of 2026-09-25, thirteenth: Codex's review of 9ea9a76

Scope. Two points of Codex's review of `9ea9a76`, both reproduced on it before anything was changed: an overflow inside the optimiser that the checks of the twelfth review did not reach, and a regression test that could not fail. Nothing of M0 was touched, and no scientific choice of M1 changed.

### The optimiser's own arithmetic

| Where | Input | What happened | Silent | Resolution |
|---|---|---|---|---|
| `models.fitting.fit_mechanistic`, through `least_squares` | Codex's case: one window whose scored readings are the prediction of the textbook start, one of them moved by 1e-13, with noise levels of 1e-155; the first start, at most two evaluations | residuals, Jacobian, loss and gradient were all doubles, the largest element of the Jacobian 1.35e156, and the checks of the twelfth review passed. With ordinary warnings the start ended as converged, status 3, by the tolerance on the step, at its own starting point, with an objective of 7.7e140, and was selected. With the warnings of the suite, `RuntimeWarning: overflow encountered in square` escaped from `fit_mechanistic` | yes, with ordinary warnings | the loss checks the norms of the columns of the Jacobian as the optimiser forms them, and the optimiser runs with floating-point errors raised: either ends the start as a numerical failure, and the other starts go on |
| the same, found while reproducing it | the same window with noise levels of 3e-152 | the norms of the columns are doubles, 4.5e153 at most, but the first radius of trust overflows: again converged by the tolerance on the step, or `overflow encountered in dot` escaping with the warnings of the suite | yes, with ordinary warnings | the same: the raised error ends the start |

The cause, traced in SciPy 1.18.1. `x_scale="jac"` makes `trf_bounds` call `compute_jac_scale`, which forms the norm of each column of the Jacobian as `np.sum(J**2, axis=0)**0.5`. Elements of 1.35e156 square beyond the largest double, so two of the three norms became infinite and their scales, the inverses of the norms, zero. The first radius of trust, `norm(x0 * scale_inv / v**0.5)`, overflowed in its dot product; the solution of the subproblem of the trust region divided infinities (`common.py`, lines 113 and 115) and cast a NaN to an integer (line 398). With a scale of zero the step vanished, and `least_squares` reported that its tolerance on the step was satisfied. The second case shows that checking the norms is not enough: with noise levels of 3e-152 every norm is a double and still the radius of trust, the starting point times the norms, overflows.

The correction. `least_squares` runs inside `numpy.errstate(over="raise", invalid="raise", divide="raise")`, so that an operation of its own whose result is not a double raises a `FloatingPointError`, whatever the filters of warnings are. That exception, and no other, is caught around the call, and ends the start as a numerical failure whose message names the operation and the last point the loss was computed at; the other starts go on, and with no selectable start the fit is a training failure, as before. The loss also checks, before handing them over, the norms of the columns of the Jacobian as the optimiser will form them, so that the commonest case is named precisely, and it refuses a proposed point that is not finite. Underflow is left as numpy leaves it: it loses nothing that matters to these results. Nothing else of the optimisation changed: the method, the scales, the tolerances, the starts and the selection among converged starts are those of D-032. No minimum was put on the noise levels, no value is clipped and no warning is silenced.

Why this changes nothing on ordinary data. The smoke run, the run with training failures and the 200 fits of the calibration, made before this change, printed no warning of floating-point arithmetic: the optimiser met none of these operations there, so raising them changes no result. The smoke run repeated with the change gave 11 444 numbers equal bit for bit to those of the run of `b5c2838` (below).

### A regression test that could not fail

`tests/test_evaluation_metrics.py` checked that J = 1e-200 is computed, and not zero, with `pytest.approx(1e-200, rel=1e-15)`. `pytest.approx` also allows an absolute difference of 1e-12 by default, so zero passed it: `0.0 == pytest.approx(1e-200, rel=1e-15)` is true. The computation was right; the test could not have noticed if it went wrong. It now asserts `small.j > 0` and compares with `abs=0.0`, which rejects zero and accepts the J of 1e-200 that the code computes. No other test of I1 compares with an expected value that small.

### Evidence

* Both regression tests of the optimiser fail on `9ea9a76` with the warnings of the suite, with the two overflows above escaping, and pass after the change, each start recorded as a numerical failure. The first also shows that a failed start does not stop the next: the second start of Codex's case fails on its own loss, which is not a double there.
* Windows, the reference environment: 855 passed, ruff clean, no whitespace error in the diff.
* Linux, in a throwaway container of the image built with the change, with pytest 9.1.1 installed apart and the tree of the change extracted by `git archive`, run by the user without administrator rights: 851 passed and 4 skipped, the four that need the git executable, which the image does not have.
* The ordinary smoke run of I1, repeated with the change in a new data directory, with the exports of M0 read only and a fingerprint of them equal before and after: every start converged, no warning was printed, and all 11 444 numbers of its summary equal those of the run of `b5c2838` bit for bit. The run was made from the working tree of this change before it was committed, so its name carries the mark `-dirty`, and its provenance holds the fingerprint of the change.

### Open limitations

* The raised floating-point errors cover the arithmetic of numpy inside the optimiser and inside the rollouts it calls. A failure inside LAPACK, such as a singular value decomposition that does not converge, would raise `LinAlgError`, which is not caught; it has not been seen and would not be silent.
* Inside a fit, a floating-point error in the code of `solve_ivp` itself ends the start; the same error in a rollout outside a fit gives a warning and then a state that is not finite, recorded as an integration failure. Both are failures, reported differently.

## Addendum of 2026-09-27: the checks that closed I1

Four things found while the criteria of I1 were checked again before its closure (D-033). One is a test corrected here; the others correct or complete earlier sections of this record without rewriting them. None changes the code of I1 or a result obtained with it.

### A second regression test that could not fail

`tests/test_evaluation_physics.py`, `test_implied_terms_computed_by_hand`, compared the rounding bound of the implied heat flow, 1.2256862191861743e-13, with `pytest.approx(..., rel=1e-15)`. The expected value lies below the absolute tolerance of 1e-12 that `pytest.approx` allows by default, so the comparison accepted zero, half the bound and eight times the bound; a reviewing agent showed that a bound multiplied by eight passed every test of I1. It is the mechanism of the second point of the thirteenth review, whose closing sentence, that no other test of I1 compares with an expected value that small, was wrong. The test now asserts that the bound is positive and compares with `abs=0.0`: the value the code computes passes, and zero, half and eight times the bound fail. To look for more of them, the whole suite was run with `pytest.approx` wrapped so as to record every comparison whose tolerance was the default absolute one and whose expected value was so small that it dominated: of the 855 tests, this was the only one.

### The numbers of the silent convergence

The table of the thirteenth review says that before its correction the first start converged at its own starting point with an objective of 7.7e140. Reproduced again on `9ea9a76`, with the window of the regression test (`window_the_start_reproduces(1e-155)` in `tests/test_models_fitting.py`) and warnings ignored: with the declared limit of 100 evaluations, the start converged, status 3 by the tolerance on the step, after two evaluations, at a point that had moved from k0 = 1.2e9 1/s to 1.2000085e9 1/s, with an objective of 2.7e149, and it was selected; with a limit of two evaluations it exhausted its budget. The defect, its cause and its correction are as the review describes them. The objective and the endpoint in its sentence are not those of the test's window, and nothing relies on them.

### A rollout under another integrator

`rollout` returns a record for every failure under LSODA, the integrator of the evaluation (D-032). It accepts the other methods of `solve_ivp`. With a right-hand side that returns a huge but finite derivative, [1e308, 0], two of them raise instead of returning a record: Radau and BDF raise `ValueError: array must not contain infs or NaNs` from inside their linear algebra. With the warnings of the suite, RK45, DOP853, Radau and BDF let `RuntimeWarning: overflow encountered in divide` escape. LSODA ends with the work limit, as recorded. No code of the project rolls out with another method except the tight references of the tests and of the smoke run, on models whose derivatives are ordinary.

### A limitation of the eleventh review that no longer holds

The eleventh review lists as open that squared errors beyond about 1e154 are refused and not scored. The correction of finding D in the twelfth review forms squares on values scaled by their largest magnitude, so errors of 1e154 and more are scored whenever the MSE and the MSE divided by sigma^2 are doubles; what is refused is what is not representable, as the table of the twelfth review states.

### Open

* A rollout with a method other than LSODA can raise on a right-hand side that overflows in the integrator's own arithmetic. Before a learned model is rolled out with another method, the rollout needs the treatment the fit received in the thirteenth review.

## Review of 2026-09-27, fourteenth: the interfaces of I2

Scope. The new module `models/identifiability.py` and the two scripts of M1-E01, `experiments/m1_e01_identifiability.py` and `experiments/m1_e01_oracle.py`, examined before the experiment was registered or run. For each interface: its domain, what it refuses where the input enters, what it returns as a record, and the scales that decide its verdicts. Nothing of M0 or I1 was changed.

### Domain and limits of each interface

| Interface | Refused at the boundary | Returned as a record | Computed |
|---|---|---|---|
| `steady_state_parameters` | a state that is not two finite values; an E/R that is negative or not finite; C_A outside (0, C_Af), where no positive first-order rate holds it; T <= 0; T = T_c, where the energy balance leaves the conductance undetermined; a conductance that is not positive and finite; an E/R for which k0 overflows | none | the rate and the conductance from the two balances, which do not depend on E/R, and k_350 from them |
| `window_information` | noise levels that are not two positive finite values; a number of context readings that is not a positive integer, or is beyond the largest double; noise levels whose variance, the variance over the number of readings, or its inverse is not a positive double | a failed rollout, or normalised sensitivities or their products that are not doubles, as `NoInformation` with its reason | S^T S; the context information by Woodbury's identity, with a 2 x 2 solve (replaced in the fifteenth review); the excess C Sigma_0 C^T; the triangular factor R of S |
| `design_covariance` | no window; weights that are not finite, are negative or are all zero, or whose number of rows is beyond the largest double; windows with different numbers of parameters | a rank-deficient design, including one with fewer rows than parameters; a singular context information; a covariance that is not a double; each with the reason and no covariance | the three covariances, the sandwich as exact^-1 + exact^-1 excess exact^-1, as `fitting.covariance` forms it, scaled to (ln k_350, E/R in K, ln UA) |
| `profile_settings` | an E/R that is not finite, or not positive, which `FitSettings` refuses; a list of starts that all move E/R | none | the settings of MR with E/R held |
| `profile_interval` | a grid that is not finite and strictly increasing; lengths that differ; no point with a value; a threshold that is not positive and finite; a lowest increase above the threshold | a side without a crossing, with its reason: the end of the grid, or a point where no start converged, across which nothing is interpolated | the crossings by linear interpolation between points of the grid |

### Scales and resolutions, stated as such

* The numerical rank of a design follows numpy's convention, the largest singular value times the number of rows of the design times the machine epsilon, applied to the singular values of the stacked, weighted triangular factors of its windows. They are the singular values of the stacked S, and S^T S is never formed to judge the rank: the smallest singular value of a rank-deficient S would come out of S^T S at about the square root of the machine epsilon and pass the test. A test checks it on a window that stays at the steady state, where three parameters cannot be determined and two can.
* The context information is judged singular by numpy's convention on its own eigenvalues. In exact arithmetic it is positive definite whenever S has full rank and the context covariance is finite: for a direction v, v^T (S^T S - C (Sigma_0^-1 + G^T G)^-1 C^T) v is positive, because Sigma_0^-1 is. In floating point it is S^T S minus a correction, with a relative error of about the machine epsilon times the square of the condition number of S. That is negligible for the windows of P3, whose S has a condition number of the order of 10^2 to 10^3. From about 10^8 on, a design whose S the rank test accepts could be reported with a singular context information. (The fifteenth review replaces the subtraction; the bound's accuracy is still set by the inversion of a Gram matrix.)
* For a convex profile the interpolated crossings lie inside the true ones: the chord lies above the profile. The profile also lies above every secant extended beyond its two points, so the true crossing is no further out than the bracketing point outside it, nor than where the neighbouring secants reach the threshold. The script reports that outer limit for each crossing; it holds under convexity alone, and where the slopes of the secants do not increase outward the limit is the bracketing point, up to 25 K away on the fine part of the grid and 500 K on the wide part. (Wrong where the slopes do decrease: the interpolated crossing is then no bound either. Corrected in the fifteenth review.)
* The fit held at the estimate of the free fit must reach its loss within 1e-3 in the sum of squared normalised residuals. A first criterion, relative to J, meant nothing when J tends to zero: on noise-free synthetic data both objectives were about 1e-7 and differed by 3 % of each other. It was replaced before the code was committed.
* The Monte Carlo over the corners of P3 uses 20 000 draws for each number of windows. The probability level of an empirical 5 % quantile is then uncertain by about 0.15 percentage points, sqrt(p (1 - p) / N); the draws are exact otherwise, since the information of each corner is computed once.

### Open limitations

* The information is local, at one point, and assumes that the model is right. The model of the target is not, which is why part 2 and part 3 exist.
* The windows of part 1 start exactly at the steady state. On the runs they stand for, the state at an onset carries a residue of the excursion before it, of the order of millikelvins after a rest (M0-E03b), which the a priori computation leaves out.
* Found by the review of the registration, before the experiment was run, and corrected with regression tests:
  * noise levels whose variance overflowed made the solve raise `LinAlgError`;
  * a window with fewer rows than parameters was judged on the smaller of two singular values;
  * weights large enough to overflow the count of rows gave a warning and then a verdict of rank deficiency;
  * a failed or rank-deficient corner in part 1 made the script raise before its summary was written;
  * a failed free fit was still given a minimum;
  * a point below the free fit was not flagged.
* Found by continuous integration after the experiment was run, on `4454109`: two tests compared quantities that pass through the context bound to 1e-12 relative. That is tighter than the bound's own accuracy of about 5e-12 stated above. A Linux runner with Python 3.12 gave 1.25e-12 where Windows gave 5.7e-13, and the same code had passed at `e166bb5`. The tests now ask 1e-9 of those comparisons. The code of the experiment is unchanged.
* The oracle's windows start from the mean of ten copies of the exact steady state. That mean can differ from the state in its last bit; the script records whether it does and by how much, and the difference does not matter to a fit whose objective is of the order of one.

## Review of 2026-09-27, fifteenth: Codex's review of `129063c`

Scope. Codex reviewed I2 at `129063c` and reported two defects, each with a reproduction. Both were reproduced on `129063c` before anything was changed, both are corrected with regression tests, and neither acted in the registered run of M1-E01: its numbers do not reach the first, and every bracket of that run passes the corrected test of the second. The experiment log records the run repeated after the correction and its comparison with the registered one.

### A. The context information could drop the cost of the initial state

`window_information` computed the context information as S^T S - C (Sigma_0^-1 + G^T G)^-1 C^T, with C = S^T G. It checked that G^T G and each product were finite, but not their sum with Sigma_0^-1. With noise levels far from one that sum overflows while each term is finite. The solve then divides by infinity, the correction comes out zero, and the context information equals the information with the initial state exact: finite, accepted, and wrong.

Codex's reproduction, repeated on `129063c`: a model of two states and two parameters with a right-hand side of zero, df/dx = 0 and df/dtheta = 1e-153 times the identity, started from [1, 350] under 170 rows of inputs one sampling period of 6 s apart, with noise levels of 1e-153 for both channels. For each channel S is t = 6 k, k = 1 to 170, and G is the identity over sigma, so the Schur complement is sum t^2 - (sum t)^2 / 180 = 59 477 220 - 42 253 245 = 17 223 975. The function returned 59 477 220 on the diagonal, the value with the initial state exact.

The correction does not form Sigma_0^-1 or G^T G.

* The initial state is counted in standard deviations of its context mean, sigma_j / sqrt(n). G becomes G Sigma_0^(1/2), with entries dx_o / dx0_j times (sigma_j / sqrt(n)) / sigma_o, and Sigma_0 becomes the identity. The excess of the sandwich is then C Sigma_0^(1/2) (C Sigma_0^(1/2))^T.
* The context information is read from the QR factorisation of the joint least-squares problem of the initial state and the parameters: the scored readings, rows [G Sigma_0^(1/2) S], and the context mean, rows [I 0]. The block of the parameters in the triangular factor is a root of the Schur complement. Nothing is subtracted, and every intermediate is bounded by representable quantities once S and G Sigma_0^(1/2) are.
* The domain changes where the formulation does. The inverse of the variance of the context mean is no longer formed, and its check is gone. The factors (sigma_j / sqrt(n)) / sigma_o are formed instead and must be positive doubles, which refuses two noise levels about 1e314 apart. The checks on sigma^2 and on sigma^2 / n stay: they are the noise model the sandwich describes.

Accuracy, measured on four corner windows of P3 at the point of the tests against the Schur complement computed in exact rational arithmetic from the same double-precision S and G: the largest error of the context bound, relative to the standard errors, is 6.3e-15, where the subtraction gave 2.1e-14. The bound still inherits the error of inverting a Gram matrix, about the machine epsilon times the square of the condition number of S, so the tests keep asking 1e-9 of the comparisons that pass through it.

Regression tests, in `tests/test_models_identifiability.py`: Codex's case against the value computed by hand, for the information with the initial state exact, the context information and the excess, each to 1e-12; and the refusal of noise levels whose ratio is not a double. Both fail on `129063c`. The tests of the fourteenth review, among them the equality of the context information with the direct generalised least-squares form and with the Schur complement of the joint information, pass unchanged.

### B. A bracket of a crossing accepted where the profile is not convex

`crossing_bracket` computed whether the sampled slopes of the profile increase outward and returned the flag, but `curvature_agrees` did not ask for it. When the flag was false, the outer limit fell back to the bracketing point and the interpolated crossing was still used as the inner end. The inner end is a bound only when the chord of the bracketing points lies above the profile, that is under convexity. Without it the true crossing can lie inside the interpolated one, and B2 could call the profile in agreement with the linearised curvature on a bracket that does not contain the true crossing. A pair of B4 would then have been read on that verdict.

Codex's reproduction, repeated on `129063c`: the points x = -3 to 3, the profile x^2 / (1 + x^2), the threshold 0.65 and the estimate 0. On both sides the interpolated half-width was 1.5, the outer limit 2.0 and the flag false; with an expected half-width of 1.6, `curvature_agrees` returned True. The true crossing is sqrt(0.65 / 0.35) = 1.3628, inside the interpolated one: the profile is concave there, and the chord lies below it.

What the points of a profile can say. No finite set of points shows how a profile behaves between them. Every bracket of a crossing therefore rests on a hypothesis about the profile. The one used is convexity over the span from the last point on the other side of the estimate to the second point beyond the crossing. Secant slopes that do not decrease along the span are necessary for it, not sufficient. A span that passes is consistent with convexity and does not demonstrate it. The fourteenth review, the registration of B2 and the result of M1-E01, where it says "the profile is convex at every crossing", describe sampled slopes as convexity. They should be read as "the sampled slopes are consistent with convexity".

The correction:

* The interpolated half-width is always reported, as a description of the sampled profile.
* The bracket, from the interpolated half-width to the outer limit, is reported only when the span is sampled and its slopes do not decrease. It is labelled as holding if the profile is convex over the span.
* Otherwise the bracket is not evaluable and its reason is recorded. There are three reasons: a point of the span is not sampled, since there is none on the other side of the estimate or fewer than two beyond the crossing; a point of the span has no value; the sampled slopes decrease somewhere in it.
* B2 gives a verdict for each side, "agrees", "narrower", "wider" or "not evaluable". A profile with a side that is not evaluable has no verdict, `curvature_agrees` returns None, and no pair of B4 that holds its run is read.

The span is new; `129063c` tested only the secants beside the bracketing one on the side of the crossing. With it, the check reaches across the minimum, so a crossing between the estimate and the first point beyond it can still be tested. Applied to the registered rows of M1-E01 before the run was repeated, it leaves every bracket evaluable, with the same outer limits bit for bit and the same verdicts.

Regression tests, in `tests/test_m1_e01_scripts.py`: Codex's case, now without a bracket or a verdict; three spans that lack what the test needs; the three verdicts on a convex profile computed by hand; and a pair of B4 left unread when either run has no verdict or disagrees. The test of the fourteenth review on convex profiles, quadratic or steeper, now also asks that their brackets be evaluable.

### Open

* The brackets remain conditional on convexity between the points, which the points cannot show. A finer grid would narrow them without removing the condition.

## Review of 2026-09-27, sixteenth: the interfaces of I3

Scope. The modules of I3: `models/learned.py`, `models/training.py` and `models/linear.py`, and the change to `models/fitting.py` that lets BL share the loss of MR. For each interface: its domain, what it refuses where the input enters, what it returns as a record, and the scales that decide its verdicts. Nothing of M0, I1 or I2 changed behaviour. MR's fits on `m0-e05` and `m0-e07`, every start and covariance, are the same bit for bit before and after the change to `fitting.py`.

### Domain and limits of each interface

| Interface | Refused at the boundary | Returned as a record | Computed |
|---|---|---|---|
| `learned.fitting_scales` | no window; a standard deviation of a state or an input that is zero or not finite, with the variable named (section 5.8) | none; `train` records the refusal as a training failure | means and population deviations over the readings of the contexts and scored readings of F, and over the inputs of its scored rows |
| `learned.initial_parameters` | an unknown family; hidden layers that are not positive integers; a hybrid without a start; BN with one | none | Glorot's uniform law for hidden layers, zero biases, a zero last layer, from the given generator |
| `learned.rhs` | nothing: it is called inside the rollouts | none; an exponential that overflows is an arithmetic error that the reference rollout records, and a value that is not finite in the training rollout makes the loss not finite | the equations of section 8.2 and 8.3 |
| `learned.LearnedModel` | an unknown family | an initial state at T <= 0 for a hybrid, as for MR | its right-hand side in numpy |
| `training.Configuration`, `TrainingSettings` | an unknown family; hidden sizes that are not positive integers; a penalty that is negative or not finite; a rate that is not positive; steps and intervals that are not positive integers, or steps that are not a multiple of the interval; betas outside [0, 1); an epsilon that is not positive | none | nothing |
| `training.train` | no window of F or of V; F and V that share a window; windows of different lengths, periods or noise levels; noise levels that are not positive; holding the parameters of BN | a refused scale; a loss or gradient that is not finite at a step, with the step; a checkpoint whose rollout of V failed, with the failures; every checkpoint failed; each as a training failure or a checkpoint that cannot be selected | the loss, Adam, the checkpoints, the selection |
| `training.select_configuration` | nothing | None when every configuration failed | the lowest criterion, the first on a tie |
| `training.training_seed` | a base, replicate or configuration that is not a non-negative integer | none | a seed from `SeedSequence` |
| `linear.excited_directions` | nominal inputs that are not four finite values; deviations that are not representable | none | amplitudes, the exact rank and a basis of the span of the rational levels |
| `linear.fit_linear` | what `fitting.check_windows` refuses for MR | each start that stops on the optimiser's arithmetic, a failed rollout or a Jacobian that is not representable, as a numerical failure; a start that uses its evaluations; no converged start, as a training failure | the fit of BL |

### Scales and resolutions, stated as such

* The training scheme. Two steps of the classical fourth-order Runge-Kutta scheme per row of 6 s. At MR on real windows it differs from the reference rollout by 5e-4 sigma (D-035). On a linear equation its error falls sixteen times when the steps are halved, across a change of the inputs (`tests/test_models_learned.py`). A learned model can be stiffer than MR. Each checkpoint therefore records the largest difference between the two rollouts on V, in sigmas, and the model is selected and evaluated by the reference.
* The gradient is the exact derivative of the computed loss, by automatic differentiation of the scheme. It is checked against central differences with a relative step of 1e-6, to 1e-6 of the largest component, at a point where every weight and parameter is moved. At the start of a hybrid, whose mechanistic parameters are at a minimum of MR's loss, those components are nearly zero and central differences lose their digits to cancellation. The check is made away from it.
* A hybrid at the identity against MR: their right-hand sides compute the rate by equivalent operations and agree to 1e-13 of the feed terms. Their reference rollouts agree to 1e-6 sigma.
* Double precision throughout: importing `models.training` sets JAX to 64 bits for the process.
* BL's coordinates are without units, and the trust region is not rescaled (D-036). Its rank is exact, computed on rationals.

### Open limitations

* The failure of a training at a step ends it and discards its earlier checkpoints, as section 8.7 lists a loss that is not finite as a training failure. The registration may choose to keep them.
* An Adam step can take a network where its exponential overflows. That ends the training as a failure; nothing restarts it or lowers the rate. A development run with a rate of 1e-2 on a strongly planted correction ended so at step 939.
* The training scheme has a fixed step, so a learned model whose dynamics become much faster than a few seconds is integrated inaccurately in training. That is seen, not prevented: the difference of the two rollouts is recorded at every checkpoint, and selection and evaluation use the reference. Measured in the pilot (experiment log, "Development runs of I3"): on the data of M0, at most 7e-4 sigma at the selected checkpoints. On a synthetic run hotter than the target, 0.046 to 0.055 sigma for HK and HKU at 20 windows, all of it at the corner where every input is high. There the largest eigenvalue of the model's Jacobian is 0.16 per second, 0.48 times a step of 3 s. Four steps per row bring those windows to 2.7e-3 sigma or less and eight to 1.3e-4; the pilot proposes four for the registration.
* The fourteenth review's limitation on rollouts with methods other than LSODA still holds. The reference rollout of the learned models is LSODA.

## Review of 2026-10-01, seventeenth: Codex's audit of `1a9fad4`

Scope. Codex audited I3 at `1a9fad4` and reported five defects, each with a reproduction, none shown to act in the pilot's results. All five were reproduced on `1a9fad4` with Codex's script before anything was changed, with Codex's numbers. They are corrected in `436a811` (F1, F2, F3, F5) and `44f37a8` (F4), each with a regression test that reproduces the audit's case. Codex has not reviewed the corrections.

### F1. The domain of the mechanistic parameters of a hybrid

Adam moved the coordinate (E/R) / T_ref without the bound that MR's fit keeps at zero. Started at E/R = 0, on synthetic data whose rate falls as the temperature rises, a training of HU took E/R to -24.68 K, reported success and selected it. Reading the selected model's parameters then raised, since `MechanisticParameters` refuses a negative E/R.

Corrected:

* Each step of Adam is projected onto the domain E/R >= 0. E/R reaches its valid limit of zero and stays there, and the steps at which the projection acted are counted in the record.
* `learned.LearnedModel` refuses parameters outside the family's domain (`learned.domain_problem`): weights and coordinates that are not finite, a negative E/R, a k0 or UA that is not a positive double. Training, evaluation and storage therefore share one domain. A checkpoint outside it records why and cannot be selected.
* Where a step leaves E/R positive the projection, a maximum with zero, returns the coordinate unchanged bit for bit. Every selected E/R of the pilot was between 8894 and 11052 K, and five trainings of the pilot run again gave the same losses bit for bit (experiment log).
* The reasons for projection rather than a change of coordinates are in D-036, clarification of 2026-10-01.

### F2. The state of Adam

The loop checked the loss and the gradient, not the moments. With noise levels of 1e-150 the loss, 2.74e302, and the gradient, at most 2.38e302, were finite. The square of the gradient overflowed the second moment, whose infinite root made every update zero; the training went on with an unchanged loss and reported success.

Corrected: the two moments, their corrections, the update and the new parameters are checked at every step. If one is not finite the training ends as a failure that names it and the step: "the second moment of Adam is not representable at step 1". The checks change no value.

### F3. The criterion on V

`validation_score` computed sqrt(mean(residuals^2)). The mean of squares of readings of 1e153 overflows, the criterion was inf, and inf was selected because only None was excluded. Under the suite's warnings as errors the same case raised.

Corrected:

* The criterion is J as the metrics of the evaluation compute it, by the scaled root mean square, shared as `metrics.normalised_rms`.
* A criterion that is not representable, because the normalised errors are not, is recorded in the checkpoint's failures and not selected. Only a finite criterion is selected.
* The difference of the two rollouts is computed under the same guard, inf when it is not representable, and no warning escapes.

Regression tests:

* 1e153, where the mean overflowed, now gives 1.4212670403551894e153, the value of `metrics.score`;
* 1e200, where each square overflows;
* noise levels of 1e170, where every square underflowed and the plain formula gave zero;
* 1e308 over a noise level of 0.5, which is not representable: every checkpoint records why, and the training is a failure.

The change of formula moves ordinary criteria by at most 3e-16 relative, the rounding of a different order of operations. No selection of the five trainings run again changed.

### F4. The accounting of the pilot

`trainings()` of `experiments/m1_i3_pilot.py` skipped the hybrids of a run and budget whose MR_F had failed. With no start, the declared grid of sixteen configurations gave four trainings of BN, and the twelve hybrids left the count.

Corrected:

* They are now outcomes: training failures whose reason names the failed start, with run, budget, family, configuration and rate.
* The pilot compares the outcomes it recorded with those expected, and its exit code is 1 if they differ.

Every MR_F of the pilot converged, so no recorded outcome was missing.

### F5. The sampling period of F and V

The training rolled out V by its scheme with F's period. A validation window at 3 s against F at 6 s was accepted, and the recorded difference of the two rollouts, up to 0.0996 sigma, compared different horizons.

Corrected: the contract that the windows of a training share their length, sampling period and noise levels now covers F and V together, and such a training is refused before any step. All windows of the pilot are at 6 s.

### Also done after the audit

* The difference of the training scheme from the reference is recorded on F at the selected checkpoint, as well as on V at every checkpoint.
* The selected model can be written and read back with its configuration, scales, settings and a SHA-256 of its content (`models/persistence.py`). It reads back bit for bit.

### Open

* A training that fails at a step discards its earlier checkpoints, the policy of D-036. Keeping them would be a decision of the registration, made before any benchmark fit.
* The fourteenth review's limitation on rollouts with integrators other than LSODA still holds.
