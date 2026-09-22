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
