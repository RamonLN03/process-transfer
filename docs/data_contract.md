# Data contract of M0

What a stored data set of the virtual plants is: its identities, its time, its channels, its quality flag, and what it may and may not contain. Written before the persistence code, and changed only together with it. The decisions behind it are D-021 to D-024 in `docs/decisions.md`; all of them are technical and reversible, and none changes an accepted scientific decision. Contract version: 1.

## Where things live

    PT_DATA_DIR/
      available/                        everything a model may read, and nothing else
        datasets/<dataset_id>/          an immutable data set: Parquet files and manifest.json
        databases/<name>.duckdb         derived from data sets; can be rebuilt at any time
        exports/<dataset_id>/           derived from a database by SQL; aligned series
      private/
        datasets/<dataset_id>/          what is needed to regenerate and to diagnose: seeds,
                                        noise streams, full configurations, checks of the truth
      experiments/<experiment>/<run>/   diagnostic artefacts of experiments, as before; they may
                                        contain hidden parameters and exact states

The two branches are siblings on purpose. Nothing under `available/` refers to a path under `private/`, no reader uses a wildcard that could cross from one to the other, and reading and exporting work when `private/` does not exist. This is a separation of content and of code paths. It is not a permission of the operating system: whoever can read the disk can read both branches.

## The five entities

The tables of `AGENTS.md`, with the columns fixed here. SI units throughout. No wall-clock column appears in any of them: two generations of the same data set must give the same content, and a time of creation would differ. When a data set was generated is recorded in its manifest.

| Table | Columns | Key |
|---|---|---|
| `plants` | `plant_id`, `name`, `process_type` | `plant_id` |
| `process_parameters` | `plant_id`, `parameter`, `value`, `unit` | `plant_id`, `parameter` |
| `sensors` | `sensor_id`, `plant_id`, `variable_name`, `channel_kind`, `channel_index`, `unit`, `sampling_period_s`, `noise_model`, `noise_std` | `sensor_id`; also unique on `plant_id` with `sensor_id` and with `variable_name` |
| `operating_runs` | `run_id`, `plant_id`, `dataset_id`, `operating_mode`, `description`, `start_time_s`, `end_time_s`, `sampling_period_s`, `n_samples`, `content_sha256` | `run_id`; also unique on `plant_id` with `run_id` |
| `measurements` | `plant_id`, `run_id`, `sensor_id`, `sample_index`, `time_s`, `value`, `quality_flag` | `run_id`, `sensor_id`, `sample_index` |

A measurement refers to its run and to its channel through the pairs (`plant_id`, `run_id`) and (`plant_id`, `sensor_id`), so a row cannot join a run of one plant to a sensor of another.

## Identity

Three things are kept apart.

The logical identity of a run, `run_id`, says what the run is: the plant, the protocol, the definition of the experiment under that protocol, and the realisation of the noise. It is built from the definition of the run and from nothing else, neither from its content nor from the clock:

    <plant_id>.<protocol>.<definition>.n<noise realisation>

    p3       e<excitation seed>.x<number of excursions>        target.p3.e0.x10.n0
    steady   d<duration in seconds>                             target.steady.d7200.n0
    step     <input>-<direction>.l<lead>.h<hold>.r<recovery>    target.step.tc-up.l600.h600.r600.n0

The steady and step protocols draw nothing at random, so their identities name no seed; the input is one of q, caf, tf and tc, the direction up or down, and the durations are whole seconds (D-026, added on 2026-09-22; contract version 1 is unchanged, since no table, column or encoding changed). The definition part is the same on every plant that runs the same experiment, which is how the runs of two plants are paired.

The noise realisation is part of the identity. Two different realisations of the noise on the same excitation are two runs, `n0` and `n1`, and cannot share an identity by accident. The seed of the noise is not part of it and appears nowhere under `available/`.

The generation attempt says when and with which code a data set was produced: UTC time and commit, as in the run directories of the experiments. It is recorded in the manifest and in the private provenance. Two attempts at the same data set are expected to give the same content.

The content hash, `content_sha256`, says what the numbers are. It is the SHA-256 of a canonical encoding of an observation set, given below. It depends on values, labels and instrument specification, not on a file format, a library version or the time of writing.

Policy on repetition, the same for a data set on disk, for a run in the database and for an export. The same identity with the same content is accepted and changes nothing: the operation is idempotent. The same identity with different content is a conflict and raises an error; nothing is overwritten. What is already there is examined before it is called the same: a data set or an export that is damaged is an integrity error, not a repetition. A published data set is never modified: new runs mean a new data set.

A run belongs to one data set. `operating_runs.dataset_id` says which, so the same run offered to a database by another data set is a conflict even when its content is the same: accepting it silently would leave the record saying something that is no longer the whole truth.

Identifiers that become part of a path, `dataset_id`, `plant_id` and `run_id`, are lower-case ASCII: a letter or digit first, then letters, digits, `.`, `_` or `-`, at most 100 characters, not ending in a dot and not a reserved device name of Windows. Lower case only, because two names that differ in case are one file on Windows and two on Linux.

### Canonical encoding of the content, version `observations/v1`

The first digest of `Observations` joined the labels with a separator character and appended the arrays without their shapes, so the structure was not part of what was hashed. Two observation sets of different shapes had the same digest when their numbers lined up and a label contained the separator: one row with four inputs against two rows with one input, reproduced before the change. Version 1 encodes structure:

    b"process-transfer/observations/v1\n", then for each field, in this order,
    plant, run, measured_names, measured_units, input_names, input_units,
    sample_period, noise_std, times, measured, inputs:
        the field name as a string, then its value
    a string           b"S", its length in bytes as 8 bytes big-endian, its UTF-8 bytes
    a list of strings  b"L", the number of strings as 8 bytes, then each string
    numbers            b"A", the number of dimensions as 8 bytes, each dimension as 8 bytes,
                       then the values as little-endian IEEE 754 doubles in row order

Every length is explicit, so no two different observation sets share an encoding. Equal hashes mean equal content. The converse holds on one machine and one set of library versions: the last bits of a simulated state may differ elsewhere, and the hash with them.

## Noise streams

The noise of a run is drawn from `SeedSequence(entropy=master seed, spawn_key=(w0, w1, w2, w3, channel))`. The master seed is one integer per data set, kept under `private/`. The four words are the first 16 bytes of the SHA-256 of the UTF-8 bytes of `run_id`, read as four big-endian 32-bit integers. The channel is the index of the state that the sensor reads.

* The assignment is deterministic and needs no registry: the stream of a run follows from its identity. Python's `hash()` is not used; it is salted per process.
* A deliberate repetition has the same `run_id` and the same master seed, so it replays the same noise and reproduces the same content.
* A new, independent realisation is a new `n<k>`, hence a new `run_id` and a new stream.
* Accidental reuse inside a data set is checked when the data set is defined: no two runs may have the same identity or the same stream. A collision of the 128-bit prefix of two different identities is astronomically unlikely, and it is detected rather than assumed away.
* Every word is below 2**32, as the rule for noise keys requires.

The master seed and the streams are recorded in the private provenance, which is enough to rebuild the noise. They are not recorded under `available/`, because the noise could then be regenerated and subtracted. With small integer seeds this is protection against accident, not against an adversary who tries seeds.

## Time

`time_s` is process time in seconds, a 64-bit float, on the clock of the run. Every run that the pipeline generates starts at 0, and `operating_runs.start_time_s` records the first instant whatever it is. It is not a date, has no time zone, and is never the time at which a file was written. No absolute timestamp is stored, because the virtual plants have no calendar; inventing one would add a conversion and nothing else.

`sample_index` is the tick of the sensor clock, an integer: row k of a complete run is at `time_s = start_time_s + k * sampling_period_s`. It is derived when the data are written and verified against `time_s`. Joins, lags and gap detection use the integer, never an equality of floats. A missing reading is a missing tick, not a renumbering.

The instants of a run are those of the simulation, bit for bit, and so are the switching instants of the inputs. A row holds the readings of the state at its instant and the inputs applied from that instant on, until the next row at which they differ: zero-order hold, right-continuous. At a switching instant the row carries the new inputs, and the reading is of a state that has not yet responded to them. Every change of the inputs falls on a row.

## Channels: measured variables and known inputs

The four inputs, q, C_Af, T_f and T_c, are stored with their full history in the same long table as the readings. The table `sensors` therefore describes channels, and `channel_kind` says which kind:

| `channel_kind` | Meaning | `noise_model`, `noise_std` |
|---|---|---|
| `measured` | a sensor reading of a state, with noise | `additive_gaussian`, the standard deviation in SI, zero or more |
| `input` | a known input, exact (D-020) | both null |

`channel_index` is the position of a channel among the channels of its kind for that plant. It fixes the order of the columns when a run is rebuilt from the long table, which the content hash depends on; a database gives no order of its own. Names, order, units, sampling period and noise level belong to the plant and must agree across its runs.

This widens the meaning of `sensors` from instruments to channels. The name is kept because it is one of the five entities of `AGENTS.md`. An input is never presented as a sensor with noise: its noise fields are null, not zero, and the schema enforces the pairing. `sensor_id` is `<plant_id>.<variable_name>`.

Alternatives set aside. Four input columns on `operating_runs` would lose the history. A sixth table, or four, would break the five entities for no gain. Storing only the switching instants would be compact, but would make every alignment a range join and would hide a missing row.

## Quality

`quality_flag` describes a stored record and nothing else. In contract version 1 the only value is 0, which means: the value is present and finite, the instant is finite and on the sampling clock of its channel, and the channel is declared for that plant with a known unit. The writer refuses to store a record that fails any of these, so it never writes another value. The column exists because later milestones will store missing and suspect readings, and will define further values then.

Three notions that must not be confused:

* quality of a record is the flag above. It knows no physics. Integrated balances are not evaluated on noisy readings, and a reading is never clipped or rejected for being negative or outside the range of the true states;
* acceptance of a simulation belongs to the truth. A true trajectory that fails `simulation/checks.py` is never observed and never stored, so every stored run comes from an accepted trajectory. The checks themselves, peaks and balance residuals, stay under `private/`;
* selection of data is what a query does when it builds a training or evaluation set. It is not stored in the data.

Records are checked twice. Before a run enters the database, its rows are loaded into a staging schema without constraints, the quality queries of `sql/quality/` must return no rows, and the content rebuilt from the staged rows must have the hash recorded for the run. The constraints of the main schema are the second line. The whole ingestion of a run is one transaction, so a run is in the database entirely or not at all.

Reading is a line of its own, for whatever wrote the files. A data set is verified when it is opened: the manifest, the exact list of files, schemas, row counts and table digests, and then what the writer enforces, since a file may have been written by something else: identifiers usable in a path, the relations between the tables, the kind and the noise fields of every channel, the quality flag of every row, the extent and sampling period of every run, and the content hash of every run. An export is verified the same way when it is opened, and before the exporter reports one as already present. What fails is refused with a message that names the rule; nothing is repaired. The content hash protects the numbers of a run; the rules protect the metadata around them, and they are rules, not a signature.

## Available information

What an engineer would know, exported through an explicit list of fields:

* `plants`: identifier, name and process type;
* `process_parameters`: reactor volume, density, heat capacity, heat of reaction and the four nominal inputs, from the known part of the plant configuration (`PlantSpec`), in SI. The exporter is given a `PlantSpec` and cannot see the rest of the file;
* `sensors`: channels, units, sampling period and the noise level of the instruments, which is a data sheet value;
* `operating_runs`: protocol, description, duration, number of samples, content hash;
* `measurements`: readings and inputs.

Never under `available/`: the hidden physics (`true_physics`: the true rate law and its parameters, the conductance law and its parameters), exact states, derivatives, measurement errors, the nominal steady state computed from the true model, initial states, the master seed and the noise streams, the acceptance checks of the truth, and copies of the plant configuration files. The nominal values of the modeller's simplified model, k0, E/R and UA, are not known plant parameters either: they are a starting point of a model and belong to M1, not to `process_parameters`.
