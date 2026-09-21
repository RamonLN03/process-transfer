# SQL

The schema and the queries of the project, as files that can be read and run on their own. The Python side (`src/process_transfer/data/database.py`) only executes them; there is no ORM and no query hidden in a string somewhere else. The data contract they implement is `docs/data_contract.md`.

    schema/001_tables.sql         the five tables, with keys, relations and checks
    schema/002_staging.sql        the same tables without constraints, in a schema called staging
    quality/q01 ... q07           data-quality checks; each returns the defective rows
    views/aligned_series.sql      one row per run and tick with C_A, T and the four inputs
    views/lagged_measurements.sql the value of a channel one period before, never across a run or a gap
    analysis/source_target_comparison.sql   observed differences between two plants, paired explicitly
    analysis/block_average.sql              averages over blocks of ticks; inputs never averaged across a change

## How they are used

Ingestion of a run is one transaction: its rows are loaded into `staging`, every query of `quality/` must return no row, the content rebuilt from the staged rows must have the hash recorded for the run, and only then are the rows inserted into the main schema. The constraints of `001_tables.sql` are the second line of defence.

The quality queries name their tables without a schema. Run them with `USE staging` before ingestion, or with `USE main` at any time. A query that returns no row has passed; a row describes a defect in words.

`source_target_comparison.sql` takes `$source` and `$target`, two plant identifiers. `block_average.sql` takes `$block_ticks`, a whole number of ticks.

## Conventions

* Time is `time_s`, process time in seconds, and `sample_index`, the integer tick of the sensor clock. Joins, lags and gap detection use the integer.
* A row of `aligned_series` holds the readings at its instant and the inputs applied from that instant on (zero-order hold, right-continuous).
* Readings are what the sensors gave. A negative concentration reading is a valid record, and no query applies a physical criterion to a reading.
* Differences between plants are observed differences. Nothing here reads a hidden quantity, because nothing hidden is in the database.
* The six-second series is the reference. Block averages are a derived summary.

From a shell, against a database built by the pipeline:

    duckdb data/available/databases/m0-e05.duckdb "USE main; .read sql/quality/q06_cadence.sql"
