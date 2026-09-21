"""The SQL of ``sql/`` on small examples whose answers are known by hand.

Quality queries are run against the staging schema, which has no constraints, so that a
defect can be put there on purpose and must be found. A valid set must give no finding
at all, and each defect must be reported by the query that exists for it.
"""

from pathlib import Path

import duckdb
import pytest

from process_transfer.data.database import (
    block_average,
    compare_plants,
    connect,
    failed_checks,
    quality_report,
    sql_files,
)

SHA = "a" * 64
CHANNELS = (  # variable, kind, index, unit
    ("C_A", "measured", 0, "mol/m^3"),
    ("T", "measured", 1, "K"),
    ("q", "input", 0, "m^3/s"),
    ("C_Af", "input", 1, "mol/m^3"),
    ("T_f", "input", 2, "K"),
    ("T_c", "input", 3, "K"),
)


def fill(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    runs: dict[str, dict[str, list[float]]],
) -> None:
    """Small valid tables. ``runs`` maps a run_id, ``<plant>.<definition>``, to the series
    of its six channels; a channel that is not given is constant."""
    defaults = {"C_A": 200.0, "T": 350.0, "q": 0.002, "C_Af": 500.0, "T_f": 350.0, "T_c": 337.5}
    for plant in sorted({run_id.split(".")[0] for run_id in runs}):
        connection.execute(
            f"INSERT INTO {schema_name}.plants VALUES (?, ?, 'cstr')", [plant, plant]
        )
        connection.execute(
            f"INSERT INTO {schema_name}.process_parameters "
            "VALUES (?, 'reactor_volume', 0.1, 'm^3')",
            [plant],
        )
        for variable, kind, index, unit in CHANNELS:
            noise = ("additive_gaussian", 5.0 if variable == "C_A" else 0.5)
            connection.execute(
                f"INSERT INTO {schema_name}.sensors VALUES (?, ?, ?, ?, ?, ?, 6.0, ?, ?)",
                [f"{plant}.{variable}", plant, variable, kind, index, unit]
                + list(noise if kind == "measured" else (None, None)),
            )
    for run_id, series in runs.items():
        plant = run_id.split(".")[0]
        n = max(len(values) for values in series.values())
        connection.execute(
            f"INSERT INTO {schema_name}.operating_runs VALUES (?, ?, 'tiny', 'p3', 'a tiny run', "
            "0.0, ?, 6.0, ?, ?)",
            [run_id, plant, 6.0 * (n - 1), n, SHA],
        )
        for variable, _, _, _ in CHANNELS:
            values = series.get(variable, [defaults[variable]] * n)
            for tick, value in enumerate(values):
                connection.execute(
                    f"INSERT INTO {schema_name}.measurements VALUES (?, ?, ?, ?, ?, ?, 0)",
                    [plant, run_id, f"{plant}.{variable}", tick, 6.0 * tick, value],
                )


TWO_PLANTS = {
    "source.p3.e0.x1.n0": {"C_A": [10.0, 20.0, 30.0, 40.0], "T": [350.0, 351.0, 352.0, 353.0]},
    "target.p3.e0.x1.n0": {"C_A": [11.0, 22.0, 33.0, 44.0], "T": [355.0, 355.5, 356.0, 356.5]},
}


@pytest.fixture
def staged() -> duckdb.DuckDBPyConnection:
    connection = connect()
    fill(connection, "staging", TWO_PLANTS)
    return connection


@pytest.fixture
def stored() -> duckdb.DuckDBPyConnection:
    connection = connect()
    fill(connection, "main", TWO_PLANTS)
    return connection


# --------------------------------------------------------------------------- #
# A. Quality
# --------------------------------------------------------------------------- #


def test_the_quality_queries_are_the_seven_files_and_a_valid_set_passes_them_all(
    staged: duckdb.DuckDBPyConnection, stored: duckdb.DuckDBPyConnection
) -> None:
    assert [Path(name).stem for name in sql_files("quality")] == [
        "q01_duplicates",
        "q02_references",
        "q03_required_fields",
        "q04_non_finite",
        "q05_metadata",
        "q06_cadence",
        "q07_missing_channels",
    ]
    for connection, schema_name in ((staged, "staging"), (stored, "main")):
        report = quality_report(connection, schema_name)
        assert len(report) == 7 and failed_checks(report) == {}
        assert connection.execute("SELECT current_schema()").fetchone()[0] == "main"
    assert failed_checks(quality_report(staged, "main")) == {}  # empty tables pass as well
    with pytest.raises(ValueError, match="'main' or 'staging'"):
        quality_report(staged, "private")


RUN = "target.p3.e0.x1.n0"
M = (
    "INSERT INTO staging.measurements VALUES "
    f"('target', '{RUN}', 'target.T', {{0}}, {{1}}, {{2}}, {{3}})"
)
T1 = "WHERE sample_index = 1 AND sensor_id = 'target.T'"  # one reading of the target
OF_TARGET = "WHERE run_id LIKE 'target%'"
DEFECTS = [
    # (what is done to the staged rows, the query that must find it, a word of its finding)
    (M.format(2, 12.0, 356.0, 0), "q01_duplicates", "measurement stored more than once"),
    (
        "INSERT INTO staging.sensors VALUES ('target.T', 'target', 'T', 'measured', 1, 'K', 6.0, "
        "'additive_gaussian', 0.5)",
        "q01_duplicates",
        "channel defined more than once",
    ),
    (
        f"UPDATE staging.measurements SET run_id = 'target.p3.e9.x1.n0' {T1}",
        "q02_references",
        "measurement of a run that does not exist",
    ),
    (
        f"UPDATE staging.measurements SET sensor_id = 'target.pH' {T1}",
        "q02_references",
        "measurement of a channel that does not exist",
    ),
    (
        f"UPDATE staging.measurements SET sensor_id = 'source.T' {T1}",
        "q02_references",
        "measurement uses the channel of another plant",
    ),
    (
        f"UPDATE staging.measurements SET plant_id = 'source' {T1}",
        "q02_references",
        "measurement names another plant than its run",
    ),
    (
        f"UPDATE staging.operating_runs SET plant_id = 'nowhere' {OF_TARGET}",
        "q02_references",
        "run of a plant that does not exist",
    ),
    (
        f"UPDATE staging.measurements SET value = NULL {T1}",
        "q03_required_fields",
        "value",
    ),
    (
        f"UPDATE staging.operating_runs SET description = '   ' {OF_TARGET}",
        "q03_required_fields",
        "description",
    ),
    (
        f"UPDATE staging.measurements SET value = 'NaN' {T1}",
        "q04_non_finite",
        "measurements.value",
    ),
    (
        "UPDATE staging.measurements SET value = '-Infinity' WHERE sample_index = 1 AND "
        "sensor_id = 'target.C_A'",
        "q04_non_finite",
        "measurements.value",
    ),
    (
        "UPDATE staging.process_parameters SET value = 'Infinity' WHERE plant_id = 'target'",
        "q04_non_finite",
        "process_parameters.value",
    ),
    (
        "UPDATE staging.sensors SET noise_model = 'additive_gaussian', noise_std = 0.0 WHERE "
        "sensor_id = 'target.T_c'",
        "q05_metadata",
        "known input presented with noise",
    ),
    (
        "UPDATE staging.sensors SET noise_std = NULL WHERE sensor_id = 'target.T'",
        "q05_metadata",
        "measured channel without a valid noise model and level",
    ),
    (
        "UPDATE staging.sensors SET noise_std = -0.5 WHERE sensor_id = 'target.T'",
        "q05_metadata",
        "measured channel without a valid noise model and level",
    ),
    (
        "UPDATE staging.sensors SET unit = 'mol/L' WHERE sensor_id = 'target.C_A'",
        "q05_metadata",
        "one variable with different units on different plants",
    ),
    (
        "UPDATE staging.sensors SET sampling_period_s = 60.0 WHERE sensor_id = 'target.T'",
        "q05_metadata",
        "run and channel disagree on the sampling period",
    ),
    (
        "UPDATE staging.sensors SET channel_kind = 'disturbance' WHERE sensor_id = 'target.q'",
        "q05_metadata",
        "channel kind is not measured or input",
    ),
    (
        f"UPDATE staging.measurements SET quality_flag = 3 {T1}",
        "q05_metadata",
        "quality flag that the contract does not define",
    ),
    (
        f"UPDATE staging.operating_runs SET content_sha256 = 'not-a-hash' {OF_TARGET}",
        "q05_metadata",
        "content hash is not a SHA-256",
    ),
    (
        f"DELETE FROM staging.measurements {OF_TARGET} AND sample_index = 2",
        "q06_cadence",
        "gap: ticks missing between two rows",
    ),
    (
        f"DELETE FROM staging.measurements {OF_TARGET} AND sample_index = 3",
        "q06_cadence",
        "ticks do not run from 0 to n_samples - 1",
    ),
    (
        f"UPDATE staging.measurements SET time_s = 12.25 {OF_TARGET} AND sample_index = 2",
        "q06_cadence",
        "instant is not on the sampling clock of its run",
    ),
    (
        f"UPDATE staging.measurements SET time_s = 3.0 {OF_TARGET} AND sample_index = 2",
        "q06_cadence",
        "time does not increase with the tick",
    ),
    (
        f"UPDATE staging.operating_runs SET end_time_s = 24.0 {OF_TARGET}",
        "q06_cadence",
        "run does not end at its last instant",
    ),
    (
        "DELETE FROM staging.measurements WHERE sensor_id = 'target.T_c' AND sample_index = 1",
        "q07_missing_channels",
        "channel absent at an instant of its run",
    ),
]


@pytest.mark.parametrize(("defect", "query", "finding"), DEFECTS, ids=[d[2] for d in DEFECTS])
def test_every_defect_put_in_on_purpose_is_found_by_its_query(
    defect: str, query: str, finding: str, staged: duckdb.DuckDBPyConnection
) -> None:
    staged.execute(defect)
    failed = failed_checks(quality_report(staged, "staging"))
    assert query in failed, failed
    assert any(finding in str(value) for row in failed[query] for value in row.values()), failed[
        query
    ]


def test_what_is_not_a_defect_is_not_reported(staged: duckdb.DuckDBPyConnection) -> None:
    """A negative concentration reading, a reading far above any true state, and an
    instant that differs from the clock in its last bit are all valid records."""
    update = (
        "UPDATE staging.measurements SET value = {0} WHERE sensor_id = '{1}' AND sample_index = 1"
    )
    staged.execute(update.format(-12.5, "target.C_A"))
    staged.execute(update.format(1.0e6, "target.T"))
    staged.execute(
        "UPDATE staging.measurements SET time_s = nextafter(18.0, 'Infinity'::DOUBLE) WHERE "
        "run_id LIKE 'target%' AND sample_index = 3"
    )
    one_bit_later = "nextafter(18.0, 'Infinity'::DOUBLE)"
    staged.execute(f"UPDATE staging.operating_runs SET end_time_s = {one_bit_later} {OF_TARGET}")
    assert failed_checks(quality_report(staged, "staging")) == {}


def test_the_finding_of_a_gap_says_where(staged: duckdb.DuckDBPyConnection) -> None:
    staged.execute(
        "DELETE FROM staging.measurements WHERE sensor_id = 'target.T' AND sample_index IN (1, 2)"
    )
    findings = failed_checks(quality_report(staged, "staging"))["q06_cadence"]
    gap = [row for row in findings if row["problem"].startswith("gap")]
    assert gap == [
        {
            "problem": "gap: ticks missing between two rows",
            "item": "target.p3.e0.x1.n0 / target.T",
            "detail": "2 missing after tick 0",
        }
    ]
    counts = [row for row in findings if row["problem"].startswith("number of rows")]
    assert counts[0]["detail"] == "2 rows, 4 samples"


# --------------------------------------------------------------------------- #
# B. Alignment
# --------------------------------------------------------------------------- #


def test_alignment_gives_one_row_per_tick_with_the_six_channels(
    stored: duckdb.DuckDBPyConnection,
) -> None:
    rows = stored.execute(
        "SELECT * FROM aligned_series WHERE run_id = 'target.p3.e0.x1.n0' ORDER BY sample_index"
    ).fetchall()
    names = [column[0] for column in stored.description]
    assert names == [
        "plant_id",
        "run_id",
        "sample_index",
        "time_s",
        "C_A",
        "T",
        "q",
        "C_Af",
        "T_f",
        "T_c",
        "n_channels",
        "n_rows",
        "complete",
    ]
    assert len(rows) == 4  # one per tick, not 4 * 4 and not 4 * 6
    assert rows[2] == (
        "target",
        "target.p3.e0.x1.n0",
        2,
        12.0,
        33.0,
        356.0,
        0.002,
        500.0,
        350.0,
        337.5,
        6,
        6,
        True,
    )
    assert stored.execute("SELECT count(*) FROM aligned_series").fetchone()[0] == 8

    # the join that multiplies rows, for comparison: C_A rows to T rows on the run alone
    multiplied = stored.execute(
        "SELECT count(*) FROM measurements a JOIN measurements b ON a.run_id = b.run_id "
        "WHERE a.sensor_id = 'target.C_A' AND b.sensor_id = 'target.T'"
    ).fetchone()[0]
    assert multiplied == 16


def test_a_channel_absent_at_an_instant_is_shown_and_not_filled_in(
    stored: duckdb.DuckDBPyConnection,
) -> None:
    stored.execute("DELETE FROM measurements WHERE sensor_id = 'target.T_c' AND sample_index = 1")
    row = stored.execute(
        'SELECT "T_c", "T_f", n_channels, n_rows, complete FROM aligned_series '
        "WHERE run_id = 'target.p3.e0.x1.n0' AND sample_index = 1"
    ).fetchone()
    assert row == (None, 350.0, 5, 5, False)
    missing = failed_checks(quality_report(stored))["q07_missing_channels"]
    assert missing == [
        {
            "problem": "channel absent at an instant of its run",
            "item": "target.p3.e0.x1.n0 / target.T_c",
            "detail": "tick 1",
        }
    ]


# --------------------------------------------------------------------------- #
# C. Windows and lags
# --------------------------------------------------------------------------- #


def test_a_lag_never_crosses_a_run_and_never_hides_a_gap() -> None:
    connection = connect()
    fill(
        connection,
        "main",
        {
            "target.p3.e0.x1.n0": {"T": [350.0, 351.0, 352.0, 353.0, 354.0]},
            "target.p3.e1.x1.n0": {"T": [360.0, 361.0, 362.0]},
        },
    )
    connection.execute(
        "DELETE FROM measurements WHERE run_id = 'target.p3.e0.x1.n0' AND sample_index = 2"
    )
    rows = connection.execute(
        "SELECT run_id, sample_index, value, value_one_period_ago, previous_stored_value, "
        "ticks_since_previous, seconds_since_previous FROM lagged_measurements "
        "WHERE sensor_id = 'target.T' ORDER BY run_id, sample_index"
    ).fetchall()
    assert rows == [
        ("target.p3.e0.x1.n0", 0, 350.0, None, None, None, None),
        ("target.p3.e0.x1.n0", 1, 351.0, 350.0, 350.0, 1, 6.0),
        # tick 2 is missing: the previous stored value is 12 s old and is not offered as a lag
        ("target.p3.e0.x1.n0", 3, 353.0, None, 351.0, 2, 12.0),
        ("target.p3.e0.x1.n0", 4, 354.0, 353.0, 353.0, 1, 6.0),
        # a new run starts: nothing of the previous run reaches it
        ("target.p3.e1.x1.n0", 0, 360.0, None, None, None, None),
        ("target.p3.e1.x1.n0", 1, 361.0, 360.0, 360.0, 1, 6.0),
        ("target.p3.e1.x1.n0", 2, 362.0, 361.0, 361.0, 1, 6.0),
    ]
    # nor does one channel reach another
    first_rows = connection.execute(
        "SELECT count(*) FROM lagged_measurements "
        "WHERE sample_index = 0 AND previous_stored_value IS NOT NULL"
    ).fetchone()[0]
    assert first_rows == 0


# --------------------------------------------------------------------------- #
# D. Source against target
# --------------------------------------------------------------------------- #


def test_the_comparison_pairs_runs_and_ticks_explicitly(stored: duckdb.DuckDBPyConnection) -> None:
    (row,) = compare_plants(stored, "source", "target")
    assert (row["definition"], row["source_run"], row["target_run"]) == (
        "p3.e0.x1.n0",
        "source.p3.e0.x1.n0",
        "target.p3.e0.x1.n0",
    )
    assert (row["n_paired"], row["coverage"], row["n_same_instant"], row["n_same_inputs"]) == (
        4,
        1.0,
        4,
        4,
    )
    # target minus source: 1, 2, 3, 4 in C_A and 5, 4.5, 4, 3.5 in T
    assert row["observed_difference_c_a_mean"] == pytest.approx(2.5)
    assert row["observed_difference_c_a_std"] == pytest.approx(1.2909944487358056)
    assert (row["observed_difference_c_a_min"], row["observed_difference_c_a_max"]) == (1.0, 4.0)
    assert row["observed_difference_t_mean"] == pytest.approx(4.25)
    assert (row["observed_difference_t_min"], row["observed_difference_t_max"]) == (3.5, 5.0)
    assert not any("error" in name or "true" in name for name in row)  # observed differences only
    assert compare_plants(stored, "target", "source")[0][
        "observed_difference_c_a_mean"
    ] == pytest.approx(-2.5)
    assert compare_plants(stored, "source", "elsewhere") == []


def test_the_comparison_reports_what_could_not_be_paired() -> None:
    connection = connect()
    runs = {
        "source.p3.e0.x1.n0": {"C_A": [10.0, 20.0, 30.0, 40.0]},
        "target.p3.e0.x1.n0": {"C_A": [11.0, 22.0, 33.0], "T_c": [337.5, 342.5, 337.5]},
        "target.p3.e5.x1.n0": {"C_A": [1.0, 2.0]},  # no run of the source has this definition
    }
    fill(connection, "main", runs)
    (row,) = compare_plants(connection, "source", "target")
    assert row["n_paired"] == 3 and row["coverage"] == 0.75  # three of the four source ticks
    assert row["n_same_inputs"] == 2 and row["n_used"] == 2  # the coolant differs at tick 1
    assert row["observed_difference_c_a_mean"] == pytest.approx(2.0)  # ticks 0 and 2: 1 and 3


# --------------------------------------------------------------------------- #
# E. Aggregation in time
# --------------------------------------------------------------------------- #


def test_block_averages_have_known_edges_labels_coverage_and_inputs() -> None:
    connection = connect()
    fill(
        connection,
        "main",
        {
            "target.p3.e0.x1.n0": {
                "C_A": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                "T_c": [337.5, 337.5, 337.5, 337.5, 342.5, 342.5, 342.5],  # steps at tick 4
            }
        },
    )
    blocks = block_average(connection, 3)
    assert [(b["block_index"], b["block_start_s"], b["block_end_s"]) for b in blocks] == [
        (0, 0.0, 18.0),
        (1, 18.0, 36.0),
        (2, 36.0, 54.0),
    ]
    assert [(b["n_expected"], b["n_present"], b["coverage"]) for b in blocks] == [
        (3, 3, 1.0),
        (3, 3, 1.0),
        (1, 1, 1.0),  # the last block of the run is shorter, and says so
    ]
    assert [b["C_A_mean"] for b in blocks] == [20.0, 50.0, 70.0]
    # the coolant steps inside block 1: it is not averaged into 340.83, it is withheld
    assert [(b["T_c"], b["inputs_constant"]) for b in blocks] == [
        (337.5, True),
        (None, False),
        (342.5, True),
    ]
    assert [b["T_f"] for b in blocks] == [350.0, 350.0, 350.0]

    connection.execute("DELETE FROM measurements WHERE sample_index = 1")
    gapped = block_average(connection, 3)[0]
    assert (gapped["n_expected"], gapped["n_present"]) == (3, 2)
    assert gapped["coverage"] == pytest.approx(2.0 / 3.0)
    assert gapped["C_A_mean"] == 20.0 and gapped["block_start_s"] == 0.0  # mean of 10 and 30

    assert len(block_average(connection, 1)) == 6  # one tick per block: the series itself
    assert len(block_average(connection, 1000)) == 1


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, None, "3"])
def test_a_block_is_a_whole_number_of_ticks(bad: object, stored: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError, match="block_ticks"):
        block_average(stored, bad)  # type: ignore[arg-type]
