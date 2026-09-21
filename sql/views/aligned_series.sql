-- Alignment: one row per run and tick, the two readings and the four known inputs of
-- the CSTR side by side. This is the series a dynamic model will read, at the original
-- sampling of six seconds.
--
-- Convention of a row (docs/data_contract.md): the readings are of the state at time_s,
-- and the inputs are those applied from time_s on, until the next row at which they
-- differ (zero-order hold, right-continuous). At a switching instant the row carries the
-- new inputs and a reading that has not yet responded to them.
--
-- How rows are kept from multiplying. The long table is not joined to itself. Joining
-- the C_A rows of a run to its T rows on run_id alone would give n * n rows instead of
-- n; here every channel is folded into its own column by an aggregate with a FILTER,
-- grouped by (run, tick), so there is exactly one row per tick whatever is stored. The
-- only join is to sensors, many rows to one through a unique key, which cannot add rows.
--
-- Nothing is filled in. A channel that is absent at a tick gives NULL in its column and
-- shows in n_channels; a row stored twice shows in n_rows. "complete" says that the six
-- channels are there once each and agree on the instant. Readings are whatever the
-- sensors gave, negative values included.

CREATE OR REPLACE VIEW aligned_series AS
SELECT
    m.plant_id,
    m.run_id,
    m.sample_index,
    min(m.time_s)                                             AS time_s,
    max(m.value) FILTER (WHERE s.variable_name = 'C_A')       AS "C_A",
    max(m.value) FILTER (WHERE s.variable_name = 'T')         AS "T",
    max(m.value) FILTER (WHERE s.variable_name = 'q')         AS "q",
    max(m.value) FILTER (WHERE s.variable_name = 'C_Af')      AS "C_Af",
    max(m.value) FILTER (WHERE s.variable_name = 'T_f')       AS "T_f",
    max(m.value) FILTER (WHERE s.variable_name = 'T_c')       AS "T_c",
    count(DISTINCT m.sensor_id)                               AS n_channels,
    count(*)                                                  AS n_rows,
    count(*) = 6 AND count(DISTINCT m.sensor_id) = 6
        AND min(m.time_s) = max(m.time_s)                     AS complete
FROM measurements m
JOIN sensors s ON s.plant_id = m.plant_id AND s.sensor_id = m.sensor_id
GROUP BY m.plant_id, m.run_id, m.sample_index;
