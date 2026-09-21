-- Lags: for every measurement, the value of the same channel one sampling period before.
--
-- The window is partitioned by plant, run and channel, so a lag never crosses from one
-- run into another, nor from one channel into another: the first row of every run has
-- no previous value.
--
-- A lag is "one period ago" only if the previous row is the previous tick. After a gap,
-- lag() would return the last row before the gap, 12 s or 60 s earlier, and nothing in
-- the value would say so. value_one_period_ago is therefore NULL unless the ticks are
-- consecutive, and ticks_since_previous says how far back the previous row is. Ticks are
-- integers, so this is exact.

CREATE OR REPLACE VIEW lagged_measurements AS
SELECT
    plant_id,
    run_id,
    sensor_id,
    sample_index,
    time_s,
    value,
    lag(value) OVER w                                                  AS previous_stored_value,
    sample_index - lag(sample_index) OVER w                            AS ticks_since_previous,
    time_s - lag(time_s) OVER w                                        AS seconds_since_previous,
    CASE WHEN sample_index - lag(sample_index) OVER w = 1
         THEN lag(value) OVER w END                                    AS value_one_period_ago
FROM measurements
WINDOW w AS (PARTITION BY plant_id, run_id, sensor_id ORDER BY sample_index);
