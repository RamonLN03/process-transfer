-- Quality 6. Cadence in time, gaps and counts, per run and channel.
--
-- A complete run has the ticks 0, 1, ..., n_samples - 1 on every channel, and the
-- instant of tick k is start_time_s + k * sampling_period_s. Ticks are integers, so a
-- gap is found by arithmetic on integers, not by comparing floats.
--
-- The instant of a tick and the instant that was stored are the same number computed by
-- two routes, a product and a sum each, so they may differ in their last bits: 0.1 * 3
-- is not 0.3. They must agree to four units in the last place. nextafter gives the size
-- of that unit at the magnitude of the numbers compared; this is the resolution of the
-- arithmetic, the same rule as process_transfer.sampling_clock, not a tolerance.
-- No row means the check passes.

WITH per_channel AS (
    SELECT m.run_id, m.sensor_id,
           count(*)            AS n_rows,
           min(m.sample_index) AS first_tick,
           max(m.sample_index) AS last_tick
    FROM measurements m
    GROUP BY m.run_id, m.sensor_id
),
steps AS (
    SELECT run_id, sensor_id, sample_index, time_s,
           lag(sample_index) OVER w AS previous_tick,
           lag(time_s)       OVER w AS previous_time_s
    FROM measurements
    WINDOW w AS (PARTITION BY run_id, sensor_id ORDER BY sample_index)
),
on_the_clock AS (
    SELECT m.run_id, m.sensor_id, m.sample_index, m.time_s,
           r.start_time_s + m.sample_index * r.sampling_period_s AS expected_time_s
    FROM measurements m
    JOIN operating_runs r ON r.run_id = m.run_id
)

SELECT 'number of rows is not the number of samples of the run' AS problem,
       c.run_id || ' / ' || c.sensor_id AS item,
       CAST(c.n_rows AS VARCHAR) || ' rows, ' || CAST(r.n_samples AS VARCHAR) || ' samples' AS detail
FROM per_channel c
JOIN operating_runs r ON r.run_id = c.run_id
WHERE c.n_rows <> r.n_samples

UNION ALL

SELECT 'ticks do not run from 0 to n_samples - 1', c.run_id || ' / ' || c.sensor_id,
       'from ' || CAST(c.first_tick AS VARCHAR) || ' to ' || CAST(c.last_tick AS VARCHAR)
FROM per_channel c
JOIN operating_runs r ON r.run_id = c.run_id
WHERE c.first_tick <> 0 OR c.last_tick <> r.n_samples - 1

UNION ALL

SELECT 'gap: ticks missing between two rows', run_id || ' / ' || sensor_id,
       CAST(sample_index - previous_tick - 1 AS VARCHAR) || ' missing after tick ' || CAST(previous_tick AS VARCHAR)
FROM steps
WHERE sample_index - previous_tick > 1

UNION ALL

SELECT 'time does not increase with the tick', run_id || ' / ' || sensor_id,
       'tick ' || CAST(sample_index AS VARCHAR) || ' at ' || CAST(time_s AS VARCHAR)
       || ' s after ' || CAST(previous_time_s AS VARCHAR) || ' s'
FROM steps
WHERE NOT (time_s > previous_time_s)

UNION ALL

SELECT 'instant is not on the sampling clock of its run', run_id || ' / ' || sensor_id,
       'tick ' || CAST(sample_index AS VARCHAR) || ' at ' || CAST(time_s AS VARCHAR)
       || ' s, expected ' || CAST(expected_time_s AS VARCHAR) || ' s'
FROM on_the_clock
WHERE NOT (
    abs(time_s - expected_time_s) <= 4 * (
        nextafter(greatest(abs(time_s), abs(expected_time_s)), 'Infinity'::DOUBLE)
        - greatest(abs(time_s), abs(expected_time_s))
    )
)

UNION ALL

SELECT 'run does not end at its last instant', r.run_id,
       'end_time_s ' || CAST(r.end_time_s AS VARCHAR) || ' s, last row at ' || CAST(max(m.time_s) AS VARCHAR) || ' s'
FROM operating_runs r
JOIN measurements m ON m.run_id = r.run_id
GROUP BY r.run_id, r.end_time_s
HAVING max(m.time_s) IS DISTINCT FROM r.end_time_s

UNION ALL

SELECT 'run without any measurement', r.run_id, CAST(r.n_samples AS VARCHAR) || ' samples recorded'
FROM operating_runs r
WHERE NOT EXISTS (SELECT 1 FROM measurements m WHERE m.run_id = r.run_id)

ORDER BY problem, item;
