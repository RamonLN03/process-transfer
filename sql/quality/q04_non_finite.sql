-- Quality 4. Numbers that are not finite: NaN, +Infinity, -Infinity.
--
-- A NULL is an absent field and belongs to q03; isfinite(NULL) is NULL and is not
-- counted here. A reading that is negative, or outside the range of the true states, is
-- finite and is not a defect: readings are stored as the sensors gave them.
-- No row means the check passes.

SELECT 'measurements.value' AS field, run_id || ' / ' || sensor_id AS item, count(*) AS n_rows
FROM measurements
WHERE NOT isfinite(value)
GROUP BY run_id, sensor_id

UNION ALL

SELECT 'measurements.time_s', run_id || ' / ' || sensor_id, count(*)
FROM measurements
WHERE NOT isfinite(time_s)
GROUP BY run_id, sensor_id

UNION ALL

SELECT 'process_parameters.value', plant_id || ' / ' || parameter, count(*)
FROM process_parameters
WHERE NOT isfinite(value)
GROUP BY plant_id, parameter

UNION ALL

SELECT 'sensors.sampling_period_s', sensor_id, count(*)
FROM sensors
WHERE NOT isfinite(sampling_period_s)
GROUP BY sensor_id

UNION ALL

SELECT 'sensors.noise_std', sensor_id, count(*)
FROM sensors
WHERE NOT isfinite(noise_std)
GROUP BY sensor_id

UNION ALL

SELECT 'operating_runs times and period', run_id, count(*)
FROM operating_runs
WHERE NOT isfinite(start_time_s) OR NOT isfinite(end_time_s) OR NOT isfinite(sampling_period_s)
GROUP BY run_id

ORDER BY field, item;
