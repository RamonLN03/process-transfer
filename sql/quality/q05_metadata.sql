-- Quality 5. Units and metadata that are incompatible.
--
-- What a channel says about itself must be coherent, and coherent with its runs:
--   * a measured channel has a noise model and a noise level that is zero or more;
--     a known input has neither. A NULL noise is not a noise of zero;
--   * one variable has one unit on every plant, or the plants cannot be compared;
--   * a run and the channels it uses agree on the sampling period;
--   * the quality flag is 0, the only value that contract version 1 defines.
-- No physical criterion is applied to readings here or anywhere else.
-- No row means the check passes.

SELECT 'channel kind is not measured or input' AS problem, sensor_id AS item, coalesce(channel_kind, 'NULL') AS detail
FROM sensors
WHERE channel_kind IS NULL OR channel_kind NOT IN ('measured', 'input')

UNION ALL

SELECT 'measured channel without a valid noise model and level', sensor_id,
       coalesce(noise_model, 'NULL') || ' / ' || coalesce(CAST(noise_std AS VARCHAR), 'NULL')
FROM sensors
WHERE channel_kind = 'measured'
  AND (noise_model IS DISTINCT FROM 'additive_gaussian' OR noise_std IS NULL OR NOT (noise_std >= 0))

UNION ALL

SELECT 'known input presented with noise', sensor_id,
       coalesce(noise_model, 'NULL') || ' / ' || coalesce(CAST(noise_std AS VARCHAR), 'NULL')
FROM sensors
WHERE channel_kind = 'input' AND (noise_model IS NOT NULL OR noise_std IS NOT NULL)

UNION ALL

SELECT 'sampling period is not positive', sensor_id, CAST(sampling_period_s AS VARCHAR)
FROM sensors
WHERE NOT (sampling_period_s > 0)

UNION ALL

SELECT 'one variable with different units on different plants', variable_name,
       string_agg(DISTINCT unit, ' against ' ORDER BY unit)
FROM sensors
GROUP BY variable_name
HAVING count(DISTINCT unit) > 1

UNION ALL

SELECT 'run and channel disagree on the sampling period', m.run_id || ' / ' || m.sensor_id,
       CAST(r.sampling_period_s AS VARCHAR) || ' s against ' || CAST(s.sampling_period_s AS VARCHAR) || ' s'
FROM (SELECT DISTINCT run_id, sensor_id FROM measurements) m
JOIN operating_runs r ON r.run_id = m.run_id
JOIN sensors s ON s.sensor_id = m.sensor_id
WHERE r.sampling_period_s IS DISTINCT FROM s.sampling_period_s

UNION ALL

SELECT 'run with an impossible extent', run_id,
       'from ' || CAST(start_time_s AS VARCHAR) || ' to ' || CAST(end_time_s AS VARCHAR)
       || ' s, ' || CAST(n_samples AS VARCHAR) || ' samples'
FROM operating_runs
WHERE NOT (end_time_s >= start_time_s) OR NOT (n_samples >= 1) OR NOT (sampling_period_s > 0)

UNION ALL

SELECT 'content hash is not a SHA-256', run_id, coalesce(content_sha256, 'NULL')
FROM operating_runs
WHERE content_sha256 IS NULL OR NOT regexp_full_match(content_sha256, '[0-9a-f]{64}')

UNION ALL

SELECT 'quality flag that the contract does not define', run_id || ' / ' || sensor_id,
       'flag ' || CAST(quality_flag AS VARCHAR) || ' on ' || CAST(count(*) AS VARCHAR) || ' rows'
FROM measurements
WHERE quality_flag IS DISTINCT FROM 0
GROUP BY run_id, sensor_id, quality_flag

ORDER BY problem, item;
