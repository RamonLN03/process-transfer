-- Quality 2. References that do not exist, or that are incoherent.
--
-- A measurement must belong to a run and to a channel that exist, and both must be of
-- the plant that the measurement names. A run, a channel and a parameter must belong to
-- a plant that exists. Counting rows per problem keeps the answer short.
-- No row means the check passes.

SELECT 'measurement of a run that does not exist' AS problem, m.run_id AS item, count(*) AS n_rows
FROM measurements m
WHERE NOT EXISTS (SELECT 1 FROM operating_runs r WHERE r.run_id = m.run_id)
GROUP BY m.run_id

UNION ALL

SELECT 'measurement of a channel that does not exist', m.sensor_id, count(*)
FROM measurements m
WHERE NOT EXISTS (SELECT 1 FROM sensors s WHERE s.sensor_id = m.sensor_id)
GROUP BY m.sensor_id

UNION ALL

SELECT 'measurement names another plant than its run',
       m.run_id || ': ' || m.plant_id || ' against ' || r.plant_id, count(*)
FROM measurements m
JOIN operating_runs r ON r.run_id = m.run_id
WHERE m.plant_id IS DISTINCT FROM r.plant_id
GROUP BY m.run_id, m.plant_id, r.plant_id

UNION ALL

SELECT 'measurement uses the channel of another plant',
       m.run_id || ' / ' || m.sensor_id || ': ' || m.plant_id || ' against ' || s.plant_id, count(*)
FROM measurements m
JOIN sensors s ON s.sensor_id = m.sensor_id
WHERE m.plant_id IS DISTINCT FROM s.plant_id
GROUP BY m.run_id, m.sensor_id, m.plant_id, s.plant_id

UNION ALL

SELECT 'run of a plant that does not exist', r.run_id || ' / ' || coalesce(r.plant_id, 'NULL'), count(*)
FROM operating_runs r
WHERE NOT EXISTS (SELECT 1 FROM plants p WHERE p.plant_id = r.plant_id)
GROUP BY r.run_id, r.plant_id

UNION ALL

SELECT 'channel of a plant that does not exist', s.sensor_id || ' / ' || coalesce(s.plant_id, 'NULL'), count(*)
FROM sensors s
WHERE NOT EXISTS (SELECT 1 FROM plants p WHERE p.plant_id = s.plant_id)
GROUP BY s.sensor_id, s.plant_id

UNION ALL

SELECT 'parameter of a plant that does not exist', pp.parameter || ' / ' || coalesce(pp.plant_id, 'NULL'), count(*)
FROM process_parameters pp
WHERE NOT EXISTS (SELECT 1 FROM plants p WHERE p.plant_id = pp.plant_id)
GROUP BY pp.parameter, pp.plant_id

ORDER BY problem, item;
