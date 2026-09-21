-- Quality 1. Duplicates: a key that appears more than once.
--
-- Returns one row per duplicated key. No row means the check passes.
-- Tables are named without a schema: run with "USE staging" or "USE main".

SELECT 'measurement stored more than once' AS problem,
       run_id || ' / ' || sensor_id || ' / tick ' || sample_index AS item,
       count(*) AS occurrences
FROM measurements
GROUP BY run_id, sensor_id, sample_index
HAVING count(*) > 1

UNION ALL

SELECT 'run defined more than once', run_id, count(*)
FROM operating_runs
GROUP BY run_id
HAVING count(*) > 1

UNION ALL

SELECT 'channel defined more than once', sensor_id, count(*)
FROM sensors
GROUP BY sensor_id
HAVING count(*) > 1

UNION ALL

SELECT 'variable with more than one channel on a plant', plant_id || ' / ' || variable_name, count(*)
FROM sensors
GROUP BY plant_id, variable_name
HAVING count(*) > 1

UNION ALL

SELECT 'two channels of one kind share a position', plant_id || ' / ' || channel_kind || ' / ' || channel_index, count(*)
FROM sensors
GROUP BY plant_id, channel_kind, channel_index
HAVING count(*) > 1

UNION ALL

SELECT 'plant defined more than once', plant_id, count(*)
FROM plants
GROUP BY plant_id
HAVING count(*) > 1

UNION ALL

SELECT 'parameter given more than once', plant_id || ' / ' || parameter, count(*)
FROM process_parameters
GROUP BY plant_id, parameter
HAVING count(*) > 1

ORDER BY problem, item;
