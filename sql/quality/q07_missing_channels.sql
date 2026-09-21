-- Quality 7. A channel that is absent at an instant at which the run has other channels.
--
-- An aligned row needs every channel of the plant at its tick: the readings and the four
-- inputs. The expected set is every tick of the run on every channel of its plant; what
-- is missing from it is reported, tick by tick, never filled in.
-- No row means the check passes.

WITH ticks AS (
    SELECT DISTINCT plant_id, run_id, sample_index
    FROM measurements
),
expected AS (
    SELECT t.run_id, t.sample_index, s.sensor_id
    FROM ticks t
    JOIN sensors s ON s.plant_id = t.plant_id
)
SELECT 'channel absent at an instant of its run' AS problem,
       e.run_id || ' / ' || e.sensor_id AS item,
       'tick ' || CAST(e.sample_index AS VARCHAR) AS detail
FROM expected e
WHERE NOT EXISTS (
    SELECT 1
    FROM measurements m
    WHERE m.run_id = e.run_id AND m.sensor_id = e.sensor_id AND m.sample_index = e.sample_index
)
ORDER BY item, e.sample_index;
