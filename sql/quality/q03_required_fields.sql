-- Quality 3. Required fields that are absent: NULL, or text that is empty or blank.
--
-- The noise fields of a channel are not listed here: whether they must be present
-- depends on the kind of channel, which is q05. No row means the check passes.

SELECT 'measurements' AS table_name, field, count(*) AS n_rows
FROM (
    SELECT unnest([
        CASE WHEN plant_id     IS NULL OR length(trim(plant_id))  = 0 THEN 'plant_id'     END,
        CASE WHEN run_id       IS NULL OR length(trim(run_id))    = 0 THEN 'run_id'       END,
        CASE WHEN sensor_id    IS NULL OR length(trim(sensor_id)) = 0 THEN 'sensor_id'    END,
        CASE WHEN sample_index IS NULL THEN 'sample_index' END,
        CASE WHEN time_s       IS NULL THEN 'time_s'       END,
        CASE WHEN value        IS NULL THEN 'value'        END,
        CASE WHEN quality_flag IS NULL THEN 'quality_flag' END
    ]) AS field
    FROM measurements
)
WHERE field IS NOT NULL
GROUP BY field

UNION ALL

SELECT 'operating_runs', field, count(*)
FROM (
    SELECT unnest([
        CASE WHEN run_id         IS NULL OR length(trim(run_id))         = 0 THEN 'run_id'         END,
        CASE WHEN plant_id       IS NULL OR length(trim(plant_id))       = 0 THEN 'plant_id'       END,
        CASE WHEN dataset_id     IS NULL OR length(trim(dataset_id))     = 0 THEN 'dataset_id'     END,
        CASE WHEN operating_mode IS NULL OR length(trim(operating_mode)) = 0 THEN 'operating_mode' END,
        CASE WHEN description    IS NULL OR length(trim(description))    = 0 THEN 'description'    END,
        CASE WHEN content_sha256 IS NULL OR length(trim(content_sha256)) = 0 THEN 'content_sha256' END,
        CASE WHEN start_time_s      IS NULL THEN 'start_time_s'      END,
        CASE WHEN end_time_s        IS NULL THEN 'end_time_s'        END,
        CASE WHEN sampling_period_s IS NULL THEN 'sampling_period_s' END,
        CASE WHEN n_samples         IS NULL THEN 'n_samples'         END
    ]) AS field
    FROM operating_runs
)
WHERE field IS NOT NULL
GROUP BY field

UNION ALL

SELECT 'sensors', field, count(*)
FROM (
    SELECT unnest([
        CASE WHEN sensor_id     IS NULL OR length(trim(sensor_id))     = 0 THEN 'sensor_id'     END,
        CASE WHEN plant_id      IS NULL OR length(trim(plant_id))      = 0 THEN 'plant_id'      END,
        CASE WHEN variable_name IS NULL OR length(trim(variable_name)) = 0 THEN 'variable_name' END,
        CASE WHEN channel_kind  IS NULL OR length(trim(channel_kind))  = 0 THEN 'channel_kind'  END,
        CASE WHEN unit          IS NULL OR length(trim(unit))          = 0 THEN 'unit'          END,
        CASE WHEN channel_index     IS NULL THEN 'channel_index'     END,
        CASE WHEN sampling_period_s IS NULL THEN 'sampling_period_s' END
    ]) AS field
    FROM sensors
)
WHERE field IS NOT NULL
GROUP BY field

UNION ALL

SELECT 'plants', field, count(*)
FROM (
    SELECT unnest([
        CASE WHEN plant_id     IS NULL OR length(trim(plant_id))     = 0 THEN 'plant_id'     END,
        CASE WHEN name         IS NULL OR length(trim(name))         = 0 THEN 'name'         END,
        CASE WHEN process_type IS NULL OR length(trim(process_type)) = 0 THEN 'process_type' END
    ]) AS field
    FROM plants
)
WHERE field IS NOT NULL
GROUP BY field

UNION ALL

SELECT 'process_parameters', field, count(*)
FROM (
    SELECT unnest([
        CASE WHEN plant_id  IS NULL OR length(trim(plant_id))  = 0 THEN 'plant_id'  END,
        CASE WHEN parameter IS NULL OR length(trim(parameter)) = 0 THEN 'parameter' END,
        CASE WHEN unit      IS NULL OR length(trim(unit))      = 0 THEN 'unit'      END,
        CASE WHEN value     IS NULL THEN 'value' END
    ]) AS field
    FROM process_parameters
)
WHERE field IS NOT NULL
GROUP BY field

ORDER BY table_name, field;
