-- The five tables of M0 (docs/data_contract.md, contract version 1). SI units throughout.
--
-- No table holds a wall-clock time. time_s is process time in seconds on the clock of a
-- run; sample_index is the integer tick of the sensor clock, and it is what joins, lags
-- and gap detection use, never an equality of floats.
--
-- The constraints below are the second line of defence. The first is the staging schema
-- (002_staging.sql), where the rows of a run are checked by the queries of sql/quality/
-- before any of them is inserted here.

CREATE TABLE IF NOT EXISTS plants (
    plant_id     VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    process_type VARCHAR NOT NULL,
    CHECK (length(trim(plant_id)) > 0),
    CHECK (length(trim(name)) > 0),
    CHECK (length(trim(process_type)) > 0)
);

-- Known parameters only: what an engineer would know about the plant. Never the hidden
-- physics, and not the nominal values of a simplified model.
CREATE TABLE IF NOT EXISTS process_parameters (
    plant_id  VARCHAR NOT NULL REFERENCES plants (plant_id),
    parameter VARCHAR NOT NULL,
    value     DOUBLE  NOT NULL,
    unit      VARCHAR NOT NULL,
    PRIMARY KEY (plant_id, parameter),
    CHECK (isfinite(value)),
    CHECK (length(trim(parameter)) > 0),
    CHECK (length(trim(unit)) > 0)
);

-- Channels of a plant: measured variables and known inputs, told apart by channel_kind.
-- An input is known exactly, so it has no noise model: its noise fields are NULL, which
-- is not the same as a noise of zero.
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id         VARCHAR PRIMARY KEY,
    plant_id          VARCHAR NOT NULL REFERENCES plants (plant_id),
    variable_name     VARCHAR NOT NULL,
    channel_kind      VARCHAR NOT NULL,
    channel_index     INTEGER NOT NULL,
    unit              VARCHAR NOT NULL,
    sampling_period_s DOUBLE  NOT NULL,
    noise_model       VARCHAR,
    noise_std         DOUBLE,
    UNIQUE (plant_id, sensor_id),
    UNIQUE (plant_id, variable_name),
    UNIQUE (plant_id, channel_kind, channel_index),
    CHECK (channel_kind IN ('measured', 'input')),
    CHECK (channel_index >= 0),
    CHECK (isfinite(sampling_period_s) AND sampling_period_s > 0),
    CHECK (length(trim(variable_name)) > 0),
    CHECK (length(trim(unit)) > 0),
    CHECK (
        (channel_kind = 'measured' AND noise_model = 'additive_gaussian'
            AND noise_std IS NOT NULL AND isfinite(noise_std) AND noise_std >= 0)
        OR (channel_kind = 'input' AND noise_model IS NULL AND noise_std IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS operating_runs (
    run_id            VARCHAR PRIMARY KEY,
    plant_id          VARCHAR NOT NULL REFERENCES plants (plant_id),
    dataset_id        VARCHAR NOT NULL,
    operating_mode    VARCHAR NOT NULL,
    description       VARCHAR NOT NULL,
    start_time_s      DOUBLE  NOT NULL,
    end_time_s        DOUBLE  NOT NULL,
    sampling_period_s DOUBLE  NOT NULL,
    n_samples         BIGINT  NOT NULL,
    content_sha256    VARCHAR NOT NULL,
    UNIQUE (plant_id, run_id),
    CHECK (isfinite(start_time_s) AND isfinite(end_time_s) AND end_time_s >= start_time_s),
    CHECK (isfinite(sampling_period_s) AND sampling_period_s > 0),
    CHECK (n_samples >= 1),
    CHECK (length(content_sha256) = 64)
);

-- A measurement refers to its run and to its channel through the pairs
-- (plant_id, run_id) and (plant_id, sensor_id), so a row cannot join a run of one plant
-- to a sensor of another.
CREATE TABLE IF NOT EXISTS measurements (
    plant_id     VARCHAR  NOT NULL,
    run_id       VARCHAR  NOT NULL,
    sensor_id    VARCHAR  NOT NULL,
    sample_index BIGINT   NOT NULL,
    time_s       DOUBLE   NOT NULL,
    value        DOUBLE   NOT NULL,
    quality_flag SMALLINT NOT NULL,
    PRIMARY KEY (run_id, sensor_id, sample_index),
    FOREIGN KEY (plant_id, run_id) REFERENCES operating_runs (plant_id, run_id),
    FOREIGN KEY (plant_id, sensor_id) REFERENCES sensors (plant_id, sensor_id),
    CHECK (sample_index >= 0),
    CHECK (isfinite(time_s)),
    CHECK (isfinite(value)),
    CHECK (quality_flag = 0)
);
