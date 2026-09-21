-- Staging: the same five tables without any constraint.
--
-- The rows of a run are loaded here first, as they come from the Parquet files. The
-- queries of sql/quality/ are run against this schema, and only if every one of them
-- returns no row are the rows inserted into the main schema, all in one transaction.
-- A defective file can therefore be loaded and described, which the constraints of the
-- main schema would only refuse with the first error they meet.
--
-- The quality queries name their tables without a schema. They are run with
-- "USE staging" before ingestion and can be run with "USE main" at any time.

CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.plants (
    plant_id     VARCHAR,
    name         VARCHAR,
    process_type VARCHAR
);

CREATE TABLE IF NOT EXISTS staging.process_parameters (
    plant_id  VARCHAR,
    parameter VARCHAR,
    value     DOUBLE,
    unit      VARCHAR
);

CREATE TABLE IF NOT EXISTS staging.sensors (
    sensor_id         VARCHAR,
    plant_id          VARCHAR,
    variable_name     VARCHAR,
    channel_kind      VARCHAR,
    channel_index     INTEGER,
    unit              VARCHAR,
    sampling_period_s DOUBLE,
    noise_model       VARCHAR,
    noise_std         DOUBLE
);

CREATE TABLE IF NOT EXISTS staging.operating_runs (
    run_id            VARCHAR,
    plant_id          VARCHAR,
    dataset_id        VARCHAR,
    operating_mode    VARCHAR,
    description       VARCHAR,
    start_time_s      DOUBLE,
    end_time_s        DOUBLE,
    sampling_period_s DOUBLE,
    n_samples         BIGINT,
    content_sha256    VARCHAR
);

CREATE TABLE IF NOT EXISTS staging.measurements (
    plant_id     VARCHAR,
    run_id       VARCHAR,
    sensor_id    VARCHAR,
    sample_index BIGINT,
    time_s       DOUBLE,
    value        DOUBLE,
    quality_flag SMALLINT
);
