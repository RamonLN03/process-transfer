-- Source against target: what was observed on two plants under the same experiment.
--
-- Parameters: $source and $target, two plant_id.
--
-- Pairing is explicit. A run of the source is paired with the run of the target that
-- has the same definition, which is the identity of the run without its plant
-- (p3.e0.x10.n0 of source.p3.e0.x10.n0). Rows are then paired tick by tick, and the
-- pairing is verified rather than assumed: n_same_instant counts the pairs whose
-- instants are equal, n_same_inputs those whose four inputs are equal, and coverage is
-- the share of the longer run that was paired. Statistics use only the pairs that are
-- complete and have equal instants and inputs.
--
-- What the numbers are. Differences between the readings of two plants, noise included.
-- They are observed differences between plants. They are not errors of a model, and no
-- hidden quantity enters: only measurements and known inputs are read.

WITH paired_runs AS (
    SELECT s.run_id AS source_run, t.run_id AS target_run,
           substr(s.run_id, length(s.plant_id) + 2) AS definition,
           s.n_samples AS n_source, t.n_samples AS n_target
    FROM operating_runs s
    JOIN operating_runs t
      ON substr(t.run_id, length(t.plant_id) + 2) = substr(s.run_id, length(s.plant_id) + 2)
    WHERE s.plant_id = $source AND t.plant_id = $target
),
paired_rows AS (
    SELECT p.definition, p.source_run, p.target_run, p.n_source, p.n_target,
           a.time_s = b.time_s AS same_instant,
           a."q" = b."q" AND a."C_Af" = b."C_Af" AND a."T_f" = b."T_f" AND a."T_c" = b."T_c" AS same_inputs,
           a.complete AND b.complete AS both_complete,
           b."C_A" - a."C_A" AS difference_c_a,
           b."T" - a."T"     AS difference_t
    FROM paired_runs p
    JOIN aligned_series a ON a.run_id = p.source_run
    JOIN aligned_series b ON b.run_id = p.target_run AND b.sample_index = a.sample_index
)
SELECT
    definition,
    source_run,
    target_run,
    count(*)                                                     AS n_paired,
    count(*) / greatest(max(n_source), max(n_target))::DOUBLE    AS coverage,
    count(*) FILTER (WHERE same_instant)                         AS n_same_instant,
    count(*) FILTER (WHERE same_inputs)                          AS n_same_inputs,
    count(*) FILTER (WHERE usable)                               AS n_used,
    avg(difference_c_a)         FILTER (WHERE usable)            AS observed_difference_c_a_mean,
    stddev_samp(difference_c_a) FILTER (WHERE usable)            AS observed_difference_c_a_std,
    min(difference_c_a)         FILTER (WHERE usable)            AS observed_difference_c_a_min,
    max(difference_c_a)         FILTER (WHERE usable)            AS observed_difference_c_a_max,
    avg(difference_t)           FILTER (WHERE usable)            AS observed_difference_t_mean,
    stddev_samp(difference_t)   FILTER (WHERE usable)            AS observed_difference_t_std,
    min(difference_t)           FILTER (WHERE usable)            AS observed_difference_t_min,
    max(difference_t)           FILTER (WHERE usable)            AS observed_difference_t_max
FROM (SELECT *, same_instant AND same_inputs AND both_complete AS usable FROM paired_rows)
GROUP BY definition, source_run, target_run
ORDER BY definition;
