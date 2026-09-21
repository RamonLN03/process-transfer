-- Aggregation in time: the six-second series averaged over blocks of $block_ticks ticks.
--
-- Parameter: $block_ticks, a whole number of ticks, 1 or more (10 ticks are 60 s).
--
-- Edges. Block k holds the ticks k * B to (k + 1) * B - 1, that is the instants from
-- block_start_s, included, to block_end_s, excluded. Blocks are counted from the first
-- tick of the run, so a block never spans two runs.
-- Label. A block is labelled by block_start_s, computed from the clock of the run and
-- not from the rows found, so it is right even if the first tick of the block is absent.
-- Coverage. n_expected is B, or fewer in the last block of a run; coverage is the share
-- of those ticks that are present and complete. A partial block is reported, not dropped.
-- Readings are averaged over the ticks present. Nothing is interpolated or filled in.
-- Inputs are not averaged across a change. Inside a block an input is either constant,
-- and its value is given, or it changes, and its column is NULL; inputs_constant says
-- which. A reading averaged over a block in which the inputs step mixes two regimes, so
-- such a block is flagged and left to the reader to exclude.
--
-- This is a derived summary. The original series of six seconds, the view
-- aligned_series, stays the reference for a dynamic model.

WITH blocks AS (
    SELECT a.*, a.sample_index // $block_ticks AS block_index
    FROM aligned_series a
    WHERE $block_ticks >= 1
)
SELECT
    b.plant_id,
    b.run_id,
    b.block_index,
    r.start_time_s + b.block_index * $block_ticks * r.sampling_period_s               AS block_start_s,
    r.start_time_s + (b.block_index + 1) * $block_ticks * r.sampling_period_s         AS block_end_s,
    least($block_ticks, r.n_samples - b.block_index * $block_ticks)                   AS n_expected,
    count(*) FILTER (WHERE b.complete)                                                AS n_present,
    count(*) FILTER (WHERE b.complete)
        / least($block_ticks, r.n_samples - b.block_index * $block_ticks)::DOUBLE     AS coverage,
    avg(b."C_A") FILTER (WHERE b.complete)                                            AS "C_A_mean",
    avg(b."T")   FILTER (WHERE b.complete)                                            AS "T_mean",
    CASE WHEN min(b."q")    = max(b."q")    THEN min(b."q")    END                    AS "q",
    CASE WHEN min(b."C_Af") = max(b."C_Af") THEN min(b."C_Af") END                    AS "C_Af",
    CASE WHEN min(b."T_f")  = max(b."T_f")  THEN min(b."T_f")  END                    AS "T_f",
    CASE WHEN min(b."T_c")  = max(b."T_c")  THEN min(b."T_c")  END                    AS "T_c",
    min(b."q") = max(b."q") AND min(b."C_Af") = max(b."C_Af")
        AND min(b."T_f") = max(b."T_f") AND min(b."T_c") = max(b."T_c")               AS inputs_constant
FROM blocks b
JOIN operating_runs r ON r.run_id = b.run_id
GROUP BY b.plant_id, b.run_id, b.block_index, r.start_time_s, r.sampling_period_s, r.n_samples
ORDER BY b.plant_id, b.run_id, b.block_index;
