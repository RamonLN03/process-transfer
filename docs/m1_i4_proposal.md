# I4 of M1: proposal

Written on 2026-10-08, after the acceptance of I2 (D-038) and the closure of I3 (D-037), for the owner and for review. Nothing of I4 is implemented and no data have been generated. M1-E02, the verification that I4 needs, is registered in the experiment log on the same day and has not been run. This document says what I4 changes and what it leaves alone, and lists what the owner decides before M1-E02 runs. It does not repeat the plan: sections 4.2, 4.3, 6.1, 6.2 and 13 of `docs/m1_plan.md` and D-030 stay the reference.

## Objective and scope

I4 adds the two extensions of P3 that the benchmark needs, and verifies them before any data use them (plan, section 13, row I4):

* the lead: 60 s at the nominal inputs at the start of every P3 run of M1, ticks 0 to 10, so that the first excursion has a context of ten readings (sections 4.2 and 4.3);
* the amplitude A5, the training region of the test of extrapolation (section 6.1; D-030, answer to Q2).

Out of scope: the physics, the plants, the operating point, the hold of 120 s, the rest of 600 s, any amplitude above A10, the definitions of the data sets of the benchmark (I5), and any data at A5, which only I6 and I7 generate (section 13; corrected on 2026-10-08, see item 4 of the next action).

## A5, exactly

The absolute amplitudes of A5 are half those of A10 (D-018, `simulation.protocols.a10_amplitudes`): q and C_Af move by +-5 % of their nominal values, T_f and T_c by +-2.5 K. The corners are the 16 corners of the input box at those amplitudes, drawn by P3 as at A10, each held 120 s and followed by 600 s at the nominal inputs. Only the vector of amplitudes changes.

## Identities and the data contract

Proposed grammar for the P3 runs of M1, with every token written out:

    <plant>.p3.e<excitation seed>.x<excursions>.l<lead in s>.a<amplitude>.n<noise>
    target.p3.e7.x40.l60.a5.n0

`a10` and `a5` name the amplitudes. An identity without `l` and `a`, as every P3 run of M0 has, keeps its present meaning: no lead, A10. The noise stream follows from the identity (D-021), so a new identity draws new noise. Set aside: leaving out a token that has its default value. A run of M1 at A10 would then differ from a run of M0 by one token only, and a reader would have to know the default.

The data contract gains the two tokens in its grammar (`docs/data_contract.md`, `data.identifiers`). Proposed: no new column, so no new version of the contract. The lead and the amplitude can be read from the identity and from the stored inputs, and a column would hold the same information twice. D-026 added the identities of the step tests on the same ground. The alternative is a column of `operating_runs` and of the export, which makes a new version of the contract.

The definitions of data sets under `configs/datasets/` gain fields for the lead and the amplitude. A definition without them, as the three of M0 are, reads as before.

## Compatibility with M0

The data sets, databases and exports of M0 (`m0-e05`, `m0-e06`, `m0-e07`), their identities, their content hashes and the results registered on them stay as they are. None is regenerated or rewritten; new runs make new data sets (`docs/data_contract.md`, policy on repetition). The tolerances and verdicts of M0-E01 to M0-E08 do not change. Tests of I4 show that every identity of M0 builds and parses as before, and that the definitions of M0 still give the content hashes recorded for them.

## The verification, M1-E02

Registered in `docs/experiment_log.md`, entry M1-E02, with its hypothesis, method, criteria, the provenance of each tolerance, the evidence the run keeps and what follows a failure. It is not repeated here. The windows of a run with a lead are found from the inputs alone by `evaluation.windows`, already tested on synthetic runs with a lead (`tests/m1_support.py`); the tests of I4 add a run built by the new protocol.

## What the owner decides before M1-E02 runs

Decided by the owner on 2026-10-08 (D-039): the four points below as proposed, with the contract kept at version 1 only where its compatibility is shown by checks.

1. The grammar of the two tokens, and that identities without them keep the meaning they have in M0, as proposed.
2. The contract: no column and no new version, as proposed, or a column and a new version.
3. M1-E02 on the target alone, as registered, since M1 generates new data on the target only (plan, section 5.2), or on both plants, as M0-E03b did.
4. M1-E02 checks the simulation and generates no data, as registered, since section 6.2 asks for the verification before any A5 data exist. The path of the new identities through the generator is shown by the tests of I4, not by data at A5.

Until the run each of these can change. A change is a dated amendment of the registration, made before the run.

## Next action

Once the owner has decided the points above:

1. Implement the lead and A5 in `simulation.protocols`, and the two tokens in `data.identifiers` and the contract, with tests, including those of compatibility with M0.
2. Write `experiments/m1_e02_p3_a5.py` as the registration describes, with tests, and commit it from a clean tree.
3. Run M1-E02 twice from that commit and record the result and its reading in the experiment log.
4. Passing M1-E02 does not authorise any data. I5 registers the benchmark and the definitions of its data sets and is reviewed before I6; I6 generates the development sets, those at A5 included, and trains; I7 generates the test sets after the technical freeze (plan, section 13).

Corrected on 2026-10-08, after Codex's review of `a9134bc`. Item 4 first read "Only if it passes, A5 data may be generated, in I5", which put the generation of data in the wrong iteration (experiment log, amendment of the registration of M1-E02, point A).
