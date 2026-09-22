# CLAUDE.md

Read `AGENTS.md` before making any repository changes.

The user communicates with Claude Code in Spanish, but all repository content must be written in English.

Before each working session:

1. inspect `git status`;
2. inspect recent `git log --oneline`;
3. read `AGENTS.md`;
4. inspect the relevant existing code and documentation.

Do not implement future milestones unless explicitly requested.

The current milestone is M1: Black-Box and Hybrid Modelling, on target-plant data only. Its plan is `docs/m1_plan.md`.

M0, Virtual Plant and Data Infrastructure, was closed on 2026-09-22 (audited technical reference `91206b2`, tag `m0-v1.0`).

For scientifically consequential choices:

* explain the alternatives;
* state the proposed choice;
* explain why;
* do not silently implement the decision when user input would materially change the research design.

Prefer small, testable changes and small commits.

Do not infer successful results before experiments have been run.

When reporting research work, separate:

* hypothesis;
* method;
* result;
* interpretation.

The repository state and tests take precedence over assumptions from previous conversations.
