"""The evaluation contract of M1 (``docs/m1_plan.md``, sections 4.3, 5.4 to 5.8 and 9).

What it holds: the known specification of a plant as an export states it (``plant``);
windows, phases and contexts found from the known inputs (``windows``); budgets and their
division into a fitting part F and a validation part V (``budgets``); metrics against
readings (``metrics``); records of failures and the rules of paired comparison
(``outcomes``); the validity bounds of a prediction and the rate and heat flow that a
right-hand side implies (``physics``).

This package is on the available side. It reads ``Observations`` and the known parameters
of an export and never imports ``process_transfer.simulation`` or
``process_transfer.generation``, which a test on the import graph enforces. It holds no
oracle diagnostic: an error against exact states needs the truth, and belongs to a script
on the truth side whose outputs go under ``experiments/``.
"""
