"""Models of M1 fitted on data of the target plant (``docs/m1_plan.md``, section 8).

What it holds in I1: the interface of a continuous-time model and its rollout under
piecewise-constant inputs, with failures returned as records (``rollout``); the modeller's
equations with the textbook values, MN, and with estimated values (``mechanistic``); the
estimation of MR on all the windows of a budget and of MR_F on its fitting part, with the
record of every start and the covariance of the estimate (``fitting``).

In I3: BL, the linear black box, fitted like MR (``linear``); the equations of the neural
black box BN and of the hybrids HK, HU and HKU, written once for numpy and JAX
(``learned``); their training with JAX and selection on V (``training``); and the selected
model of a training written to a file and read back (``persistence``).

This package is on the available side, like ``evaluation``. A model is built from the known
parameters of an export and the modeller's values of ``configs/modeller_cstr.yaml``, and
fitted on the ``WindowData`` it is handed. It never imports ``process_transfer.simulation``
or ``process_transfer.generation`` and never reads a plant configuration file, which a test
on the import graph enforces.
"""
