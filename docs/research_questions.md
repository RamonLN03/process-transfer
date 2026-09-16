# Research questions

## Main question

Can the physical structure shared between related process systems be used to automatically determine what knowledge should be transferred, recalibrated or relearned when adapting a hybrid process model to a data-scarce target system?

## Secondary questions

* How much target-plant data can be saved through transfer?
* Does explicitly preserving conservation laws improve transfer performance?
* Does a hybrid physical and data-driven model transfer better than a purely black-box model?
* How does transfer performance degrade as the source-target domain shift grows?
* When does transfer become negative transfer?
* Can model components be classified automatically as transferable or plant-specific?
* Can sensitivity analysis and parameter identifiability determine what should be recalibrated?
* Can a learned transfer policy outperform a manually designed engineering heuristic?
* Can such a policy generalise to unseen process systems or unseen domain shifts?
* Can the framework estimate when transfer should not be attempted?
* Can uncertainty estimates detect unreliable adaptation?
* Can the framework eventually generalise across classes of process equipment?

## Baselines

Every method is compared against the simplest credible alternative. The required baselines (AGENTS.md) are:

1. Oracle model using the complete true physics (an upper bound, never a candidate).
2. Mechanistic model with target parameter re-estimation (what an engineer does today).
3. Target-only black-box model.
4. Target-only hybrid model.
5. Standard transfer model (pretrain on source, fine-tune on target).
6. Hybrid transfer model.
7. Physics-aware manual transfer heuristic.

Any later learned transfer policy must beat or meaningfully complement the manual heuristic. Reinforcement learning is not introduced until manual baselines work, and only if the transfer decision is genuinely sequential.

## Experimental design: adaptation budget

The target plant generates an amount of ground-truth and evaluation data comparable to the source plant. The restriction lives in the experimental layer, not in the simulator: only a budget of target data is made available for adaptation, initially about 2 h of operation. Later experiments compare budgets of 30 min, 1 h, 2 h, 4 h, 8 h and beyond, on the same target plant and, as far as possible, the same fixed evaluation set. Target data is measured in hours of operation or number of excitation tests, with the fraction of the full record reported alongside.

The central result should be a curve of target prediction error against target data available, for the competing approaches.

## Metrics

Prediction performance is not the only metric. Experiments should report:

* **Predictive accuracy**: RMSE, MAE, possibly R-squared.
* **Data efficiency**: performance as a function of target data available.
* **Physical consistency**: mass-balance residual, energy-balance residual, constraint violations.
* **Extrapolation**: performance outside the training operating region.
* **Transfer gain**: G = E_target_only - E_transfer; a negative gain is negative transfer.
* **Uncertainty**: prediction intervals and calibration, eventually.
* **Computational cost**: training and adaptation time, later.

## Negative transfer

Transfer is never assumed beneficial. One objective is to determine when the source is too different from the target for transfer to help. Possible indicators: distribution distance, maximum mean discrepancy, parameter differences, process-graph differences, model residual patterns, sensitivity changes. A useful system can return "transfer recommended" or "transfer not recommended" with supporting diagnostics.

## Sensitivity and identifiability

The sensitivity S_i = dy/dp_i indicates whether predictions depend meaningfully on a parameter. Parameters may be classified as high sensitivity and identifiable, low sensitivity and weakly identifiable, or unidentifiable from the available data. A possible policy: shared and unidentifiable, retain the source value or prior; target-specific and identifiable, recalibrate; unknown dynamics, use a data-driven component.

## Manual physics-aware transfer policy

Before any learned agent exists, an explicit expert heuristic is implemented:

* known invariant conservation law: preserve;
* known target parameter: replace with the target value;
* unknown but identifiable physical parameter: recalibrate;
* learned mechanism believed shared: initialise from source and fine-tune;
* large mechanism shift: relearn;
* strong source-target incompatibility: reject transfer.

Later automated methods must outperform this, so it has to be meaningful.

## Research discipline

Hypothesis, method, result and interpretation are kept separate. Conclusions are not written before results exist. A single random seed is not evidence; experiments use multiple seeds and report uncertainty. Negative results are recorded.
