# Project scope

ProcessTransfer is a long-term research and software engineering project on transferring process models from a data-rich source plant to a related target plant with much less available data.

Related industrial processes often share a large amount of physical knowledge even when their scale, geometry, operating conditions or equipment parameters differ. Conservation laws, stoichiometry and parts of the underlying mechanisms may be unchanged, while heat-transfer coefficients, mixing behaviour, equipment geometry, catalyst activity or sensor characteristics may need to be recalibrated or relearned. The project investigates whether this physical structure can be exploited to make transfer between process systems more data-efficient, reliable and interpretable.

## Objective

The objective is more than a single transfer-learning model. The aim is a reusable framework that determines what knowledge should be preserved, directly transferred, fine-tuned, recalibrated, estimated from target data, or discarded and relearned.

Eventually the framework should become an engineering product: an engineer provides source-plant information, target-plant information and the available data, and receives a physically consistent model adapted to the target plant.

The envisioned system has three layers:

1. **Physics-aware hybrid modelling.** The core model combines mechanistic knowledge with machine learning while enforcing physical constraints.
2. **Automatic transfer engine.** Initially rule based, later possibly learned, this layer decides how each piece of knowledge is transferred between processes.
3. **Engineering interface.** Eventually an LLM-assisted interface interprets documentation, process descriptions and data mappings into a structured representation that the deterministic framework validates and uses.

The project is at once a research project, a software engineering project and a potential product. The three objectives stay distinct but compatible.

## Three kinds of knowledge

The modelling philosophy rests on distinguishing three types of knowledge.

**Invariant or shared physical knowledge**: conservation of mass and energy, stoichiometry, thermodynamic identities, known reaction networks. Normally preserved, not relearned.

**Plant-specific physical knowledge**: reactor volume, equipment dimensions, heat-transfer area and coefficients, residence time, mixing, pressure drops, equipment efficiencies. May need replacement or recalibration on the target.

**Unknown or incomplete mechanisms**: rate corrections, catalyst degradation, fouling, complex mixing, effective mass transfer, unresolved nonlinearities, sensor biases. Candidates for data-driven modelling.

The conceptual hybrid model is

    dx/dt = f_known(x, u, p) + f_unknown(x, u; theta)

where the first term is mechanistic and the second may be represented by machine learning. Machine learning is not used to rediscover physics that is already known exactly.

## Physical consistency

Physical validity is a first-class requirement. A black-box model can reach low prediction error while producing impossible trajectories. The framework should assess at least the mass-conservation residual and the energy-conservation residual, and potentially positivity, elemental conservation, stoichiometric consistency, bounded variables and thermodynamic feasibility.

Where possible, consistency is enforced structurally rather than as a loss penalty. Instead of asking a network to predict all reactor concentrations, the network estimates an unknown reaction rate r(C_A, T) that is inserted into known balances. Physically coupled quantities are not left for a network to determine independently.

## Product vision (long term)

A process engineer should be able to provide, for the source and the target process: a process description, the equipment configuration, material and energy balances, known parameters and historical data (abundant for the source, limited for the target). The software should then validate the data, align equivalent variables, identify shared structure and differing parameters, identify uncertain or missing mechanisms, build or load a source hybrid model, choose a transfer strategy, adapt the model, validate prediction performance and physical consistency, quantify uncertainty, detect possible negative transfer, and explain what was transferred and why.

A future interface might report, for each model component, whether it was preserved (mass conservation, stoichiometry, energy-balance structure), transferred (reaction mechanism, learned kinetic correction), re-estimated (heat-transfer coefficient), replaced (reactor volume) or learned on the target (mixing correction, sensor bias), together with the target data used, prediction error, balance residuals and a transfer confidence.

The aim is not simply to automate model fitting. The system should expose enough information that an engineer can understand and challenge the model.

## Novelty strategy

Transfer learning, hybrid modelling, physics-informed machine learning, reinforcement learning, LLM agents and automatic process modelling all exist individually. No novelty is claimed for any of them. The research opportunity lies in their integration around automatic, physically structured model transfer.

A possible eventual contribution: a framework that decomposes process knowledge into invariant physics, plant-specific physics and learned mechanisms, then determines what should be preserved, recalibrated or relearned when adapting a hybrid model to a data-scarce related process. A stronger contribution would be a learned transfer policy over structured process representations that generalises across process systems.

Novelty is re-evaluated against the literature before any academic claim is made.

## Commercialisation and intellectual property

Because the long-term objective may include a product, the repository is not assumed to become public immediately. Initially: a private repository, no permissive open-source licence by default, clear authorship and development history, and research notes that document original decisions. Before a paper, a public repository or a substantial technical description, reconsider whether protectable intellectual property exists. This does not prevent scientific work; publication and commercialisation decisions are deliberate.

## Success criterion

The final success state is not "a neural network predicts a CSTR", nor "transfer learning improves RMSE on one dataset". It is a general engineering framework that takes a data-rich source process and a related data-scarce target process, understands which physical and learned knowledge can be reused, constructs an adaptation strategy, produces a physically consistent target model, quantifies uncertainty and explains the transfer decisions to the engineer. If the research succeeds, that framework should become accessible through an interface suitable for process engineers rather than machine-learning specialists.
