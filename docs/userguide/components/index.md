# Core Components

Earth2Studio workflows compose small, interchangeable components. Fetch physical
fields, run a model, and pass its labelled outputs to another model, a statistic, or
an output backend.

| Component | Role |
| --- | --- |
| [Data and metadata](coordinates.md) | Describe fields, grids, times, and tabular schemas. |
| [Prognostic models](prognostic.md) | Advance a forecast through time. |
| [Diagnostic models](diagnostic.md) | Derive or transform fields in a single call. |
| [Data assimilation models](data_assimilation.md) | Combine observations and background states. |
| [Data sources](datasources.md) | Fetch gridded fields or tabular observations. |
| [Perturbations](perturbation.md) | Generate perturbed fields for ensembles. |
| [IO backends](io.md) | Store labelled outputs in memory or on disk. |
| [Statistics](statistics.md) | Reduce fields and compare predictions with observations. |

Start with [Data and Metadata](coordinates.md) for the data exchanged between
components. See the [Overview](../about/overview.md) for composing them into pipelines.

Each component page introduces its interface, shows usage, and explains how to
implement your own component. Full requirements live in the
[specifications](../../specs/README.md). You can implement
these interfaces in your own project without inheriting a protocol class. For an
agent-assisted custom model, start with
[Wrap Your Own Prognostic Model](prognostic.md#wrap-your-own-prognostic-model).
