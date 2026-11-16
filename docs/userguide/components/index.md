# Core Components

Earth2Studio workflows compose small, interchangeable components. Fetch physical
fields, run a model, and pass its labelled outputs to another model, a statistic, or
an output backend.

| Component | Role |
| --- | --- |
| [Prognostic models](prognostic.md) | Advance a forecast through time. |
| [Diagnostic models](diagnostic.md) | Derive or transform fields in a single call. |
| [Data assimilation models](data_assimilation.md) | Combine observations and background states. |
| [Data sources](datasources.md) | Fetch gridded fields or tabular observations. |
| [Perturbations](perturbation.md) | Generate perturbed fields for ensembles. |
| [IO backends](io.md) | Store labelled outputs in memory or on disk. |
| [Statistics](statistics.md) | Reduce fields and compare predictions with observations. |

See the [Overview](../about/overview.md) for data exchange, coordinate signatures,
and composing components into workflows.

Each component page covers its interface, specification, and usage. You can implement
these interfaces in your own project without inheriting a protocol class. For an
agent-assisted custom model, start with
[Wrap Your Own Prognostic Model](prognostic.md#wrap-your-own-prognostic-model).

## Domain Specification

The [model contract][domain-spec] defines these additional compliance codes only
for models implementing `DomainAdaptable`.

| Code | Requirement |
| --- | --- |
| `G1` | `with_region(region)` returns a view covering the region, or raises `ValueError`. |
| `G2` | Views leave parent and sibling signatures, parameters, and results unchanged. |
| `G3` | A view reaches the full supported footprint with the original model's validation. |
| `G4` | Views reject domain-dependent prognostic state from an incompatible domain. |

Coverage in `G1` is geometric, including every output slot and a rectangle's center
and orientation; it does not guarantee prediction skill or absence of edge artifacts.
`G2` applies to sequential use with equivalent seeding.

[domain-spec]: https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/dev/spec/MODEL_CONTRACT_SPEC.md#domain
