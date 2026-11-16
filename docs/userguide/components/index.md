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

## Shared Conventions

- **Fields carry their coordinates.** Gridded components exchange `xarray.DataArray`
  objects with named dimensions, coordinate labels, and grid metadata. Values use
  physical units; models handle their own normalization. See
  [Data Movement](../about/overview.md#data_userguide).
- **Signatures describe fields without storing values.** A `CoordinateSystem` is an
  allocation-free DataArray built with `coord_array`. Inspect `.dims`, `.sizes`, and
  `.coords`; do not read its field `.values`. `output_coords` supports planning without
  running inference.
- **Dynamic dimensions are explicit.** A model signature may declare a zero-sized
  leading prefix, often `batch`, that accepts concrete leading axes such as `time`
  and `ensemble`. The remaining dimensions, order, labels, and metadata must match.
  This is not permission to reorder or broadcast arbitrary axes.
- **Metadata is part of the interface.** Grid identity, coordinate reference system,
  and temporal statistics travel with fields. A label such as `tp:sum:6h` denotes
  a six-hour accumulation, not an instantaneous precipitation field.
- **Composition is explicit.** Callers fetch and align inputs. Components do not
  silently reorder variables, regrid fields, or convert physical units to make a
  mismatched model input fit. Utilities such as `fetch_data` perform requested
  preparation before execution.

Each component page covers its interface, core rules, and usage. You can implement
these interfaces in your own project without inheriting a protocol class. For an
agent-assisted custom model, start with
[Wrap Your Own Prognostic Model](prognostic.md#wrap-your-own-prognostic-model).
