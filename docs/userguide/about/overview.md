<!-- markdownlint-disable MD033 -->

# Overview { #overview_userguide }

Earth2Studio is a Python package of building blocks for AI weather and climate
inference. In 1.0, shared interfaces and labeled data connect data sources, models,
and analysis tools so you can construct pipelines for your own application.

## Package Philosophy

Each component has a focused role: data sources fetch fields or observations;
models forecast, diagnose, downscale, or assimilate; regridders align spatial
grids; perturbations and statistics support ensembles; IO backends store results.
Use them directly, combine them with the execution tools, or supply your own
implementations of the same interfaces.

These interfaces form **contracts**: methods define what a component does, while
coordinate signatures and schemas declare the data it accepts and produces.
This makes pipelines easier for both humans and coding agents to build and inspect.
A person can trace labeled fields through a workflow; an agent can inspect the same
declarations to select inputs, connect compatible components, and diagnose mismatches
without having to infer every model's requirements from its implementation.

## Data Movement { #data_userguide }

Two core data structures carry data between components:

- **DataArrays** (`xarray.DataArray`) hold multidimensional fields, with values,
  named dimensions, coordinates, and metadata together. NumPy backs CPU values;
  CuPy backs CUDA values. Models, perturbations, statistics, and IO backends work
  with these labeled fields directly.
- **DataFrames** (pandas or cuDF, where supported) hold tabular data such as sparse
  observations. A `FrameSchema` describes the expected columns and constraints.
  Assimilation models can accept DataArrays, DataFrames, or a combination.

`fetch_data` prepares fields for model inputs, including device placement,
regridding, and requested temporal statistics. `fetch_dataframe` prepares tabular
inputs. Torch-based models convert at their computation boundary with
`.e2s.to_torch()` and `from_torch()`, keeping the public exchange labeled.

!!! note
    Data is moved between Earth2Studio components in physical units. Normalization,
    scaling, and model-specific preprocessing should be handled inside the relevant
    model or component.

## Coordinate Systems { #coordinates_userguide }

Coordinates describe what a model needs and what it produces, beyond array shape:
which variables, which grid, and which times. Gridded declarations use allocation-free
`CoordinateSystem` DataArray signatures: inspect `.dims`, `.sizes`, and `.coords`
without allocating field values. Tabular declarations use `FrameSchema` instead.

| Declaration | Contract |
| --- | --- |
| `input_coords()` | Inputs required for a call or rollout initialization. |
| `forcing_coords()` | Stateful conditioning, such as boundary fields or observations. |
| `output_coords(*input_coords)` | Outputs produced for the supplied input signatures. |

A model with several inputs or outputs declares an ordered tuple of signatures,
one per argument or result. Common coordinates include `variable`, `time`,
`lead_time`, `lat`, and `lon`; dynamic leading dimensions allow batching.
For example, an input signature can specify two historical atmospheric states on
one grid, while the output signature declares a forecast six hours later.
These declarations let execution tools check compatibility and plan data requests
and output storage before inference.

## Constructing Pipelines

A typical pipeline fetches initial conditions and observations, aligns them to model
contracts, executes models, and stores or scores their outputs. Two layers assemble
these building blocks:

- **`Pipeline`** (`earth2studio.run`) drives a runner over work items specifying
  initialization times, forecast horizons, and optional ensemble members. It handles
  output filtering and regridding, storage, optional scoring, distribution across
  ranks, and resuming completed work. `Pipeline.from_model(...)` builds a prognostic
  runner from a model and data sources, with optional diagnostics.
- **`CoupledRunner`** (`earth2studio.coupling`) connects multiple models through
  `Connector` declarations. Each connector feeds a producer's fields into a consumer's
  slot, with explicit variable mapping, time selection, and optional regridding.
  The coupler uses model signatures to schedule components at their own cadences
  and fetch inputs not supplied by another component.

Pass a `CoupledRunner` to `Pipeline` to run a coupled system with the same output,
scoring, and work-management tools as a single-model forecast. The contracts remain
visible at each boundary, so humans and agents can replace a component or extend a
pipeline while checking that its inputs and outputs still fit.

## Where To Go Next

<div class="grid cards" markdown>

- **Install Earth2Studio**

    Set up the package, optional dependencies, cache locations, and model-specific
    extras.

    [:octicons-arrow-right-24: Install guide](install.md)

- **Learn the components**

    Read about prognostic models, diagnostic models, data sources, perturbations, IO,
    and statistics.

    [:octicons-arrow-right-24: Core components](../components/index.md)

- **Run examples**

    Jump directly into executable examples for forecasts, diagnostics, downscaling,
    data assimilation, and extension patterns.

    [:octicons-arrow-right-24: Examples](../../examples/index.md)

- **Contribute or extend**

    Find development setup, testing, style, documentation, and package-extension
    guidance.

    [:octicons-arrow-right-24: Developer guide](../developer/index.md)

</div>
