<!-- markdownlint-disable MD013 -->

# Data Assimilation

`earth2studio.models.da`

Data assimilation models are a class of models that integrate observational data into
model states or grids. These models can ingest both sparse observations (via
DataFrames) and dense fields (via DataArrays) to produce output suitable for
downstream tasks such as driving a prognostic model or generating a guided forecast.
`AssimilationModel` describes independent calls and input, forcing, output, source,
and device declarations. `StatefulAssimilationModel` extends that contract with
`initialize`, `step`, and `create_iterator`, carrying continuation state explicitly.
Initialization and the iterator's first yield produce the first computed analysis;
they do not return the initial background.

All DA models declare `stochastic` and `set_rng(seed, reset=True)`. Deterministic
models use a no-op; stochastic calls advance a local stream. `reset=False` preserves
an existing stream. Reseeding does not change an active trajectory's explicit state.

HealDA, interpolation, and CorrDiff use independent calls. CorrDiff's background
input does not make it stateful. StormCastSDA and StormCastCONUSSDA support stateful continuation,
with required gridded conditioning and optional observations at each step. Inputs
and outputs may be DataArrays or DataFrames as declared by each model. See the
[DA user guide](../userguide/components/data_assimilation.md) for calling patterns.

StormCastCONUSSDA wraps the CONUS checkpoint and supports cropped domains. Its
conditioning and observations both target the output valid time (+1 hour); use the
`stormcast-conus` installation extra and the StormCast-CONUS SDA gallery example.

!!! warning
    Data Assimilation models are a new addition to Earth2Studio and APIs might be subject
    to change without warning while the implementation is hardened.

<!-- e2s-autosummary
currentmodule: earth2studio.models.da
template: dataassim
output: generated/models/da
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa class:nowcasting class:downscaling class:medium-range class:subseasonal-seasonal class:data-assimilation class:climate provider:nvidia provider:ecmwf provider:ai2 provider:google provider:microsoft backend:pytorch backend:jax backend:onnx product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu year:2021 year:2022 year:2023 year:2024 year:2025 year:2026 gpu:96gb gpu:80gb gpu:48gb gpu:40gb mode=or order=fixed toggle=true labels=label hidden="class provider backend product year" -->
{% autosummary %}
earth2studio.models.da.CorrDiffCosmoEra5SDA
earth2studio.models.da.HealDA
earth2studio.models.da.InterpEquirectangular
earth2studio.models.da.StormCastCONUSSDA
earth2studio.models.da.StormCastSDA
{% endautosummary %}

<!-- mkdocs-badges:end -->
