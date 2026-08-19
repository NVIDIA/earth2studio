<!-- markdownlint-disable MD013 -->

# Data Assimilation

`earth2studio.models.da`

Data assimilation models are a class of models that integrate observational data into
model states or grids. These models can ingest both sparse observations (via
DataFrames) and dense fields (via DataArrays) to produce output suitable for
downstream tasks such as driving a prognostic model or generating a guided forecast.
Data assimilation models support both stateless and stateful operation, allowing them
to process observations independently or maintain internal state across time steps.

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
earth2studio.models.da.HealDA
earth2studio.models.da.InterpEquirectangular
earth2studio.models.da.StormCastSDA
{% endautosummary %}

<!-- mkdocs-badges:end -->
