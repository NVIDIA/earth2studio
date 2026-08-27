<!-- markdownlint-disable MD013 -->

# Diagnostics

`earth2studio.models.dx`

Diagnostic models are a class of models that do not perform time-integration.
These may be used to map between weather/climate variables to other quantities of
interest, used to enable additional analysis, improve prediction accuracy, downscale,
etc.

!!! warning
    Pre-trained diagnostic models provided in Earth2Studio may be provided
    under different licenses. We encourage users to familiarize themselves with each
    prior to use.

<!-- e2s-autosummary
currentmodule: earth2studio.models.dx
template: diagnostic
output: generated/models/dx
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa class:nowcasting class:downscaling class:medium-range class:subseasonal-seasonal class:data-assimilation class:climate provider:nvidia provider:ecmwf provider:ai2 provider:google provider:microsoft backend:pytorch backend:jax backend:onnx product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu year:2021 year:2022 year:2023 year:2024 year:2025 year:2026 gpu:96gb gpu:80gb gpu:48gb gpu:40gb mode=or order=fixed toggle=true labels=label hidden="class provider backend product year" -->
{% autosummary %}
earth2studio.models.dx.CBottleInfill
earth2studio.models.dx.CBottleSR
earth2studio.models.dx.CBottleTCGuidance
earth2studio.models.dx.CorrDiffCMIP6
earth2studio.models.dx.CorrDiffCosmoEra5
earth2studio.models.dx.CorrDiffTaiwan
earth2studio.models.dx.ClimateNet
earth2studio.models.dx.DerivedRH
earth2studio.models.dx.DerivedRHDewpoint
earth2studio.models.dx.DerivedSurfacePressure
earth2studio.models.dx.DerivedTCWV
earth2studio.models.dx.DerivedVPD
earth2studio.models.dx.DerivedWS
earth2studio.models.dx.DLESyMv0_ISCCP_ERA5Precip
earth2studio.models.dx.OrbitGlobalPrecip
earth2studio.models.dx.PrecipitationAFNO
earth2studio.models.dx.PrecipitationAFNOv2
earth2studio.models.dx.SolarRadiationAFNO1H
earth2studio.models.dx.SolarRadiationAFNO6H
earth2studio.models.dx.StormScopeDxNSRDB
earth2studio.models.dx.TCTrackerWuDuan
earth2studio.models.dx.TCTrackerVitart
earth2studio.models.dx.WindgustAFNO
earth2studio.models.dx.Identity
{% endautosummary %}

<!-- mkdocs-badges:end -->
