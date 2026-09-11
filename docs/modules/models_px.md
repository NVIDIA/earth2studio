<!-- markdownlint-disable MD013 -->

# Prognostics

`earth2studio.models.px`

Prognostic models are a class of models that perform time-integration.
Thus are typically used to generate forecast predictions.

!!! warning
    Pre-trained prognostic models provided in Earth2Studio may be provided
    under different licenses. We encourage users to familiarize themselves with each
    prior to use.

<!-- e2s-autosummary
currentmodule: earth2studio.models.px
template: prognostic
output: generated/models/px
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa class:nowcasting class:downscaling class:medium-range class:subseasonal-seasonal class:data-assimilation class:climate provider:nvidia provider:ecmwf provider:ai2 provider:google provider:microsoft backend:pytorch backend:jax backend:onnx product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu year:2021 year:2022 year:2023 year:2024 year:2025 year:2026 gpu:96gb gpu:80gb gpu:48gb gpu:40gb mode=or order=fixed toggle=true labels=label hidden="class provider backend product year" -->
{% autosummary %}
earth2studio.models.px.ACE2ERA5
earth2studio.models.px.AIFS
earth2studio.models.px.AIFS2
earth2studio.models.px.AIFS2ENS
earth2studio.models.px.AIFSENS
earth2studio.models.px.Atlas
earth2studio.models.px.AtlasCRPS
earth2studio.models.px.Aurora
earth2studio.models.px.Aurora1p5
earth2studio.models.px.Aurora1p5Ensemble
earth2studio.models.px.CBottleVideo
earth2studio.models.px.DataReplay
earth2studio.models.px.DiagnosticWrapper
earth2studio.models.px.DLESyM
earth2studio.models.px.DLESyMLatLon
earth2studio.models.px.DLESyMv0_ISCCP_ERA5
earth2studio.models.px.DLESyMv0_ISCCP_ERA5LatLon
earth2studio.models.px.DLWP
earth2studio.models.px.FCN
earth2studio.models.px.FCN3
earth2studio.models.px.FengWu
earth2studio.models.px.FuXi
earth2studio.models.px.GenCastMini
earth2studio.models.px.GraphCastOperational
earth2studio.models.px.GraphCastSmall
earth2studio.models.px.InterpModAFNO
earth2studio.models.px.Pangu24
earth2studio.models.px.Pangu6
earth2studio.models.px.Pangu3
earth2studio.models.px.Persistence
earth2studio.models.px.SamudrACE
earth2studio.models.px.SFNO
earth2studio.models.px.StormCast
earth2studio.models.px.StormCastCONUS
earth2studio.models.px.StormScopeGOES
earth2studio.models.px.StormScopeMeteosatEU
earth2studio.models.px.StormScopeMRMS
earth2studio.models.px.UCast
earth2studio.models.px.WeatherNext2Cyclones
earth2studio.models.px.WeatherNext2CyclonesMini
{% endautosummary %}

<!-- mkdocs-badges:end -->
