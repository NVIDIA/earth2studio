<!-- markdownlint-disable MD013 -->

# [`earth2studio.models.dx`][earth2studio.models.dx]: Diagnostics

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
badges:
- region:global
- region:na
- region:eu
- region:as
- region:au
- region:af
- region:sa
- class:nwc
- class:ds
- class:mrf
- class:s2s
- class:da
- class:cm
- product:wind
- product:precip
- product:temp
- product:atmos
- product:ocean
- product:land
- product:veg
- product:solar
- product:radar
- product:sat
- product:insitu
- year:2021
- year:2022
- year:2023
- year:2024
- year:2025
- year:2026
- gpu:96gb
- gpu:80gb
- gpu:48gb
- gpu:40gb
filter:
  mode: or
  order: fixed
  toggle: 'true'
  hidden: product year
objects:
- CBottleInfill
- CBottleSR
- CBottleTCGuidance
- CorrDiffCMIP6
- CorrDiffCosmoEra5
- CorrDiffTaiwan
- ClimateNet
- DerivedRH
- DerivedRHDewpoint
- DerivedSurfacePressure
- DerivedTCWV
- DerivedVPD
- DerivedWS
- DLESyMv0_ISCCP_ERA5Precip
- OrbitGlobalPrecip
- PrecipitationAFNO
- PrecipitationAFNOv2
- SolarRadiationAFNO1H
- SolarRadiationAFNO6H
- StormScopeDxNSRDB
- TCTrackerWuDuan
- TCTrackerVitart
- WindgustAFNO
- Identity
-->

<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa class:nwc class:ds class:mrf class:s2s class:da class:cm product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu year:2021 year:2022 year:2023 year:2024 year:2025 year:2026 gpu:96gb gpu:80gb gpu:48gb gpu:40gb mode=or order=fixed toggle=true hidden="product year" -->

{% autosummary %}
modules/generated/models/dx/CBottleInfill.md
modules/generated/models/dx/CBottleSR.md
modules/generated/models/dx/CBottleTCGuidance.md
modules/generated/models/dx/CorrDiffCMIP6.md
modules/generated/models/dx/CorrDiffCosmoEra5.md
modules/generated/models/dx/CorrDiffTaiwan.md
modules/generated/models/dx/ClimateNet.md
modules/generated/models/dx/DerivedRH.md
modules/generated/models/dx/DerivedRHDewpoint.md
modules/generated/models/dx/DerivedSurfacePressure.md
modules/generated/models/dx/DerivedTCWV.md
modules/generated/models/dx/DerivedVPD.md
modules/generated/models/dx/DerivedWS.md
modules/generated/models/dx/DLESyMv0_ISCCP_ERA5Precip.md
modules/generated/models/dx/OrbitGlobalPrecip.md
modules/generated/models/dx/PrecipitationAFNO.md
modules/generated/models/dx/PrecipitationAFNOv2.md
modules/generated/models/dx/SolarRadiationAFNO1H.md
modules/generated/models/dx/SolarRadiationAFNO6H.md
modules/generated/models/dx/StormScopeDxNSRDB.md
modules/generated/models/dx/TCTrackerWuDuan.md
modules/generated/models/dx/TCTrackerVitart.md
modules/generated/models/dx/WindgustAFNO.md
modules/generated/models/dx/Identity.md
{% endautosummary %}

<!-- mkdocs-badges:end -->
