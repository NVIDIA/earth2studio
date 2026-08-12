<!-- markdownlint-disable MD013 -->

# [`earth2studio.models.px`][earth2studio.models.px]: Prognostics

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
- ACE2ERA5
- AIFS
- AIFS2
- AIFS2ENS
- AIFSENS
- Atlas
- Aurora
- Aurora1p5
- Aurora1p5Ensemble
- CBottleVideo
- DataReplay
- DiagnosticWrapper
- DLESyM
- DLESyMLatLon
- DLESyMv0_ISCCP_ERA5
- DLESyMv0_ISCCP_ERA5LatLon
- DLWP
- FCN
- FCN3
- FengWu
- FuXi
- GenCastMini
- GraphCastOperational
- GraphCastSmall
- InterpModAFNO
- Pangu24
- Pangu6
- Pangu3
- Persistence
- SFNO
- StormCast
- StormCastCONUS
- StormScopeGOES
- StormScopeMRMS
- UCast
-->

<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa class:nwc class:ds class:mrf class:s2s class:da class:cm product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu year:2021 year:2022 year:2023 year:2024 year:2025 year:2026 gpu:96gb gpu:80gb gpu:48gb gpu:40gb mode=or order=fixed toggle=true hidden="product year" -->

{% autosummary %}
modules/generated/models/px/ACE2ERA5.md
modules/generated/models/px/AIFS.md
modules/generated/models/px/AIFS2.md
modules/generated/models/px/AIFS2ENS.md
modules/generated/models/px/AIFSENS.md
modules/generated/models/px/Atlas.md
modules/generated/models/px/Aurora.md
modules/generated/models/px/Aurora1p5.md
modules/generated/models/px/Aurora1p5Ensemble.md
modules/generated/models/px/CBottleVideo.md
modules/generated/models/px/DataReplay.md
modules/generated/models/px/DiagnosticWrapper.md
modules/generated/models/px/DLESyM.md
modules/generated/models/px/DLESyMLatLon.md
modules/generated/models/px/DLESyMv0_ISCCP_ERA5.md
modules/generated/models/px/DLESyMv0_ISCCP_ERA5LatLon.md
modules/generated/models/px/DLWP.md
modules/generated/models/px/FCN.md
modules/generated/models/px/FCN3.md
modules/generated/models/px/FengWu.md
modules/generated/models/px/FuXi.md
modules/generated/models/px/GenCastMini.md
modules/generated/models/px/GraphCastOperational.md
modules/generated/models/px/GraphCastSmall.md
modules/generated/models/px/InterpModAFNO.md
modules/generated/models/px/Pangu24.md
modules/generated/models/px/Pangu6.md
modules/generated/models/px/Pangu3.md
modules/generated/models/px/Persistence.md
modules/generated/models/px/SFNO.md
modules/generated/models/px/StormCast.md
modules/generated/models/px/StormCastCONUS.md
modules/generated/models/px/StormScopeGOES.md
modules/generated/models/px/StormScopeMRMS.md
modules/generated/models/px/UCast.md
{% endautosummary %}

<!-- mkdocs-badges:end -->
