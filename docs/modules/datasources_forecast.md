<!-- markdownlint-disable MD013 -->

# [`earth2studio.data`][earth2studio.data]: Forecast Sources

Extended data sources that allow users to download forecast data, these are not
interchangeable with standard data sources.
Typically used in intercomparison workflows.

<!-- e2s-autosummary
currentmodule: earth2studio
template: datasource
output: generated/data/forecast
badges:
- region:global
- region:na
- region:eu
- region:as
- region:au
- region:af
- region:sa
- dataclass:analysis
- dataclass:reanalysis
- dataclass:observation
- dataclass:simulation
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
filter:
  mode: or
  order: fixed
  toggle: 'true'
  hidden: product
objects:
- data.AIFS_FX
- data.CAMS_FX
- data.AIFS_ENS_FX
- data.CFS_FX
- data.CFS_FX_Flux
- data.CFS_Reforecast_FX
- data.CFS_Reforecast_FX_Flux
- data.DynamicalGFS_FX
- data.DynamicalGEFS_FX
- data.DynamicalHRRR_FX
- data.DynamicalICON_EU_FX
- data.DynamicalIFS_ENS_FX
- data.DynamicalAIFS_FX
- data.DynamicalAIFSENS_FX
- data.EarthMoverBrightBandIFS_FX
- data.GFS_FX
- data.GEFS_FX
- data.GEFS_FX_721x1440
- data.HRRR_FX
- data.IFS_FX
- data.IFS_ENS_FX
-->

<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true hidden="product" -->

{% autosummary %}
modules/generated/data/forecast/data_AIFS_FX.md
modules/generated/data/forecast/data_CAMS_FX.md
modules/generated/data/forecast/data_AIFS_ENS_FX.md
modules/generated/data/forecast/data_CFS_FX.md
modules/generated/data/forecast/data_CFS_FX_Flux.md
modules/generated/data/forecast/data_CFS_Reforecast_FX.md
modules/generated/data/forecast/data_CFS_Reforecast_FX_Flux.md
modules/generated/data/forecast/data_DynamicalGFS_FX.md
modules/generated/data/forecast/data_DynamicalGEFS_FX.md
modules/generated/data/forecast/data_DynamicalHRRR_FX.md
modules/generated/data/forecast/data_DynamicalICON_EU_FX.md
modules/generated/data/forecast/data_DynamicalIFS_ENS_FX.md
modules/generated/data/forecast/data_DynamicalAIFS_FX.md
modules/generated/data/forecast/data_DynamicalAIFSENS_FX.md
modules/generated/data/forecast/data_EarthMoverBrightBandIFS_FX.md
modules/generated/data/forecast/data_GFS_FX.md
modules/generated/data/forecast/data_GEFS_FX.md
modules/generated/data/forecast/data_GEFS_FX_721x1440.md
modules/generated/data/forecast/data_HRRR_FX.md
modules/generated/data/forecast/data_IFS_FX.md
modules/generated/data/forecast/data_IFS_ENS_FX.md
{% endautosummary %}

<!-- mkdocs-badges:end -->
