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
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true hidden="product" -->
{% autosummary %}
earth2studio.data.AIFS_FX
earth2studio.data.CAMS_FX
earth2studio.data.AIFS_ENS_FX
earth2studio.data.CFS_FX
earth2studio.data.CFS_FX_Flux
earth2studio.data.CFS_Reforecast_FX
earth2studio.data.CFS_Reforecast_FX_Flux
earth2studio.data.DynamicalGFS_FX
earth2studio.data.DynamicalGEFS_FX
earth2studio.data.DynamicalHRRR_FX
earth2studio.data.DynamicalICON_EU_FX
earth2studio.data.DynamicalIFS_ENS_FX
earth2studio.data.DynamicalAIFS_FX
earth2studio.data.DynamicalAIFSENS_FX
earth2studio.data.EarthMoverBrightBandIFS_FX
earth2studio.data.GFS_FX
earth2studio.data.GEFS_FX
earth2studio.data.GEFS_FX_721x1440
earth2studio.data.HRRR_FX
earth2studio.data.IFS_FX
earth2studio.data.IFS_ENS_FX
{% endautosummary %}

<!-- mkdocs-badges:end -->
