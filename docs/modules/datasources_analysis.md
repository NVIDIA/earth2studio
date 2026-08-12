<!-- markdownlint-disable MD013 -->

# [`earth2studio.data`][earth2studio.data]: Data Sources

Data sources used for downloading, caching and reading different weather / climate data
APIs into [Xarray data arrays](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html).
Used for fetching initial conditions for inference and validation data for scoring.

!!! warning
    Each data source provided in Earth2Studio may have its own respective
    license. We encourage users to familiarize themselves with each and the limitations
    it may impose on their use case.

## Data Sources

<!-- e2s-autosummary
currentmodule: earth2studio
template: datasource
output: generated/data/analysis
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
- data.ARCO
- data.CDS
- data.CMIP6
- data.CMIP6MultiRealm
- data.DynamicalAIFS
- data.DynamicalAIFS_ENS
- data.DynamicalGFS
- data.DynamicalGEFS
- data.DynamicalHRRR
- data.DynamicalIFS_ENS
- data.DynamicalMRMS
- data.EarthMoverBrightBandIFS
- data.EarthMoverERA5
- data.GFS
- data.GOES
- data.GOESGLMGrid
- data.HimawariAHI
- data.HRRR
- data.IFS
- data.IFS_ENS
- data.JPSS
- data.MRMS
- data.MeteosatFCI
- data.NClimGridDaily
- data.NCAR_ERA5
- data.OPERA
- data.PlanetaryComputerECMWFOpenDataIFS
- data.PlanetaryComputerGOES
- data.PlanetaryComputerMODISFire
- data.PlanetaryComputerOISST
- data.PlanetaryComputerSentinel3AOD
- data.Random
- data.WB2ERA5
- data.WB2ERA5_121x240
- data.WB2ERA5_32x64
- data.WB2Climatology
- data.DataArrayFile
- data.DataSetFile
- data.DataArrayPathList
-->

<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true hidden="product" -->

{% autosummary %}
modules/generated/data/analysis/data_ARCO.md
modules/generated/data/analysis/data_CDS.md
modules/generated/data/analysis/data_CMIP6.md
modules/generated/data/analysis/data_CMIP6MultiRealm.md
modules/generated/data/analysis/data_DynamicalAIFS.md
modules/generated/data/analysis/data_DynamicalAIFS_ENS.md
modules/generated/data/analysis/data_DynamicalGFS.md
modules/generated/data/analysis/data_DynamicalGEFS.md
modules/generated/data/analysis/data_DynamicalHRRR.md
modules/generated/data/analysis/data_DynamicalIFS_ENS.md
modules/generated/data/analysis/data_DynamicalMRMS.md
modules/generated/data/analysis/data_EarthMoverBrightBandIFS.md
modules/generated/data/analysis/data_EarthMoverERA5.md
modules/generated/data/analysis/data_GFS.md
modules/generated/data/analysis/data_GOES.md
modules/generated/data/analysis/data_GOESGLMGrid.md
modules/generated/data/analysis/data_HimawariAHI.md
modules/generated/data/analysis/data_HRRR.md
modules/generated/data/analysis/data_IFS.md
modules/generated/data/analysis/data_IFS_ENS.md
modules/generated/data/analysis/data_JPSS.md
modules/generated/data/analysis/data_MRMS.md
modules/generated/data/analysis/data_MeteosatFCI.md
modules/generated/data/analysis/data_NClimGridDaily.md
modules/generated/data/analysis/data_NCAR_ERA5.md
modules/generated/data/analysis/data_OPERA.md
modules/generated/data/analysis/data_PlanetaryComputerECMWFOpenDataIFS.md
modules/generated/data/analysis/data_PlanetaryComputerGOES.md
modules/generated/data/analysis/data_PlanetaryComputerMODISFire.md
modules/generated/data/analysis/data_PlanetaryComputerOISST.md
modules/generated/data/analysis/data_PlanetaryComputerSentinel3AOD.md
modules/generated/data/analysis/data_Random.md
modules/generated/data/analysis/data_WB2ERA5.md
modules/generated/data/analysis/data_WB2ERA5_121x240.md
modules/generated/data/analysis/data_WB2ERA5_32x64.md
modules/generated/data/analysis/data_WB2Climatology.md
modules/generated/data/analysis/data_DataArrayFile.md
modules/generated/data/analysis/data_DataSetFile.md
modules/generated/data/analysis/data_DataArrayPathList.md
{% endautosummary %}

<!-- mkdocs-badges:end -->

## AI Data Sources

<!-- e2s-autosummary
currentmodule: earth2studio
template: diagnostic
output: generated/data/analysis
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
- data.CBottle3D
-->

<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true hidden="product" -->

{% autosummary %}
modules/generated/data/analysis/data_CBottle3D.md
{% endautosummary %}

<!-- mkdocs-badges:end -->
