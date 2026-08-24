<!-- markdownlint-disable MD013 -->

# Data Sources

`earth2studio.data`

Data sources used for downloading, caching and reading different weather / climate data
APIs into [Xarray data arrays](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html).
Used for fetching initial conditions for inference and validation data for scoring.

!!! warning
    Each data source provided in Earth2Studio may have its own respective
    license. We encourage users to familiarize themselves with each and the limitations
    it may impose on their use case.

## Available Data Sources

<!-- e2s-autosummary
currentmodule: earth2studio
template: datasource
output: generated/data/analysis
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true labels=label hidden="product" -->
{% autosummary %}
earth2studio.data.ARCO
earth2studio.data.CDS
earth2studio.data.CMIP6
earth2studio.data.CMIP6MultiRealm
earth2studio.data.DynamicalAIFS
earth2studio.data.DynamicalAIFS_ENS
earth2studio.data.DynamicalGFS
earth2studio.data.DynamicalGEFS
earth2studio.data.DynamicalHRRR
earth2studio.data.DynamicalIFS_ENS
earth2studio.data.DynamicalMRMS
earth2studio.data.EarthMoverBrightBandIFS
earth2studio.data.EarthMoverERA5
earth2studio.data.GFS
earth2studio.data.GOES
earth2studio.data.GOESGLMGrid
earth2studio.data.HimawariAHI
earth2studio.data.HRRR
earth2studio.data.IFS
earth2studio.data.IFS_ENS
earth2studio.data.JPSS
earth2studio.data.MRMS
earth2studio.data.MeteosatFCI
earth2studio.data.NClimGridDaily
earth2studio.data.NCAR_ERA5
earth2studio.data.OPERA
earth2studio.data.PlanetaryComputerECMWFOpenDataIFS
earth2studio.data.PlanetaryComputerGOES
earth2studio.data.PlanetaryComputerMODISFire
earth2studio.data.PlanetaryComputerOISST
earth2studio.data.PlanetaryComputerSentinel3AOD
earth2studio.data.Random
earth2studio.data.SamudrACEData
earth2studio.data.SamudrACEForcingData
earth2studio.data.WB2ERA5
earth2studio.data.WB2ERA5_121x240
earth2studio.data.WB2ERA5_32x64
earth2studio.data.WB2Climatology
earth2studio.data.DataArrayFile
earth2studio.data.DataSetFile
earth2studio.data.DataArrayPathList
{% endautosummary %}

<!-- mkdocs-badges:end -->

## AI Data Sources

<!-- e2s-autosummary
currentmodule: earth2studio
template: diagnostic
output: generated/data/analysis
-->
<!-- mkdocs-badges:filter region:global region:na region:eu region:as region:au region:af region:sa dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu mode=or order=fixed toggle=true labels=label hidden="product" -->
{% autosummary %}
earth2studio.data.CBottle3D
{% endautosummary %}

<!-- mkdocs-badges:end -->
