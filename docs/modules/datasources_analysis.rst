.. _earth2studio.data.analysis:

:mod:`earth2studio.data`: Data Sources
--------------------------------------

Data sources used for downloading, caching and reading different weather / climate data
APIs into `Xarray data arrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.
Used for fetching initial conditions for inference and validation data for scoring.

.. warning ::

   Each data source provided in Earth2Studio may have its own respective
   license. We encourage users to familiarize themselves with each and the limitations
   it may impose on their use case.

.. automodule:: earth2studio.data
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio

.. badge-filter:: region:global region:na region:as
   dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   :filter-mode: or
   :badge-order-fixed:
   :group-visibility-toggle:
   :group-hidden: product

   .. autosummary::
      :nosignatures:
      :toctree: generated/data/
      :template: datasource.rst

      data.ARCO
      data.CDS
      data.CMIP6
      data.CMIP6MultiRealm
      data.GFS
      data.GOES
      data.HRRR
      data.IFS
      data.IFS_ENS
      data.JPSS
      data.MRMS
      data.NClimGridDaily
      data.NCAR_ERA5
      data.PlanetaryComputerECMWFOpenDataIFS
      data.PlanetaryComputerGOES
      data.PlanetaryComputerMODISFire
      data.PlanetaryComputerOISST
      data.PlanetaryComputerSentinel3AOD
      data.Random
      data.WB2ERA5
      data.WB2ERA5_121x240
      data.WB2ERA5_32x64
      data.WB2Climatology
      data.DataArrayFile
      data.DataSetFile
      data.DataArrayPathList

:mod:`earth2studio.data`: AI Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data sources that leverage an AI model to generate weather / climate data that can be
used for downstream tasks in real time.
Unlike prognostic or diagnostic models, these sources do not require any input state
for subsequent predictions.

.. currentmodule:: earth2studio

.. badge-filter:: region:global region:na region:as
   dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   :filter-mode: or
   :badge-order-fixed:
   :group-visibility-toggle:
   :group-hidden: product

   .. autosummary::
      :toctree: generated/data/
      :template: diagnostic.rst

      data.CBottle3D
