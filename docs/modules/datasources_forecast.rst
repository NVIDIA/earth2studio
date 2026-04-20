.. _earth2studio.data.forecast:

:mod:`earth2studio.data`: Forecast Sources
-------------------------------------------

Extended data sources that allow users to download forecast data, these are not
interchangeable with standard data sources.
Typically used in intercomparison workflows.

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

      data.AIFS_FX
      data.CAMS_FX
      data.AIFS_ENS_FX
      data.GFS_FX
      data.GEFS_FX
      data.GEFS_FX_721x1440
      data.HRRR_FX
      data.IFS_FX
      data.IFS_ENS_FX
