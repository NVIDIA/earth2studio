.. _earth2studio.data.dataframe:

:mod:`earth2studio.data`: DataFrame Sources
--------------------------------------------

Data sources that provide tabular data as DataFrames.

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

      data.ISD
      data.JPSS_ATMS
      data.MetOpAMSUA
      data.MetOpAVHRR
      data.RandomDataFrame
      data.UFSObsConv
      data.UFSObsSat
