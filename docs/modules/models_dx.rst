.. _earth2studio.models.dx:

:mod:`earth2studio.models.dx`: Diagnostics
------------------------------------------

.. automodule:: earth2studio.models.dx
    :no-members:
    :no-inherited-members:

Diagnostic models are a class of models that do not perform time-integration.
These may be used to map between weather/climate variables to other quantities of
interest, used to enable additional analysis, improve prediction accuracy, downscale,
etc.

.. warning ::

   Pre-trained diagnostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. currentmodule:: earth2studio.models.dx

.. badge-filter:: region:global region:na region:as
   class:nwc class:ds class:mrf class:s2s class:da class:cm
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   year:2021 year:2022 year:2023 year:2024 year:2025 year:2026
   gpu:96gb gpu:80gb gpu:48gb gpu:40gb
   :filter-mode: or
   :badge-order-fixed:

   .. autosummary::
      :nosignatures:
      :toctree: generated/models/dx/
      :template: diagnostic.rst

      CBottleInfill
      CBottleSR
      CBottleTCGuidance
      CorrDiffCMIP6
      CorrDiffTaiwan
      ClimateNet
      DerivedRH
      DerivedRHDewpoint
      DerivedSurfacePressure
      DerivedTCWV
      DerivedVPD
      DerivedWS
      PrecipitationAFNO
      PrecipitationAFNOv2
      SolarRadiationAFNO1H
      SolarRadiationAFNO6H
      TCTrackerWuDuan
      TCTrackerVitart
      WindgustAFNO
      Identity
