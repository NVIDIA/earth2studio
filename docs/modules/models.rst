:mod:`earth2studio.models`: Models
----------------------------------
.. automodule:: earth2studio.models
    :no-members:
    :no-inherited-members:


.. _earth2studio.models.px:

:mod:`earth2studio.models.px`: Prognostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prognostic models are a class of models that perform time-integration.
Thus are typically used to generate forecast predictions.

.. warning ::

   Pre-trained prognostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. currentmodule:: earth2studio.models.px

.. badge-filter:: region:global region:na class:nwc class:ds class:mrf class:s2s class:da year:2021 year:2022 year:2023 year:2024 year:2025 domain:wind domain:precip domain:temp domain:atmos domain:ocean gpu:96gb gpu:80gb gpu:48gb gpu:40gb gpu:24gb
   :filter-mode: or

   .. autosummary::
      :nosignatures:
      :toctree: generated/models/px/
      :template: prognostic.rst

      ACE2ERA5
      AIFS
      AIFSENS
      Atlas
      Aurora
      CBottleVideo
      DiagnosticWrapper
      DLESyM
      DLESyMLatLon
      DLWP
      FCN
      FCN3
      FengWu
      FuXi
      GraphCastOperational
      GraphCastSmall
      InterpModAFNO
      Pangu24
      Pangu6
      Pangu3
      Persistence
      SFNO
      StormCast
      StormScopeGOES
      StormScopeMRMS

.. _earth2studio.models.dx:

:mod:`earth2studio.models.dx`: Diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diagnostic models are a class of models that do not perform time-integration.
These may be used to map between weather/climate variables to other quantities of
interest, used to enbable additional analysis, improve prediction accuracy, downscale,
etc.

.. warning ::

   Pre-trained diagnostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. currentmodule:: earth2studio.models.dx

.. badge-filter:: region:global region:us application:nowcasting application:downscaling application:medium-range application:seasonal application:data-assimilation year:2021 year:2022 year:2023 year:2024 year:2025 field:winds field:precipitation field:temperature field:ocean gpu:96gb gpu:80gb gpu:48gb gpu:40gb gpu:24gb
   :filter-mode: or

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

:mod:`earth2studio.models.da`: Data Assimilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data assimilation models are a class of models that incorporate observational data
into model states or grids. These models process sparse, irregularly-distributed
observations (typically from DataFrames) and map them to regular grids or model
coordinate systems (typically as DataArrays). Data assimilation models support both
stateless and stateful operation, allowing them to process observations independently
or maintain internal state across time steps.

.. warning ::

   Data Assimilation models are a new addition to Earth2Studio and APIs might be subject
   to change without warning while the implementation is hardened.

.. currentmodule:: earth2studio.models.da

.. badge-filter:: region:global region:us application:nowcasting application:downscaling application:medium-range application:seasonal application:data-assimilation year:2021 year:2022 year:2023 year:2024 year:2025 field:winds field:precipitation field:temperature field:ocean gpu:96gb gpu:80gb gpu:48gb gpu:40gb gpu:24gb
   :filter-mode: or

   .. autosummary::
      :nosignatures:
      :toctree: generated/models/da/
      :template: dataassim.rst

      HealDA
      InterpEquirectangular
      StormCastSDA

:mod:`earth2studio.models`: Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: earth2studio.models

.. autosummary::
   :nosignatures:
   :toctree: generated/models/
   :template: class.rst

   auto.Package
   batch.batch_func
