.. _earth2studio.models.px:

:mod:`earth2studio.models.px`: Prognostics
------------------------------------------

.. automodule:: earth2studio.models.px
    :no-members:
    :no-inherited-members:

Prognostic models are a class of models that perform time-integration.
Thus are typically used to generate forecast predictions.

.. warning ::

   Pre-trained prognostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. currentmodule:: earth2studio.models.px

.. badge-filter:: region:global region:na region:as
   class:nwc class:ds class:mrf class:s2s class:da class:cm
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   year:2021 year:2022 year:2023 year:2024 year:2025 year:2026
   gpu:96gb gpu:80gb gpu:48gb gpu:40gb
   :filter-mode: or
   :badge-order-fixed:
   :group-visibility-toggle:
   :group-hidden: product year

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
