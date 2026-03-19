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

.. autosummary::
   :nosignatures:
   :toctree: generated/models/px/
   :template: prognostic.rst

   AIFS
   Aurora
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

.. autosummary::
   :nosignatures:
   :toctree: generated/models/dx/
   :template: diagnostic.rst

   CBottleInfill
   CBottleSR
   CorrDiffTaiwan
   ClimateNet
   DerivedRH
   DerivedRHDewpoint
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

:mod:`earth2studio.models`: Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: earth2studio.models

.. autosummary::
   :nosignatures:
   :toctree: generated/models/
   :template: class.rst

   auto.Package
   batch.batch_func
