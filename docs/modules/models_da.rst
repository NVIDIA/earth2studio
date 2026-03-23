.. _earth2studio.models.da:

:mod:`earth2studio.models.da`: Data Assimilation
------------------------------------------------

.. automodule:: earth2studio.models.da
    :no-members:
    :no-inherited-members:

Data assimilation models are a class of models that integrate observational data into
model states or grids. These models can ingest both sparse observations (via
DataFrames) and dense fields (via DataArrays) to produce output suitable for
downstream tasks such as driving a prognostic model or generating a guided forecast.
Data assimilation models support both stateless and stateful operation, allowing them
to process observations independently or maintain internal state across time steps.

.. warning ::

   Data Assimilation models are a new addition to Earth2Studio and APIs might be subject
   to change without warning while the implementation is hardened.

.. currentmodule:: earth2studio.models.da

.. badge-filter:: region:global region:na region:as
   class:nwc class:ds class:mrf class:s2s class:da class:cm
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   year:2021 year:2022 year:2023 year:2024 year:2025 year:2026
   gpu:96gb gpu:80gb gpu:48gb gpu:40gb
   :filter-mode: or
   :badge-order-fixed:

   .. autosummary::
      :nosignatures:
      :toctree: generated/models/da/
      :template: dataassim.rst

      HealDA
      InterpEquirectangular
      StormCastSDA
