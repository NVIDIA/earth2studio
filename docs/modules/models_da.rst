.. _earth2studio.models.da:

:mod:`earth2studio.models.da`: Data Assimilation
------------------------------------------------

.. automodule:: earth2studio.models.da
    :no-members:
    :no-inherited-members:

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

.. badge-filter:: region:global region:na class:nwc class:ds class:mrf class:s2s class:da year:2021 year:2022 year:2023 year:2024 year:2025 domain:wind domain:precip domain:temp domain:atmos domain:ocean gpu:96gb gpu:80gb gpu:48gb gpu:40gb gpu:24gb
   :filter-mode: or

   .. autosummary::
      :nosignatures:
      :toctree: generated/models/da/
      :template: dataassim.rst

      HealDA
      InterpEquirectangular
      StormCastSDA
