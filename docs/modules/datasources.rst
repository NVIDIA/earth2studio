.. _earth2studio.data:

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

.. autosummary::
   :toctree: generated/data/
   :template: datasource.rst

   data.ARCO
   data.CDS
   data.GFS
   data.HRRR
   data.IFS
   data.IMERG
   data.NCAR_ERA5
   data.Random
   data.WB2ERA5
   data.WB2ERA5_121x240
   data.WB2ERA5_32x64
   data.WB2Climatology
   data.DataArrayFile
   data.DataSetFile
   data.DataArrayPathList

Forecast Sources
~~~~~~~~~~~~~~~~

Extended data sources that allow users to download forecast data, these are not
interchangable with standard data sources.
Typically used in intercomparison workflows.

.. currentmodule:: earth2studio

.. autosummary::
   :toctree: generated/data/
   :template: datasource.rst

   data.GFS_FX
   data.GEFS_FX
   data.GEFS_FX_721x1440
   data.HRRR_FX

Functions
~~~~~~~~~
.. currentmodule:: earth2studio

.. autosummary::
   :toctree: generated/data/
   :template: function.rst

   data.datasource_to_file
   data.fetch_data
   data.prep_data_array
