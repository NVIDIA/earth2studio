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
   data.Random
   data.DataArrayFile
   data.DataSetFile
   data.WB2Climatology

Functions
~~~~~~~~~
.. currentmodule:: earth2studio

.. autosummary::
   :toctree: generated/data/
   :template: function.rst

   data.datasource_to_file
   data.fetch_data
   data.prep_data_array
