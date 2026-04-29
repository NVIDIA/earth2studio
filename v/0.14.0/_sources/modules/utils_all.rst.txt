.. _earth2studio.utils_api:

:mod:`earth2studio`: Utilities
-------------------------------

.. _earth2studio.utils.coords:

:mod:`earth2studio.utils`: Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: earth2studio.utils
    :no-members:
    :no-inherited-members:

A collection of utilities to manipulate and check coordinate systems dictionaries.

.. currentmodule:: earth2studio

.. autosummary::
   :toctree: generated/utils/
   :template: function.rst

   utils.coords.handshake_dim
   utils.coords.handshake_coords
   utils.coords.handshake_size
   utils.coords.map_coords
   utils.coords.split_coords

.. autosummary::
   :toctree: generated/utils/
   :template: class.rst

   utils.interp.LatLonInterpolation

.. _earth2studio.utils.time:

:mod:`earth2studio.utils`: Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A collection of utilities to help interact with time coordinate arrays. Earth2Studio
uses ``np.datetime64[ns]`` to represent time.
The following functions can be used to convert to and from these numpy arrays.

.. autosummary::
   :toctree: generated/utils/
   :template: function.rst

   utils.time.timearray_to_datetime
   utils.time.to_time_array

.. _earth2studio.data.functions:

:mod:`earth2studio.data`: Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: earth2studio

.. autosummary::
   :toctree: generated/data/
   :template: function.rst

   data.datasource_to_file
   data.fetch_data
   data.prep_data_array

.. _earth2studio.models.utils_api:

:mod:`earth2studio.models`: Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: earth2studio.models
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio.models

.. autosummary::
   :nosignatures:
   :toctree: generated/models/
   :template: class.rst

   auto.Package
   batch.batch_func
