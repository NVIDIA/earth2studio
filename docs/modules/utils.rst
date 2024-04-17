.. _earth2studio.utils_api:

:mod:`earth2studio.utils`: Utils
---------------------------------

.. automodule:: earth2studio.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio


:mod:`earth2studio.utils`: Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A collection of utilities to manipulate and check coordinate systems dictionaries.

.. autosummary::
   :toctree: generated/utils/
   :template: function.rst

   utils.coords.handshake_dim
   utils.coords.handshake_coords
   utils.coords.handshake_size
   utils.coords.map_coords
   utils.coords.extract_coords

:mod:`earth2studio.utils`: Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A collection of utilities to help interact with time coordinate arrays. Earth2Studio
uses `np.datetime64[ns]` to represent time.
The following functions can be used to convert to and from these numpy arrays.

.. autosummary::
   :toctree: generated/utils/
   :template: function.rst

   utils.time.timearray_to_datetime
   utils.time.to_time_array
