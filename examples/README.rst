Examples
========

This is a collection of examples in Earth2Studio that demonstrate various functionality
and commonly used workflows.

.. dropdown:: Running Examples
    :color: info
    :icon: rocket

    Earth2Studio examples can be downloaded as a notebook or runnable Python script.
    Each requires installation of different optional dependency groups or additional
    packages for the specific models used or post-processing steps.
    Use uv to auto install dependencies on execution:

    ``uv run <example_script>.py``

    If you are using a container or other type of environment, then pip installing will
    likely be needed.
    Look for the `uv inline metadata <https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>`_
    blocks of the form:

    .. code-block:: python

        # /// script
        # dependencies = [
        #   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
        #   "cartopy",
        # ]
        # ///

    Pip install these packages then execute the example with:

    ``python <example_script>.py``
