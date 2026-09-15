# Examples

This is a collection of examples in Earth2Studio that demonstrate various functionality
and commonly used workflows.

??? info "Running Examples"
    Earth2Studio examples can be downloaded as a notebook or runnable Python script.
    Each requires installation of different optional dependency groups or additional
    packages for the specific models used or post-processing steps.
    Use uv to auto install dependencies on execution:

    ```bash
    uv run <example_script>.py
    ```

    If you are using a container or other type of environment, then pip installing will
    likely be needed.
    Look for the [uv inline metadata](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)
    blocks of the form:

    ```python
    # /// script
    # dependencies = [
    #   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
    #   "cartopy",
    # ]
    # ///
    ```

    Pip install these packages then execute the example with:

    ```bash
    python <example_script>.py
    ```
