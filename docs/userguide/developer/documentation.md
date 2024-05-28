# Documentation

Earth2Studio uses Sphinx to build documentation that is hosted on Github pages.
We have the following philosophies for the three main sections of documentation:

1. API documentation is a requirement. [Interogate](https://github.com/econchick/interrogate)
is used enforce that all public methods are documented.

2. Examples are generated using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html)
and are used as end-to-end tests for the package.

3. User guide is written using [MyST](https://myst-parser.readthedocs.io/en/latest/index.html)
and aims to document concepts that cannot be fully communicated in examples.

## API Documentation

API documentation or doc-strings are a requirement for public Earth2studio classes and
functions.
Consistent documentation styling improves user and developer experience.
To make the doc-strings between different parts of the code as consistent as possible,
the following styles are used:

- [NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
doc-strings are used in all Python files.

- Class doc-strings are placed under the class definition not the `__init__` function.

- Type hints are included in the doc strings for each input argument / returned object.

- Optional/keyword arguments are denoted by `optional` following the type hint. The
default value is provided by adding ", by default [default value]." to the end of the
doc string.

- Periods should be used at the end of all sentences.

For VSCode users, the
[autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
is highly encouraged.
See the following doc-string samples for guidance.

```python
def handshake_dim(
    input_coords: CoordSystem,
    required_dim: str,
    required_index: int | None = None,
) -> None:
    """Simple check to see if coordinate system has a dimension in a particular index.

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinate system to validate.
    required_dim : str
        Required dimension (name of coordinate).
    required_index : int, optional
        Required index of dimension if needed, by default None.

    Raises
    ------
    KeyError
        If required dimension is not found in the input coordinate system.
    ValueError
        If the required index is outside the dimensionality of the input coordinate system.
    ValueError
        If dimension is not in the required index.

    Returns
    -------
        None
    """
```

```python
class Random:
    """A randomly generated normally distributed data. Primarily useful for testing.

    Parameters
    ----------
    domain_coords: OrderedDict[str, np.ndarray]
        Domain coordinates that the random data will assume (such as lat, lon).
    """

    def __init__(
        self,
        domain_coords: OrderedDict[str, np.ndarray],
    ):
        # ...
```

## Example Documentation

Examples in Earth2Studio are created with the intent to teach / demonstrate a specific
feature/workflow/concept/use case to users.
If you are interested in contributing an example, please reach out to us in a Github
issue to discuss further.
The example scripts used to populate the documentation are placed in the
[examples](https://github.com/NVIDIA/earth2studio/tree/main/examples) folder of the repo.
The python scripts and Jupyter notebooks present on the documentation webpage are auto
generated using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html).

Examples should be short and concise, designed to be ran in a short wall-clock time of
under 10 minutes on a typical data-center level GPU.
The script must be runnable on a single 32Gb A100 GPU using minimal extra dependencies.
Additional requirements may be required for running the example inside the CD pipeline,
which will be addressed on a case-by-case basis.

## Building Documentation

To build the core documentation without executing examples use:

```bash
make docs
```

For full docs, where all examples are ran use:

```bash
make docs-full
```

Sometimes when developing an example you want to see what the end result looks like.
For development, its recommend to use the following process:

```bash
# Run this once
make docs

# Run this to generate your example
make docs-dev FILENAME=<example filename .py>
```

Build files will always be in `docs/_build/html`.
