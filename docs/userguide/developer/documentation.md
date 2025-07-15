# Documentation

Earth2Studio uses Sphinx to build documentation that is hosted on Github pages.
We have the following philosophies for the three main sections of documentation:

1. API documentation is a requirement. [Interrogate](https://github.com/econchick/interrogate)
is used to enforce that all public methods are documented.

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

- The doc string description starts on the same line as the first `"""`.

- Class doc-strings are placed under the class definition not the `__init__` function.

- Type hints are included in the doc strings for each input argument / returned object.

- Optional/keyword arguments are denoted by `optional` following the type hint. The
default value is provided by adding ", by default [default value]" to the end of the
doc string.

- Periods should be used at the end of complete sentences, but are not required at the
end of "by default [default value]" or incomplete sentences

For VSCode users, the
[autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
is highly encouraged.
See the following doc-string samples for guidance.

```python
def timearray_to_datetime(time: TimeArray) -> list[datetime]:
    """Simple converter from numpy datetime64 array into a list of datetimes.

    Parameters
    ----------
    time : TimeArray
        Numpy datetime64 array

    Returns
    -------
    list[datetime]
        List of datetime object
    """
```

```python
class CorrelatedSphericalGaussian:
    """Produces Gaussian random field on the sphere with Matern covariance peturbation
    method output to a lat lon grid.

    Warning
    -------
    Presently this method generates noise on equirectangular grid of size [N, 2*N] when
    N is even or [N+1, 2*N] when N is odd.

    Parameters
    ----------
    noise_amplitude : float | torch.Tensor
        Overall amplitude scaling factor for the noise field. Must be provided.
    sigma : float, optional
        Standard deviation of the noise field, by default 1.0
    length_scale : float, optional
        Spatial correlation length scale in meters, by default 5.0e5
    time_scale : float, optional
        Temporal correlation scale in hours for the AR(1) process, by default 48.0

    Raises
    ------
    ValueError
        If noise_amplitude is not provided
    """
    ...
```

(examples_userguide)=

## Example Documentation

Examples in Earth2Studio are created with the intent to teach / demonstrate a specific
feature/workflow/concept/use case to users.
If you are interested in contributing an example, please reach out to us in a Github
issue to discuss further.
The example scripts used to populate the documentation are placed in the
[examples](https://github.com/NVIDIA/earth2studio/tree/main/examples) folder of the repo.
The python scripts and Jupyter notebooks present on the documentation webpage are auto
generated using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html).

### Definition

Examples demonstrate how to use Earth2Studio APIs.
Examples should be short and concise, designed to be ran in a short wall-clock time of
under 10 minutes on a typical data-center level GPU.
The script must be runnable on a single 32Gb A100 GPU using minimal extra dependencies.
Additional requirements may be required for running the example inside the CD pipeline,
which will be addressed on a case-by-case basis.

### Creating a New Example

In Earth2Studio, vanilla Jupyter lab notebooks are not used to avoid bloat in the commit
history.
Instead, [Sphinx Gallery](https://sphinx-gallery.github.io/stable/index.html) is used to
represent notebooks as code.
Sphinx Gallery supports embedded reST (reStructuredText) syntax in the form of comments.
These comments are automatically rendered as formatted text on the documentation web
pages and as markdown cells in the converted Jupyter notebooks.
For more details on the syntax, see the [embedded reST documentation](https://sphinx-gallery.github.io/stable/syntax.html#embed-rest-in-your-example-python-files).

To specify required dependencies, use [uv inline metadata](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>)
which allows each example to use different dependency groups or packages not shipped
with Earth2Studio.
To create an example, the best method is to copy an existing example and follow the
structure:

```python
# %%
"""
Example Title
==============

Brief one-liner of what this will do (used on hover of thumbnail in docs)

Then add a more verbose description on a new line here.
This can be multiply lines with references if needed
Include a list of the specific unique items this example includes, e.g.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Running a simple built in workflow
- Post-processing results
"""
# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "other depedencies...",
# ]
# ///

# %%
# Set Up
# ------
# RST section briefly describing set up of the needed components
#
# This should include an explicit list of key features like so, this enable cross-referencing
# in the API docs.
#
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Let's save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
# Add the following to set up output folder and env variables
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

# Import modules and initialize

# %%
# Execute the Workflow
# --------------------
# RST section briefly describing running the inference job

# %%
import earth2studio.run as run

# Execute inference

# %%
# Post-Processing
# ---------------
# RST section briefly describing what will be plotted

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Some post-processing part, save figure to outputs folder
plt.savefig("outputs/XX_my_example.jpg")
```

Examples should always have at least one or more graphics.
The first will always be the thumbnail.
Animations are not supported.

## Building Documentation

To build the core documentation without executing examples use:

```bash
make docs

# The above command required sphinx-build command.
# If it is not automatically found, try setting it explicitly.
SPHINXBUILD="python -m sphinx.cmd.build" make docs
```

For full docs, where all examples are ran, use:

```bash
make docs-full
```

Sometimes when developing an example you want to see what the end result looks like.
For development, it is recommend to use the following process:

```bash
# Run this once
make docs

# Run this to generate your example
make docs-dev FILENAME=<example filename .py>
```

Build files will always be in `docs/_build/html`.
Since the docs are static, Python can be used to host them [locally](http://localhost:8000):

```bash
cd docs/_build/html

python -m http.server 8000
```
