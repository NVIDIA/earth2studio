# Documentation

Earth2Studio uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme to build
static documentation hosted on GitHub Pages. The docs are organized around three
main areas:

1. API documentation is required for public Earth2Studio classes and functions.
   [Interrogate](https://github.com/econchick/interrogate) is used to enforce that
   public methods are documented.

2. Examples are rendered with `earth2studio-gallery`, which executes percent-format
   Python examples in the project documentation environment and then renders the
   retained results into a MkDocs Material gallery.

3. API landing pages are Markdown files in `docs/modules/` with autosummary
   blocks read by `docs/generate_api.py`. Model and data-source badges are rendered
   and filtered with `mkdocs-badges`.

4. The user guide is written in Markdown and documents concepts that cannot be fully
   communicated in examples.

## API Documentation

API documentation or doc-strings are a requirement for public Earth2Studio classes and
functions. Consistent documentation styling improves user and developer experience. To
make doc-strings between different parts of the code as consistent as possible, the
following styles are used:

- [NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
  doc-strings are used in all Python files.

- The doc string description starts on the same line as the first `"""`.

- Class doc-strings are placed under the class definition not the `__init__` function.

- Type hints are included in the doc strings for each input argument / returned object.

- Optional/keyword arguments are denoted by `optional` following the type hint. The
  default value is provided by adding ", by default [default value]" to the end of the
  doc string.

- Periods should be used at the end of complete sentences, but are not required at the
  end of "by default [default value]" or incomplete sentences.

For VSCode users, the
[autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
is highly encouraged. Refer to the following doc-string samples for guidance:

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

## Example Documentation

Examples in Earth2Studio are created with the intent to teach or demonstrate a specific
feature, workflow, concept, or use case to users. If you are interested in contributing
an example, reach out to us in a GitHub issue to discuss further. The example scripts
used to populate the documentation are placed in the
[examples](https://github.com/NVIDIA/earth2studio/tree/main/examples) folder of the repo.

The MkDocs documentation uses `earth2studio-gallery` instead of Sphinx Gallery. The
source examples remain in the repository under `examples/`, and generated gallery pages
are written to `docs/examples/` during the docs build.

## Building Documentation

To build the documentation locally, use:

```bash
make docs
```

For the full documentation environment, including package extras and executed examples,
use:

```bash
make docs-full
```

For local development with live reload, use:

```bash
make docs-dev
```

To execute and refresh a single example before serving the site, pass a gallery selector:

```bash
make docs-dev FILENAME=01_getting_started/01_deterministic_workflow.py
```

Build files are written to `site/`.

The empty `docs/.nojekyll` file is intentionally kept in the MkDocs `docs_dir`.
MkDocs copies it to the root of the built site, which tells GitHub Pages to serve
asset directories such as `_static/` and `examples/_assets/` without Jekyll
filtering. Keep this as a source file rather than adding a custom copy step to the
docs build pipeline.

## Versioning

Documentation versioning uses [Mike](https://github.com/jimporter/mike). To build and
publish a versioned docs tree from a release branch, set `DOC_VERSION` and optionally
`DOC_ALIAS`:

```bash
DOC_VERSION=0.18.0 DOC_ALIAS=latest make docs-deploy-version
```

Versioned docs are deployed under the `v/` prefix on the `gh-pages` branch.
