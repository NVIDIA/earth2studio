# Documentation

Earth2Studio builds its documentation with
[MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Source layout

- `docs/modules/`: API landing pages consumed by `docs/generate_api.py`.
- `docs/userguide/`: conceptual and developer guides.
- `examples/`: executable gallery sources.
- `docs/examples/`: generated gallery pages; do not edit these directly.

The docs generators also create the model catalog, installation options, and scorecard pages.
`earth2studio-gallery` renders examples and retained execution results without Sphinx.

## Docstrings

Public classes and functions require docstrings; Interrogate enforces coverage. Use
[NumPy-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
with these conventions:

- Start the summary on the opening `"""` line.
- Document classes on the class definition, not `__init__`.
- Include argument and return types.
- Mark optional arguments and state their defaults.
- End complete sentences with periods.

```python
def timearray_to_datetime(time: TimeArray) -> list[datetime]:
    """Convert a NumPy datetime64 array into Python datetimes.

    Parameters
    ----------
    time : TimeArray
        NumPy datetime64 array.

    Returns
    -------
    list[datetime]
        Converted datetime objects.
    """
```

## Local builds

- `make docs`: generate all pages and build from retained example results.
- `make docs-full`: execute stale examples, regenerate the gallery, and build.
- `make docs-dev`: serve locally with live reload.
- `make docs-dev FILENAME=<selector>`: execute one example, then serve the complete gallery.

For example:

```bash
make docs-dev FILENAME=01_getting_started/01_deterministic_workflow.py
```

Build output is written to `site/`. Project-mode examples use the repository's `pyproject.toml`
and `uv.lock`, syncing only their declared extras.

## CI builds

- Pull requests run `make docs` only. They do not execute examples, upload the site, write caches,
  or deploy.
- Pushes to `main` run the cache-only docs workflow and deploy the site.
- The manual full workflow publishes the cache-only site, runs all eight example sections
  sequentially, then publishes the site again with the updated Gallery results.

Each example section has its own data cache. Gallery results use one rolling cache passed between
sections. Missing caches are valid cold starts; interrupted restores fail the job. Model downloads
are not cached. The remaining sections continue after a section failure, but the final publish
requires every section to succeed.

Use the `util-clear-cache` workflow to clear Gallery or docs-data caches.

## GitHub Pages

Keep `docs/.nojekyll` in the source tree. It allows GitHub Pages to serve generated directories
such as `_static/` and `examples/_assets/`.

Versioned deployments use [Mike](https://github.com/jimporter/mike):

```bash
DOC_VERSION=0.18.0 DOC_ALIAS=latest make docs-deploy-version
```

Versions are published beneath `v/` on the `gh-pages` branch.
