# Documentation

Earth2Studio uses Sphinx to build documentation that is hosted on Github pages.
We have the following philosophies for the three main sections of documentation:

1. API documentation is a requirement. [Interogate](https://github.com/econchick/interrogate)
is used enforce that all public methods are documented.

2. Examples are generated using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html)
and are used as end-to-end tests for the package.

3. User guide is written using [MyST](https://myst-parser.readthedocs.io/en/latest/index.html)
and aims to document concepts that cannot be fully communicated in examples.

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
