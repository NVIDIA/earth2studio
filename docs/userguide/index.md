(userguide)=

# User Guide

Welcome to the Earth2Studio user guide.
This guide provides a verbose documentation of the package and the underlying
design.
If you want to skip to running code, have a look at the examples instead
and come back here when you have questions.

In this user guide, we'll delve into the intricacies of Earth2Studio,
exploring its fundamental components, features, and the ways in which
it can be extended and customized to suit specific research or production needs.
Whether you're a seasoned expert or just beginning your journey in the realm of
AI-driven weather and climate analysis, this guide aims to equip you with the knowledge
and resources necessary to leverage the full potential of Earth2Studio.

## Quick Start

Install Earth2Studio:

```bash
pip install earth2studio[dlwp]
```

Run a deterministic weather prediction in just a few lines of code:

```python
from earth2studio.models.px import DLWP
from earth2studio.data import GFS
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic as run

model = DLWP.load_model(DLWP.load_default_package())
ds = GFS()
io = NetCDF4Backend("output.nc")

run(["2024-01-01"], 10, model, ds, io)
```

## About

- [Install](about/install)
- [Introduction](about/intro)
- [Data Movement](about/data)

## Core Components

- [Prognostic Models](components/prognostic)
- [Diagnostic Models](components/diagnostic)
- [Datasources](components/datasources)
- [Perturbations](components/perturbation)
- [Statistics](components/statistics)
- [IO Backends](components/io)

## Advanced Usage

- [Batch Dimension](advanced/batch)
- [AutoModels](advanced/auto)
- [Lexicon](advanced/lexicon)

## Developer Guide

- [Overview](developer/overview)
- [Dependencies](developer/dependency)
- [Style](developer/style)
- [Documentation](developer/documentation)
- [Testing](developer/testing)
- [Build](developer/build)
- [Recipes](developer/recipes)

## Support

- [Troubleshooting](support/troubleshooting)
- [Frequently Asked Questions](support/faq)

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/data

```

```{toctree}
:caption: Core Components
:maxdepth: 1
:hidden:

components/prognostic
components/diagnostic
components/datasources
components/perturbation
components/io
components/statistics
```

```{toctree}
:caption: Advanced Usage
:maxdepth: 1
:hidden:

advanced/batch
advanced/auto
advanced/lexicon
```

```{toctree}
:caption: Developer Guide
:maxdepth: 1
:hidden:

developer/overview
developer/dependency
developer/style
developer/documentation
developer/testing
developer/build
developer/recipes
```

```{toctree}
:caption: Support
:maxdepth: 1
:hidden:
support/troubleshooting
support/faq
```
