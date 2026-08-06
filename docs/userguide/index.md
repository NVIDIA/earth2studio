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

## Getting Started

- [Install](about/install.md)
- [Introduction](about/intro.md)
- [Data Movement](about/data.md)

## Core Components

- [Prognostic Models](components/prognostic.md)
- [Diagnostic Models](components/diagnostic.md)
- [Data Sources](components/datasources.md)
- [Perturbations](components/perturbation.md)
- [Statistics](components/statistics.md)
- [IO Backends](components/io.md)

## Advanced Usage

- [Checkpointing](advanced/checkpointing.md)
- [Batch Dimension](advanced/batch.md)
- [AutoModels](advanced/auto.md)
- [Lexicon](advanced/lexicon.md)

## Developer Guide

- [Overview](developer/overview.md)
- [Dependencies](developer/dependency.md)
- [Style](developer/style.md)
- [Documentation](developer/documentation.md)
- [Testing](developer/testing.md)
- [Build](developer/build.md)
- [Recipes](developer/recipes.md)

## Support

- [Troubleshooting](support/troubleshooting.md)
- [Frequently Asked Questions](support/faq.md)
