# Earth2Studio

Earth2Studio is an open-source framework for exploring, building, and
deploying AI weather and climate workflows.

- **User Guide**: Learn the core concepts, installation flow, workflow
  patterns, and extension points. [Start with the user guide](userguide/index.md).
- **API Reference**: Browse generated API summaries with the same badge
  filters used by the previous Sphinx docs. [Open the API reference](modules/index.md).
- **Examples**: Browse MkDocs-native example pages rendered from the
  repository examples. [View examples](examples/index.md).
- **Recipes**: Browse larger workflows and integrations that live alongside
  the package source in the [recipes directory][recipes].

[recipes]: https://github.com/NVIDIA/earth2studio/tree/main/recipes

## Quick Start

```bash
pip install earth2studio
```

```python
from earth2studio.models.px import DLWP
from earth2studio.data import GFS
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic

model = DLWP.load_model(DLWP.load_default_package())
data = GFS()
io = NetCDF4Backend("forecast.nc")

deterministic(["2024-01-01"], 10, model, data, io)
```
