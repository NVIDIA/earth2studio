# Extending the eval recipe

Three common extension points — bring-your-own data, plug in a different
model, and add a new pipeline — in that order from lightest to heaviest.
Each section is self-contained: if the first two are enough for your
case, you can stop there.

## Bring your own data (BYO)

If you already have initial-condition and/or verification data on disk,
you can bypass `predownload.py` entirely and point the recipe at a
custom `DataSource` implementation.

Set one or both overrides in your config:

```yaml
# cfg/default.yaml (or your campaign override)
ic_source:
    _target_: my_pkg.data.MyZarrSource
    store_path: /data/my_ics.zarr

verification_source:
    _target_: my_pkg.data.MyZarrSource
    store_path: /data/my_verification.zarr
```

A `DataSource` only needs to implement `__call__(time, variable) →
xr.DataArray` (returning data with dims `[time, variable, <spatial...>]`
and, for predictions, an optional `lead_time` dim).  See
[src/data.py](../src/data.py):`PredownloadedSource` for a minimal
reference implementation.

When `ic_source` is set, `main.py` reads ICs directly and never touches
`data.zarr`.  When `verification_source` is set, `score.py` uses it in
place of the local verification zarr.  Predownload automatically skips
whichever side is BYO; if both are set, the predownload sentinel check
is skipped entirely and running `predownload.py` is unnecessary.

For the single-source path (most pipelines), the resolution order is
**BYO → predownloaded zarr → live `data_source`**.  This chain lives in
[src/data.py](../src/data.py):`resolve_ic_source`; any custom pipeline
using a single IC source should call that helper from `setup` so BYO
works the same way.

## Plug in a different model

For prognostic or diagnostic models that already conform to the
[earth2studio](https://nvidia.github.io/earth2studio/) protocol (a
class with `load_default_package`/`load_model` classmethods plus the
`input_coords`/`output_coords`/`__call__` surface), the change is a
config edit only.

```yaml
# cfg/model/my_model.yaml
architecture: my_pkg.models.MyPrognostic   # must be importable at runtime
# Optional: point at a non-default weights bundle
# package_path: /path/to/weights
# Optional: extra kwargs forwarded to load_model
# load_args:
#     foo: bar
```

Select it from the command line or a campaign file:

```bash
python main.py model=my_model
```

The recipe drives your model via the standard earth2studio iterator
(`create_iterator` for prognostics; `__call__` for diagnostics), so no
Python changes are needed as long as the class honours that contract.

If your model has non-standard behavior (extra inputs, coupled model
components, a different iteration pattern) and you need to add
recipe-side logic to accommodate it, skip to the pipeline section
below — that's when writing a custom pipeline is the right call.

## Add a custom pipeline

Pipelines encapsulate "how a work item runs" — the orchestration between
data sources, model iteration, coordinate bookkeeping, and the output
store.  Subclass `src.pipelines.base.Pipeline` when:

* your model has a non-trivial iteration pattern (e.g. multiple output
  lead times per step, or coupled models advancing at different time
  scales — see [src/pipelines/dlesym.py](../src/pipelines/dlesym.py));
* you need more than one data source (e.g. IC + separate conditioning
  stream — see
  [src/pipelines/stormscope.py](../src/pipelines/stormscope.py));
* your predownload requirements don't fit the default IC-plus-optional-
  verification pattern.

### Minimum required surface

```python
# my_pkg/my_pipeline.py
from collections.abc import Iterator

import numpy as np
import torch
from omegaconf import DictConfig

from earth2studio.data import DataSource
from earth2studio.utils.coords import CoordSystem

from src.pipelines.base import Pipeline
from src.work import WorkItem


class MyPipeline(Pipeline):
    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        # Load models, cache coord metadata, set self._spatial_ref.
        ...

    def build_total_coords(
        self, times: np.ndarray, ensemble_size: int,
    ) -> CoordSystem:
        # Return the full output zarr schema.
        ...

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # Yield one or more (tensor, coords) pairs on the model's native grid.
        ...
```

Point your config at the new class:

```yaml
pipeline: my_pkg.my_pipeline.MyPipeline
```

Or with a `{_target_: ...}` block if you need to pass kwargs at
construction time.

### Optional hooks

The base class provides sensible defaults for everything else.  Override
these only when your pipeline needs to:

| Hook | Default | When to override |
|---|---|---|
| `predownload_stores(cfg)` | returns `[]` | declare IC / verification / conditioning zarrs for `predownload.py` — single-source pipelines can reuse [`src.predownload_utils.declare_single_source_stores`](../src/predownload_utils.py) |
| `verification_source(cfg)` | BYO → `verification.zarr` → `data.zarr` → `data_*.zarr` glob | you store verification in a non-standard layout |
| `verification_zarr_paths(cfg)` | the glob list used by the default `verification_source` | your verification stores live under different names (the report package also consults this) |
| `_inject_ensemble(...)` | prepends an `ensemble` axis | your model already carries ensemble along another dim |
| `needs_data_source` | `True` | your pipeline resolves its own sources inside `setup` — `main.py` skips the top-level `data_source` instantiation |
| `_run_item_includes_batch_dim` | `False` | your model output carries a size-1 `batch` axis that should be squeezed before output filtering |

### What the base class does for you

You do **not** need to handle any of the following in `run_item`:

* distributing work items across ranks (`main.py` does this);
* filtering yielded tensors to the configured `output.variables`;
* regridding to a configured output grid (when `self._output_regridder` is set);
* injecting the ensemble dimension at write time;
* writing zarr chunks or flushing resume markers.

See [src/pipelines/base.py](../src/pipelines/base.py):`Pipeline.run` for the
shared outer loop — that's the ground truth on ordering if you're
deciding where to put subclass logic.

### Testing

`test/test_pipeline.py` has a minimal `_StubPipeline` demonstrating the
ABC contract.  Copy its shape to start a new pipeline's test file; the
`build_pipeline` resolver supports both FQN strings and `_target_`
blocks (`test/test_pipeline.py::TestBuildPipeline` covers both paths).
