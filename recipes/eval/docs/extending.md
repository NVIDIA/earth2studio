# Extending the eval recipe

Three common extension points — bring-your-own data, plug in a different
model, and add a new runner, in that order from lightest to heaviest.
Each section is self-contained: if the first two are enough for your
case, you can stop there.

## Bring your own (BYO) data

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
and, for predictions, an optional `lead_time` dim). See
[src/data.py](../src/data.py):`PredownloadedSource` for a minimal
reference implementation.

With `ic_source`, `main.py` reads initial conditions directly and never touches
`data.zarr`. With `verification_source`, `score.py` and online scoring use
it in place of `verification.zarr`. `predownload.py` automatically skips
whichever side is BYO; with both, `main.py` skips the sentinel check
entirely and running `predownload.py` is unnecessary.

Every runner slot resolves the same way: a BYO node, then a pre-downloaded
`<role>.zarr`, then the live source, then the model's own
`default_sources()`. This chain is
[src/data.py](../src/data.py):`resolve_source(cfg, role, byo, live)`. A
custom runner builder should call it for each source it binds, so BYO and
pre-downloaded stores work the same way as for the built-ins.

## Bring your own model

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
Multi-step outputs, coupled components on different cadences (DLESyM)
and forcing slots (StormCast) are all handled by the default `forecast`
runner, which is the library's `PrognosticRunner`.

If your model needs a different iteration pattern (models coupled by
hand, outputs reduced before storage, sources the model protocol cannot
express), skip to the runner section below.

## Add a custom runner

A runner encapsulates "how a work item runs": binding data sources,
fetching, model iteration and the coordinate bookkeeping of what it
yields. Everything above one item belongs to `earth2studio.run.Pipeline`,
as [Pipeline responsibilities](#pipeline-responsibilities) lists. Write a
runner when:

* your rollout needs post-processing before storage (window means of every
  step; see [src/runners/seasonal.py](../src/runners/seasonal.py));
* you couple models yourself, with more than one source, and more than one
  output stream (GOES and MRMS; see
  [src/runners/stormscope.py](../src/runners/stormscope.py));
* your data requirements do not fit the default IC-plus-forcing pattern.

### Runner protocol

A runner is any object satisfying `earth2studio.run.Runner`:

```python
# my_pkg/my_runner.py
from collections.abc import Iterator, Mapping

import numpy as np
import torch
import xarray as xr

from earth2studio.run import WorkItem
from earth2studio.run.runner import DataRequest
from earth2studio.utils.type import CoordinateSystem


class MyRunner:
    supports_member_batching = False   # one member per item

    def __init__(self, model, source) -> None:
        self.model = model
        self.source = source
        self.source_roles = {"data": source}   # for predownload (optional)

    def to(self, device: torch.device) -> "MyRunner":
        self.model = self.model.to(device)
        return self

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        # One planning signature per stream: an allocation-free DataArray
        # whose dimensions are lead_time followed by the field dimensions.
        ...

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        # Every fetch run_item will make for this item; never fetches itself.
        return (DataRequest(self.source, np.array([item.time]), variables, leads),)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        # Yield, per step, one DataArray for each stream that published at it.
        ...
        yield {"forecast": x}
```

| Member | Contract |
| --- | --- |
| `supports_member_batching` | `True` if an item may carry more than one member |
| `to(device)` | Move the models to `device`; return `self` |
| `output_coords(final_lead_time)` | One signature per stream; `run_item` outputs must match it |
| `data_requests(item)` | Tuple of `DataRequest` for the gridded fetches, or `None` if unknown |
| `run_item(item)` | Iterator of `{stream: DataArray}` mappings, one per step |

A *stream* is a named sequence of outputs sharing variables, grid and
lead-time pattern. Give outputs with different grids or cadences
separate streams, keep stream names stable across runs (they key the
output store and resume), and declare every stream in `output_coords`.
The recipe writes every stream into the one `forecast.zarr` group and
splits `output.variables` over the streams carrying each name; a variable
no stream produces is an error.

Member batching: with `supports_member_batching = True` an item may carry
more than one member, and your outputs must carry a leading `ensemble`
dimension labelled by them (see
[src/runners/ensemble.py](../src/runners/ensemble.py):`EnsembleRunner`).
Seed per member with `earth2studio.run.work.member_seed(seed, item.time,
member)`. A batch cannot seed a stochastic model per member, so report
`supports_member_batching = False` for it, as `EnsembleRunner` does unless
the campaign sets `batch_stochastic_members: true`.

### Builder and registration

`cfg.runner` names a *builder*: a callable
`(cfg, prefer_predownloaded=True, **options) -> Runner`. `main.py`
calls it with `prefer_predownloaded=True`, `predownload.py` with
`False` so the runner binds live sources. Options come from the config
mapping:

```yaml
runner:
    _target_: my_pkg.my_runner.my_builder
    window: 30D          # passed as my_builder(cfg, prefer_predownloaded, window="30D")
```

```python
def my_builder(cfg, prefer_predownloaded: bool = True, window="30D") -> MyRunner:
    model = load_prognostic(cfg)                     # src.models
    source = resolve_source(                         # src.data
        cfg, "data", byo=cfg.get("ic_source"), live=cfg.get("data_source"),
        prefer_predownloaded=prefer_predownloaded,
    )
    return MyRunner(model, source)
```

The built-ins follow the same pattern;
[src/runners/\_\_init\_\_.py](../src/runners/__init__.py):`BUILTIN_RUNNERS`
lists them, and `runner: {name: window_mean, window: 30D}` is shorthand for
the `_target_` of a built-in. A builder that wraps another runner, as
`window_mean` wraps `forecast`, is the cheapest way to post-process a
rollout. `WindowMeanRunner` delegates `to`, `data_requests`, and
`supports_member_batching` to the inner runner. It rewrites the `lead_time`
axis of every inner stream in `output_coords` and averages the inner
`run_item` yields into one output per window.

### Pre-download and verification

A runner needs no pre-download hooks. `predownload.py` builds the runner
with live sources, collects `runner.data_requests(item)` over every work
item, and groups the requests by the source object they target. Each group
becomes one store `<output.path>/<role>.zarr` holding every
`time + lead_time` the requests cover. For a `ForecastSource` the store
instead holds every initial time and lead time, on
`(time, lead_time, <spatial...>)`, and reads back as a forecast source. The
role is the key under which the
runner lists that source in its optional `source_roles` dict
(`{"data": source, "forcing0": ...}`); a source the runner did not name
gets `source<k>`. At inference `resolve_source(cfg, role, ...)` finds the
store again by the same role name, which is why a builder should resolve
its sources through it. A runner whose `data_requests` returns `None`
caches nothing.

Runners that consume observation DataFrames also expose
`frame_requests(item) -> tuple[FrameRequest, ...]`
([src/data.py](../src/data.py):`FrameRequest`); `predownload.py` writes one
parquet file per analysis time under `<output.path>/<name>.parquet/`,
and `PredownloadedFrameSource` serves it back (see
[src/runners/assimilation.py](../src/runners/assimilation.py):`ObsSources`).

Verification is independent of the runner. With
`predownload.verification.enabled=true`, `predownload.py` always writes a
separate `<output.path>/verification.zarr`. It covers every initial time
plus every lead of the runner's `forecast` stream (or its first stream),
on the `output.regrid` grid, reached with its method, when the config names
one. `score.py` and online scoring resolve the verification source as the
BYO `verification_source`, then `verification.zarr`, then the live
`predownload.verification.source` or `data_source`.

### Pipeline responsibilities

`run_item` does not handle any of the following:

* distributing work items across ranks and marking them done for resume;
* filtering yielded fields to the configured `output.variables`;
* interpolating onto the `output.regrid` grid;
* seeding `(time, member)` consistently across ranks (use `member_seed`);
* writing zarr chunks or flushing.

See `earth2studio/run/pipeline.py`:`Pipeline.run` for the shared outer
loop and `dev/spec/EXECUTION_CONTRACT_SPEC.md` for the contract it
enforces. The spec settles where runner logic belongs.

### Testing

`test/test_main.py` runs `build_runner` and the full `main.run` on the
small-grid fixtures of `test/conftest.py` (`Persistence` prognostic,
`Random` source, `compose_cfg(tmp_path, overrides)` over the real
`cfg/default.yaml`); `test/test_runners.py` covers `EnsembleRunner`.
`test/test_seasonal.py` and `test/test_stormscope.py` test the two worked
examples on CPU with fake models, including their builders through
`build_runner`; copy their shape to start a new runner's test file.
