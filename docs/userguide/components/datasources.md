# Data and Forecast Sources { #datasources_userguide }

Sources fetch weather and climate data from remote services or local files. They
provide initial conditions, conditioning fields, observations, and verification
targets. Browse the [analysis](../../modules/datasources_analysis.md),
[forecast](../../modules/datasources_forecast.md), and
[tabular](../../modules/datasources_dataframe.md) source APIs.

## Interface

<!-- markdownlint-disable-next-line MD042 -->
[](){ #earth2studio.data.DataSource }

| Source | Call | Output |
| --- | --- | --- |
| `DataSource` | `source(time, variable)` | DataArray: `time, variable, ...` |
| `ForecastSource` | `source(time, lead_time, variable)` | Forecast DataArray |
| `DataFrameSource` | `source(time, variable, fields=None)` | Observation DataFrame |
| `ForecastFrameSource` | `source(time, lead_time, variable, fields=None)` | DataFrame |

Forecast DataArrays have dimensions `time, lead_time, variable, ...`.
Forecast `time` is the initialization time; `lead_time` selects offsets from that
initialization. Analysis `time` selects valid times directly. Tabular sources expose
`SCHEMA`; `fields` selects observation columns. The protocols are in
`earth2studio.data.base`. Some sources also offer an async `fetch` with equivalent
arguments and results.

### Ensemble Members

<!-- markdownlint-disable-next-line MD042 -->
[](){ #earth2studio.data.EnsembleDataSource }

`EnsembleDataSource` and `EnsembleForecastSource` add a `member` keyword. It selects
source-native member labels and returns a leading `ensemble` dimension, including
for a single requested member. Omitting it returns the source's configured member
without adding that dimension.

See [E2S-05: Data Fetching][fetch-spec] for request handling, output assembly,
ensemble selection, and device-placement requirements.

[fetch-spec]: ../../specs/E2S-05_FETCH_DATA_SPEC.md

## Usage

### Fetch Gridded Fields

`fetch_data` assembles a field with dimensions `time, lead_time, variable, ...`,
optionally preceded by `ensemble`. It accepts both analysis and forecast sources:
analysis sources are queried at `time + lead_time`, while forecast sources retain
their initialization cycle. Floating values are converted to float32.

Choose variable names and physical units using the source's
[lexicon](../advanced/lexicon.md#lexicon_userguide). Set `device` on the source
constructor to choose where it returns data; `fetch_data` can then move gridded
fields to the model's execution device.

```python
import numpy as np

from earth2studio.data import GFS, fetch_data

times = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
variables = np.array(["t2m", "u10m", "v10m"])
field = fetch_data(
    GFS(), times, variables,
    lead_time=np.array([0], dtype="timedelta64[h]"),
    device="cpu",
)
```

Use nonempty, one-dimensional datetime and timedelta arrays with finite labels.
The result is a field DataArray: access values through `.data` and coordinates
through `.coords`. Remote sources commonly cache downloads; see
[Configuration](../about/install.md#configuration_userguide).

### Match a Model Grid

For a single-input model, request its declared variables, history, and geometry:

```python
from earth2studio.grids import infer_grid

signature = model.input_coords()
x = fetch_data(
    GFS(), times, signature.coords["variable"].values,
    lead_time=signature.coords["lead_time"].values,
    target_grid=infer_grid(signature),
    device="cuda:0",
)
```

Matching grids pass through; exact crops are selected. Otherwise `target_grid`
uses nearest-neighbor regridding and rejects targets outside source coverage.
For repeated requests or another method, build a compatible regridder once and
pass `regridder=...`; `list_regridders(source_grid, target_grid)` lists available
methods. Temporal reductions happen on the source grid, followed by device
placement and regridding.

`RegriddedSource` combines a source and reusable regridder. `CompositeSource` joins
analysis sources supplying different variables on a shared grid;
`CompositeForecastSource` also supports forecast sources.

### Temporal Statistics

For a source supporting the requested statistic:

```python
field = fetch_data(source, times, np.array(["t2m:mean:24h"]))
```

Windows are left-open and right-closed: a six-hourly source includes samples at
`T-18h`, `T-12h`, `T-6h`, and `T` for a 24-hour mean ending at `T`. NaN samples
propagate. Forecast windows must stay within the requested initialization's
available leads. Summing stored accumulations with overlapping windows can
double-count totals; check the source's temporal-statistics documentation.
The output's `earth2studio_statistics` metadata describes the reduction performed.
See [Temporal Statistics](../advanced/temporal_statistics.md) for label syntax,
supported sources, calendar-day windows, and local-array reductions.

### Tabular Observations

`fetch_dataframe` fetches requested variables and fields and records
`attrs["request_time"]` for assimilation models. `fetch_dataframes` makes multiple
requests concurrently. For a model whose inputs are all observation schemas:

```python
from earth2studio.data import fetch_dataframes

schemas = model.input_coords()
observations = fetch_dataframes(list(zip(sources, schemas)), times)
analysis = model(*observations)
```

Each `(source, schema)` pair uses the schema's `variable` entry for variables and
other keys for fields. A list returns frames in the same order; a dictionary of
named requests returns a dictionary. Prepare mixed gridded/tabular slots separately.

## Create Your Own Source

Wrap a local dataset or service in a callable that follows the appropriate source
interface. Translate variable names and units, preserve request order and grid
metadata, and report unsupported times or variables explicitly. Accept `device`
in the constructor and expose the configured device on the instance.

The [earth2studio-create-datasource skill][datasource-skill] provides templates for
all four source types. Ask your agent to create a wrapper in your project, using
sample files or service responses to verify indexing, units, coordinates, and
device placement. See the [extension examples](../../examples/index.md#extend).

[datasource-skill]: https://github.com/NVIDIA/earth2studio/tree/1.0.0-rc/skills/earth2studio-create-datasource
