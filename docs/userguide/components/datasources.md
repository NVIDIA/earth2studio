# Data and Forecast Sources { #datasources_userguide }

Datasources are objects that offer a simple API to access a "dataset" of weather/climate
data at a certain index.
Many implemented in the package provide access to data generated from numerical models,
data assimilation results or even generative AI models.
These typically serve as an initial state for inference of an AI model or some other
downstream task or target data to evaluate the accuracy of a particular model.
Data sources may be remote cloud-based data stores or files on your local machine.
The list of data sources that are already built into Earth2Studio can be found in
the API documentation [earth2studio.data.analysis](../../modules/datasources_analysis.md).

| Source type | Interface | Return type |
| --- | --- | --- |
| Data source | `(time, variable)` | `xr.DataArray` |
| Forecast source | `(time, lead_time, variable)` | `xr.DataArray` |
| DataFrame source | `(time, variable)` | `pd.DataFrame` |
| ForecastFrame source | `(time, lead_time, variable)` | `pd.DataFrame` |

!!! note
    Earth2Studio has data and forecast sources. The only difference being the latter
    has a lead time input. Some data stores may have both implemented where the data source
    provides the initial states and data-assimilated data, while the forecast source provides
    results from a predictive model.

## Data Source Interface

<!-- markdownlint-disable-next-line MD042 -->
[](){ #earth2studio.data.DataSource }

The full requirements for a standard diagnostic model are defined explicitly in the
`earth2studio/models/dx/base.py`.

```python
--8<-- "earth2studio/data/base.py:data-source-interface"
```

!!! note
    While not a requirement, built-in remote data sources offer local caching when fetching
    data which is stored in the Earth2Studio cache. Refer to
    [Configuration](../about/install.md#configuration_userguide) for details on
    how to customize this location.

### Beyond N-D Array Data

Earth2Studio attempts to stick with N-D array data structures when possible,
however this is not always possible or practical.
As a result, Earth2Studio also supports remote data sources for tabular data, which is
typically used when sparse observations or measurements are involved.

- `earth2studio.data.base.DataFrameSource`
- `earth2studio.data.base.ForecastFrameSource`

The call signatures mirror data/forecast sources; the difference is the return type: a
pandas DataFrame (tabular) instead of an Xarray DataArray.

## Data Source Usage

The `__call__` function is the way data is fetched from the data source and placed
into an in-memory Xarray data array.
You must provide both the times and variables for the data source to fetch.
Variables can differ between data-sources and models.
The package lexicon is used as the source of truth and translator for data sources
discussed in more detail in the [Lexicon](../advanced/lexicon.md#lexicon_userguide) section.

This data array can then be used on the CPU for postprocessing and saving to file.
However, to use this as an initial state for inference with a model this Xarray data
array will need to get moved to the GPU and follow the standard data movement pattern
of Earth2Studio detailed in the [Data Movement](../about/overview.md#data_userguide) section.
There are a few utility functions inside Earth2Studio to make this process easy.
These utility functions are commonly used in workflows.

!!! warning
    Each data source has its own methods for serving or calculating each variable.
    Users should be aware that the same variable across multiple data sources will
    potentially not be identical.
    Refer to each data source's documentation for details.

For async use cases some data/forecast sources support an async `fetch` function
that is available.
In these data sources, the `__call__` function is just a synchronous wrapper
around the async function.
The functionality is identical between the two.
Use the synchronous `__call__` for most workflows; use async `fetch` when batching
many requests or optimizing download performance.
Not all data sources have an async implementation, reference
[earth2studio.data.analysis](../../modules/datasources_analysis.md) for more
information.
Async-based data sources provide extremely fast download speeds compared to others,
so users should explore and test different ones if possible.

### `earth2studio.data.fetch_data`

`fetch_data` returns an Xarray **field DataArray** with dimensions
`[time, lead_time, variable, ...]`. Values are NumPy-backed on CPU and CuPy-backed
when `device="cuda:0"` is requested. Dimension labels, spatial auxiliary coordinates,
the array name, and source attributes travel with the values.

Use qualified variable labels to request temporal statistics.
Fields stay on the source grid unless a regridder is supplied (see below).

```python
import numpy as np

from earth2studio.data import fetch_data
field = fetch_data(
    source,
    np.array([np.datetime64("2024-01-02T00")]),
    np.array(["t2m:mean:24h"]),
    delta_t=np.timedelta64(6, "h"),
)
```

Here `source` is an analysis or forecast source providing instantaneous `t2m`.
The mean includes the four samples at -24, -18, -12 and -6 hours. Windows are
left-closed and right-open. The cadence defaults to `source.time_step` when
available; otherwise provide `delta_t` explicitly. Qualified variable labels
such as `"t2m:mean:24h"` declare reductions; duplicate normalized quantities are rejected.
Output `earth2studio_statistics` describes the reductions actually performed.
Already aggregated source variables cannot be reduced again.

`ARCO_ERA5.time_step` is `np.timedelta64(1, "h")`, so ARCO statistics requests
use hourly samples automatically without an explicit `delta_t`.

Analysis sources fetch absolute valid timestamps, while forecast sources reduce
lead times separately for each initialization. Missing or duplicate reduction
samples raise an error.

The former tuple return and `legacy` argument have been removed. Access values
through `field.data` and coordinates through `field.coords`.

#### Regridding

Pass `target_grid` to receive fields on another grid, typically a model's:

```python
from earth2studio.data import GFS, fetch_data
from earth2studio.grids import infer_grid

target = infer_grid(model.input_coords())
field = fetch_data(GFS(), time, variable, device="cuda", target_grid=target)
```

Fields already on the target pass through, and exact crops of the source grid are
selected. Otherwise `fetch_data` logs and regrids by nearest neighbor, and raises
if the target extends beyond the source.

Each of these calls builds a regridder, which computes its weights. To choose
another method, or to fetch repeatedly, build one once and pass it;
`list_regridders(source_grid, target)` lists the methods available:

```python
from earth2studio.grids import BilinearRegridder

regridder = BilinearRegridder("latlon-0.25deg", target)
field = fetch_data(GFS(), time, variable, device="cuda", regridder=regridder)
```

`fetch_data` reduces temporal statistics on the source grid, moves the field to
`device`, then regrids. `RegriddedSource(source, regridder)` packages a source and
regridder as a single source, which regrids every field it fetches. Models use it to
recommend sources on another grid. `CompositeSource` joins analysis sources that
provide different variables on a shared grid; `CompositeForecastSource` also joins
forecast sources, fetching analysis components at each lead's valid time.

#### Device

Data sources take a `device` keyword, defaulting to `"cpu"`. On a CUDA device they
return CuPy-backed fields directly on that GPU, which avoids slow NumPy
post-processing of large arrays. `fetch_data` accepts either and moves data only
when it is not already on the requested `device`.

### `earth2studio.data.fetch_dataframe` and `earth2studio.data.fetch_dataframes`

`fetch_dataframe` is the DataFrame counterpart of `fetch_data`: it fetches the
requested variables and fields from a DataFrame source and records the request time
in the frame's `attrs["request_time"]`, which data assimilation models read.
`fetch_dataframes` runs several such requests concurrently. Each request is a source
and a schema, for example an entry of a data assimilation model's `input_coords()`:
the schema's `variable` entry lists the variables and its other keys are the fields.
A list of `(source, schema)` pairs returns a list of frames in the same order, and a
dict of named pairs returns a dict.

```python
schemas = model.input_coords()
obs = fetch_dataframes(list(zip(sources, schemas)), time)
analysis = model(*obs)
```

### `earth2studio.data.prep_data_array`

The `prep_data_array` function is another useful utility when interacting more directly
with a data source.
This function takes an Xarray data array and returns a tensor and coordinate system to
be used with other components.
Typically, it is used as part of various utils in Earth2Studio, but may prove
useful to users implementing custom data sources where greater control is needed.

## Custom Data Sources

Custom data sources are often essential when working with large or on-prem
datasets.
So long as the data source can satisfy the API outlined in the interface above, it can
integrate seamlessly into Earth2Studio.
Accept a `device` keyword defaulting to `"cpu"`, store it as a `device` attribute,
and return CuPy-backed arrays on it when it is a CUDA device.
We recommend that you review the [extension examples](../../examples/index.md#extend)
examples, which will step you through the basic process of implementing your own
data source.

## Contributing a Datasource

We are always looking for new remote data stores that our users may be interested in for
running inference.
It's essential to make sure data sources can be accessed by all users and allow the
partial downloads of the data based on the users requests.
If you happen to manage a data source or have a data source in mind,
[open an issue](https://github.com/NVIDIA/earth2studio/issues) on the repo and we can discuss.
