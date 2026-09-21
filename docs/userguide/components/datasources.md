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

Pass a coordinate signature as `metadata` to declare temporal statistics.
Spatial regridding is currently a pass-through: the result retains the source's
grid, spatial coordinates, and grid attributes. `interp_to`, `interp_method`,
`bounds`, and `bounds_crs` are reserved for future spatial processing and currently
have no effect. Fetch does not reconstruct or validate grid metadata.

```python
import numpy as np

from earth2studio.data import fetch_data
from earth2studio.utils.coords import coord_array

target = coord_array(
    ("time", "lead_time", "variable"),
    {"lead_time": [np.timedelta64(0, "h")], "variable": ["t2m:mean:24h"]},
    dynamic=("time",),
)
field = fetch_data(
    source,
    np.array([np.datetime64("2024-01-02T00")]),
    target.coords["variable"].values,
    metadata=target,
    delta_t=np.timedelta64(6, "h"),
)
```

Here `source` is an analysis or forecast source providing instantaneous `t2m`.
The mean includes the four samples at -24, -18, -12 and -6 hours. Windows are
left-closed and right-open. The cadence defaults to `source.time_step` when
available; otherwise provide `delta_t` explicitly. Qualified variable labels
such as `"t2m:mean:24h"` declare reductions, and coordinate signatures derive their
statistics metadata from those labels. Metadata and qualified labels must agree;
duplicate normalized quantities are rejected.
Output `earth2studio_statistics` describes the reductions actually performed.
Already aggregated source variables cannot be reduced again.

`ARCO_ERA5.time_step` is `np.timedelta64(1, "h")`, so ARCO statistics requests
use hourly samples automatically without an explicit `delta_t`.

Analysis sources fetch absolute valid timestamps, while forecast sources reduce
lead times separately for each initialization. Missing or duplicate reduction
samples raise an error.

#### Calendar-day means (FuXi-S2S)

FuXi-S2S labels each daily mean by its **starting midnight**. Use explicit forward
windows with hourly source cadence:

```python
from earth2studio.models.px.fuxi_s2s import DAILY_VARIABLES

field = fetch_data(
    source,
    np.array([np.datetime64("2024-01-02T00")]),
    np.array(DAILY_VARIABLES),
    lead_time=np.array([-24, 0], dtype="timedelta64[h]"),
    delta_t=np.timedelta64(1, "h"),
)
```

Ordinary fields such as `t2m:mean:0h:24h` average samples at 00–23 UTC.
The interval-ending sources `tp:mean:1h:25h` and `ttr:mean:1h:25h` average
one-hour accumulations ending at 01 UTC through the next midnight. These are
means, not daily totals. The two requested leads retain their start-of-day labels
and full qualified variable names. `mean:24h` instead selects the preceding day;
it is not interchangeable with these forward windows.

Variables sharing a normalized window are fetched as one block using their base
names. The full labels and normalized statistics metadata are restored after
reduction. Already-prepared daily means should be labeled directly and passed to
the model without applying these reductions again.

The former tuple return and `legacy` argument have been removed. Access values
through `field.data` and coordinates through `field.coords`.

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
