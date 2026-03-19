(datasources_userguide)=

# Data and Forecast Sources

Datasources are objects that offer a simple API to access a "dataset" of weather/climate
data at a certain index.
Many implemented in the package provide access to data generated from numerical models,
data assimilation results or even generative AI models.
These are typically as an initial state for inference of an AI model or some other
downstream task or target data to evaluate the accuracy of a particular model.
Data sources may be remote cloud based data stores or files on your local machine.
The list of data sources that are already built into Earth2Studio can be found in
the API documentation {ref}`earth2studio.data`.

:::{note}
Earth2Studio has data and forecast sources. The only difference being the latter
has a lead time input. Some data stores may have both implemented where the data source
provides the initial states and data-assimilated data, while the forecast source provides
results from a predictive model.
:::

## Data Source Interface

The full requirements for a standard diagnostic model are defined explicitly in the
`earth2studio/models/dx/base.py`.

```{literalinclude} ../../../earth2studio/data/base.py
:lines: 26-78
:language: python
```

:::{note}
While not a requirement, built in remote data sources offer local caching when fetching
data which is stored in the Earth2Studio cache. See {ref}`configuration_userguide` for
details on how to customize this location.
:::

### Beyond N-D Array Data

Earth2Studio attempts to stick with N-D array data structures when possible,
however this is not always possible or practical.
As a result, Earth2Studio also supports remote data sources for tabular data, which is
typically used when sparse observations or measurements are involved.

- {py:obj}`earth2studio.data.base.DataFrameSource`
- {py:obj}`earth2studio.data.base.ForecastFrameSource`

The call signatures mirror data/forecast sources; the difference is the return type: a
pandas DataFrame (tabular) instead of an Xarray DataArray.

## Data Source Usage

The {func}`__call__` function is the way data is fetched from the data source and placed
into an in-memory Xarray data array.
You must provide both the times and variables for the data source to fetch.
Variables can differ between data-sources and models.
The package lexicon is used as the source of truth and translator for data sources
discussed in more detail in the {ref}`lexicon_userguide` section.

This data array can then be used on the CPU for post process and saving to file.
However, to use this as an initial state for inference with a model this Xarray data
array will need to get moved to the GPU and follow the standard data movement pattern
of Earth2Studio detailed in the {ref}`data_userguide` section.
There are a few utility functions inside Earth2Studio to make this process easy.
These utility functions are commonly used in workflows.

:::{warning}
Each data source has its own methods for serving or calculating each variable.
Users should be aware that the same variable across multiple data sources will
potentially not be identical.
Refer to each data source's documentation for details.
:::

For async use cases some data/forecast sources support an async {func}`fetch` function
that is available.
In these data sources, the {func}`__call__` function is just a synchronous wrapper
around the async function.
The functionality is identical between the two.
Not all data sources have an async implementation, reference {ref}`earth2studio.data`
for more information.
Async-based data sources provide extremely fast download speeds compared to others,
so users should explore and test different ones if possible.

### {mod}`earth2studio.data.fetch_data`

The {func}`fetch_data` function is useful for getting a PyTorch tensor and
coordinate system for a given model.
This utility fetches data for an array of times and lead times for the specified
variables.
For example, in the deterministic workflow {mod}`earth2studio.run.deterministic`, it is
used to get the initial state for the provided prognostic.

```{literalinclude} ../../../earth2studio/run.py
:language: python
:start-after: sphinx - fetch data start
:end-before: sphinx - fetch data end
```

### {mod}`earth2studio.data.prep_data_array`

The {func}`prep_data_array` function is another useful utility when interacting more directly
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
We recommend that you review the {ref}`extension_examples` examples, which will
step you through the basic process of implementing your own data source.

## Contributing a Datasource

We are always looking for new remote data stores that our users may be interested in for
running inference.
It's essential to make sure data sources can be accessed by all users and allow the
partial downloads of the data based on the users requests.
If you happen to manage a data source or have a data source in mind, open an issue on
the repo and we can discuss.
