(datasources_userguide)=

# Data Sources

Datasources are objects used to access data that is typically viewed as ground-truth
data used typically as an initial state for inference of an AI model.
But data sources can also be used for providing target data to evaluate the accuracy
of a particular model.
Data sources may be remote cloud based data stores or files on your local machine.
The list of datasources that are already built into Earth2studio can be found in
the API documentation {ref}`earth2studio.data`.

:::{note}
Data sources do not represent forecast systems / predictions such as numerical weather
simulators. They may include the initial states these simulators use or outputs from
data assimilation processes.
:::

## Data Source Interface

The full requirements for a standard diagnostic model are defined explicitly in the
`earth2studio/models/dx/base.py`.

```{literalinclude} ../../../earth2studio/data/base.py
:lines: 25-
:language: python
```

:::{note}
While not a requirement, built in remote data sources offer local caching when fetching
data which is stored in the Earth2Studio cache. See {ref}`configuration_userguide` for
details on how to customize this location.
:::

## Data Source Usage

The {func}`__call__` function is the way data is fetched from the data source and placed
into a in memory Xarray data array.
A user needs to provide both the time(s) and variables for the data source to fetch.
Variables can differ between data-sources and models.
The package lexicon is used as the source of truth and translator for data sources
discussed in more detail in the {ref}`lexicon_userguide` section.

This data array can then be used on the CPU for post process, saving to file, etc.
However, to use this as an initial state for inference with a model this Xarray data
array will need to get moved to the GPU and follow the standard data movement pattern
of Earth2Studio detailed in the {ref}`data_userguide` section.
There are a few utility functions inside Earth2Studio to make this process easy which
is commonly used in workflows.

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

The {func}`prep_data_array` is another useful utility when interacting more directly
with a data source.
This method will take a Xarray data array and return a tensor and coordinate system to
be used with other components.
Typically, this is used under the hood of various utils in Earth2Studio but may prove
useful to users implementing custom data sources where greater control is needed.

## Custom Data Sources

Custom data sources are often essential when working with large / on-prem
datasets.
So long as the data source can satisfy the API outlined in the interface above, it can
integrate seamlessly into Earth2Studio.
We recommend users have a look at the {ref}`extension_examples` examples, which will
step users through the simple process of implementing your own data source.

## Contributing a Datasource

We are always looking for new remote data stores that our users may be interested in for
running inference.
Its essential to make sure data sources can be accessed by all users and allow the
partial downloads of the data based on the users requests.
If you happen to manage a data source or have a data source in mind, open an issue on
the repo and we can discuss.
