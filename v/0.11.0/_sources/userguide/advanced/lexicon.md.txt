(lexicon_userguide)=

# Lexicon

As discussed in detail in the {ref}`data_userguide` section, Earth2Studio tracks the
geo-physical representation of tensor data inside workflows.
This includes the name of the variable / parameter / property the data represents, which
is tracked explicitly via Earth2Studios lexicon.
Similar to ECMWF's [parameter database](https://codes.ecmwf.int/grib/param-db/),
Earth2Studio's lexicon aims to provide an opinioned and explicit list of short variables
names that is used across the package found in {py:obj}`earth2studio.lexicon.base.E2STUDIO_VOCAB`.
Many of these names are based on ECMWF's parameter database but not all.

Below are a few examples:

- `t2m`: Temperature in Kelvin at 2 meters
- `u10m`: u-component (eastward/zonal) of winds at 10 meters
- `v10m`: v-component (northward/meridional) of winds at 10 meters
- `u200`: u-component of winds at 200 hPa
- `z250`: Geo-potential at 250 hPa
- `z500`: Geo-potential at 500 hPa
- `tcwv`: Total column water vapor

Additionally, the lexicon may also be used to track various metadata / coordinate
fields.
This is particularly relevant for {py:obj}`earth2studio.data.base.DataFrameSource` where
tabular data is used and variables map to the columns of a data frame.
Earth2Studio takes a best-effort approach to make these metadata fields standardized
across the package.

Some examples include:

- `lat`: Latitude coordinate of data / observation / sensor
- `lon`: Longitude coordinate of data / observation / sensor
- `elev`: Elevation (meters) relative to mean sea level of data / observation / sensor

## Altitude / Pressure Levels

Note that there are a variety of ways to represent the vertical coordinates for 3D
atmospheric variables. The most common method is to slice variables along pressure
levels (surfaces of constant pressure), and this is considered the "default" in terms
of variable names within the lexicon (e.g., `z500` is the geo-potential) at the 500 hPa
pressure level. Variables which are represented based on altitude above the surface
contain an "m" at the end, to denote height in meters, such as `u10m`.

Some models or workflows, however, require using their own custom vertical coordinate
which is neither pressure-level nor terrain-following. These are typically referred to
as "native" or "hybrid" vertical levels, and are defined differently for different
use-cases. The lexicon supports these custom levels by indexing the vertical level and
appending a suffix to the variable name to denote it is a custom vertical level, as in
`u100k` to indicate the u-component of winds at the custom vertical level with index
100 (indexed by `k`). We leave the choice of suffix up to each use-case, and reserve
the following special-case suffixes:

- No suffix: assumed to be pressure-level, as in `z500` for geo-potential at 500 hPa level
- `m`: altitude in meters above the surface

:::{warning}
Only use custom vertical level data with caution. The definition of these vertical
levels changes with each data source, model, or use-case, and thus they are not
necessarily interoperable. Transforming between different custom vertical levels will
likely require custom interpolation schemes, possibly using pressure-levels as an
intermediate step.
:::

## Datasource Lexicon

A common challenge when working with different sources of weather/climate data is that
variables used may be named / denoted in different ways.
The Lexicon is also used to track the translation between Earth2Studios naming scheme
and the scheme needed to parse the remote data source.
Each remote data store has its own lexicon, which is a dictionary that has the
Earth2Studio variable name as the keys and a string used to parse the remote data store.
Typically, this value is a string that corresponds to the variable name inside the remote
data store.

The following snippet is part of the lexicon for the GFS dataset.
Note that the class has a `metaclass=LexiconType` which is present in
{py:mod}`earth2studio.lexicon.base.py` used for type checking.

```{literalinclude} ../../../earth2studio/lexicon/gfs.py
    :lines: 24-60
    :language: python
```

Values of each variable is left up the the data source.
The present pattern is to split by the string based on the separator `::`, and then used
to access the required data.
For example, the variable `u100`, zonal winds at 100 hPa, the value `UGRD::100 mb` is
split into `UGRD` and `100 mb` which are then used with the remote Grib index file to
fetch the correct data.

```{literalinclude} ../../../earth2studio/data/gfs.py
    :start-after: "# sphinx - lexicon start"
    :end-before: "# sphinx - lexicon end"
    :language: python
```

It is a common pattern for data source lexicons to contain a modifier function that is
used to apply adjustments to align data more uniformly with the package.
A good example of this is the GFS dataset which uses the modifier function to transform
the GFS supplied the geo-potential height to geo-potential to better align with other
sources inside Earth2Studio.

```{literalinclude} ../../../earth2studio/lexicon/gfs.py
    :start-after: "# sphinx - modifier start"
    :end-before: "# sphinx - modifier end"
    :language: python
```

:::{warning}
The lexicon does not necessarily contain every variable inside the remote data store.
Rather it explicitly lists what is available inside Earth2Studio. See some variable
missing you would like to add? Open an issue!
:::
