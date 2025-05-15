(lexicon_userguide)=

# Lexicon

As discussed in detail in the {ref}`data_userguide` section, Earth2Studio tracks the
geo-physical representation of tensor data inside workflows.
This includes the name of the variable / parameter / property the data represents, which
is tacked explicitly via Earth2Studios lexicon.
Similar to ECMWF's [parameter database](https://codes.ecmwf.int/grib/param-db/),
Earth2Studio's lexicon aims to provide an opinioned and explicit list of short variables
names that is used across the package found in {py:obj}`earth2studio.lexicon.base.E2STUDIO_VOCAB`.
Many of these names are based on ECMWF's parameter database but not all.

Below are a few examples:

- `t2m`: Temperature in Kelvin at 2 meters
- `u10m`: u-component of Zonal winds at 10 meters
- `v10m`: v-component of Zonal winds at 10 meters
- `u200`: u-component of Zonal winds at 200 hPa
- `z250`: Geo-potential at 250 hPa
- `z500`: Geo-potential at 500 hPa
- `tcwv`: Total column water vapor

:::{admonition} Altitude / Pressure Levels
:class: tip
Note that 3D atmospheric variables are sliced to their individual pressure levels.
This is better suited when working with various AI models that may use different
pressure levels.
Levels based on altitude contain an "m" at the end to distinguish height in meters.
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
    :lines: 24-44
    :language: python
```

Valuee of each variable is left up the the data source.
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
