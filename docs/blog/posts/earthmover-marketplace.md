---
date: 2026-10-05
readtime: 8
categories:
  - Documentation
tags:
  - Blog
  - Earthmover
  - Arraylake
  - Icechunk
  - ERA5
  - IFS
slug: earthmover-marketplace-data
---

# Access Earthmover Marketplace Data with Earth2Studio

Earth2Studio now provides data sources for reading ERA5 reanalysis and Brightband
ECMWF IFS data directly from the
[Earthmover Marketplace](https://app.earthmover.io/marketplace). The integrations
use Arraylake and Icechunk to select only the requested times and variables, without
requiring a custom download or parsing pipeline.

<!-- more -->

## Marketplace data in an Earth2Studio workflow

The Earthmover Marketplace provides versioned, cloud-native weather and climate
datasets as Arraylake repositories. Earth2Studio currently has dedicated integrations
for three listings:

| Earth2Studio data source | Dataset |
| --- | --- |
| [`EarthMoverERA5`](../../modules/datasources_analysis.md) | [ERA5 0.25° reanalysis](https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2) |
| [`EarthMoverBrightBandIFS`](../../modules/datasources_analysis.md) | [Brightband ECMWF IFS 0.25° initial conditions](https://app.earthmover.io/marketplace/697162921880507a6587c31b) |
| [`EarthMoverBrightBandIFS_FX`](../../modules/datasources_forecast.md) | [Brightband ECMWF IFS 0.1° 15-day forecast](https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04) |

This list covers Marketplace datasets with dedicated Earth2Studio classes. Other
Marketplace listings are not automatically supported by these integrations.

The underlying chunks remain in the provider's object store. Free listings use direct
subscriptions, while paid listings use filtered subscriptions whose metadata manifests
describe the chunks available to the subscriber. In either case, Arraylake reads data
lazily rather than downloading an entire dataset.

## Get started

### 1. Subscribe

Open the listing linked above, select **Subscribe**, and choose the Arraylake
organization that should receive the read-only subscription repository. A subscription
is required even for free listings.

Repository names vary by listing and do not map directly from the Marketplace URL or
Earth2Studio class name. Retain the provider's default name when possible, and verify
the resulting `organization/repository` path in Arraylake. For example:

- ERA5 commonly uses `<your-org>/era5-subscription`.
- Brightband IFS initial conditions commonly use
  `<your-org>/ecmwf-ifs-initial-conditions-open-subscription`.

### 2. Install and authenticate

Arraylake-backed sources require Python 3.12 through 3.14 and Earth2Studio's optional
data dependencies:

```bash
pip install earth2studio[data]
```

Create an API client in your Arraylake organization settings, then export its token and
your organization name:

```bash
export EARTHMOVER_API_KEY="<your-arraylake-api-key>"
export EARTHMOVER_ORGANIZATION="<your-org-name>"
```

Keep the token secret. `EARTHMOVER_ORGANIZATION` lets Earth2Studio derive the default
subscription repository; you can instead pass `repo="organization/repository"`
explicitly.

### 3. Read ERA5 data

With both environment variables configured, `EarthMoverERA5` can derive the repository
and authenticate automatically:

```python
from datetime import datetime

from earth2studio.data import EarthMoverERA5

source = EarthMoverERA5()
data = source(
    time=datetime(2021, 6, 1),
    variable=["t2m", "u10m", "z500"],
)
```

The result is an `xarray.DataArray` with `time`, `variable`, `lat`, and `lon`
dimensions, ready for postprocessing or use as a model initial state.

### 4. Read recent IFS data

The Brightband repositories retain a rolling 15-day window. Use a recent six-hour
cycle instead of a fixed historical timestamp:

```python
from datetime import UTC, datetime, timedelta

from earth2studio.data import EarthMoverBrightBandIFS

recent_cycle = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
recent_cycle = recent_cycle.replace(
    hour=6 * (recent_cycle.hour // 6),
    minute=0,
    second=0,
    microsecond=0,
)

source = EarthMoverBrightBandIFS(
    repo="my-org/ecmwf-ifs-initial-conditions-open-subscription"
)
data = source(
    time=recent_cycle,
    variable=["t2m", "z500", "u850"],
)
```

Forecast sources also accept lead times:

```python
import numpy as np

from earth2studio.data import EarthMoverBrightBandIFS_FX

source = EarthMoverBrightBandIFS_FX()
forecast = source(
    time=recent_cycle,
    lead_time=np.array([np.timedelta64(hour, "h") for hour in [0, 6, 12]]),
    variable=["t2m", "u10m"],
)
```

If a recent cycle is still being ingested, select an earlier cycle within the rolling
window. When calling a data source directly, pass `datetime` objects rather than raw
ISO strings.

## Bring your own authenticated client

Applications that already manage Arraylake clients can inject an authenticated
`AsyncClient`:

```python
import os

import arraylake
from earth2studio.data import EarthMoverERA5

client = arraylake.AsyncClient(token=os.environ["EARTHMOVER_API_KEY"])
source = EarthMoverERA5(
    repo="my-org/era5-subscription",
    client=client,
)
```

## Write Earth2Studio output to Arraylake

Earth2Studio's [`IceChunkBackend`](../../userguide/components/io.md) can write workflow
output to an Arraylake repository. The authenticated `arraylake.Client` obtains the
repository's credentialed `icechunk.Storage`; the backend receives that storage
object:

```python
import os

import arraylake
from earth2studio.io import IceChunkBackend

client = arraylake.Client(token=os.environ["EARTHMOVER_API_KEY"])
storage = client.get_icechunk_storage("my-org/my-output-repo")
io = IceChunkBackend(storage=storage, branch="main")
```

The repository must already exist. Create it with
`client.create_repo("my-org/my-output-repo")` or through the Arraylake web interface.
Earth2Studio workflows then use the backend's normal `add_array`, `write`, and
`commit` operations. For direct Xarray writes outside an Earth2Studio workflow, see
[`icechunk.xarray.to_icechunk`](https://icechunk.io/en/stable/reference/xarray/#icechunk.xarray.to_icechunk).

## Use custom Arraylake repositories

The integrations can also open compatible private repositories through an explicit
`repo` argument. The repository must use a regular one-dimensional latitude/longitude
grid, recognizable time coordinates, compatible variable metadata, and the group
layout expected by the selected class:

```python
# Reads the single/spatial and pressure/spatial groups.
era5 = EarthMoverERA5(repo="my-org/my-era5-repo")

# Reads the repository root.
ifs = EarthMoverBrightBandIFS(repo="my-org/my-ifs-repo")
```

Variable matching uses GRIB parameter identifiers, short names, CF variable names, and
CF standard names. Earth2Studio raises an actionable `ValueError` if a requested
variable cannot be resolved unambiguously.

## Learn more

- [Earthmover Marketplace](https://app.earthmover.io/marketplace)
- [Marketplace data-user documentation](https://docs.earthmover.io/marketplace/data-users)
- [Arraylake installation](https://docs.earthmover.io/setup/installation)
- [Earth2Studio data sources](../../userguide/components/datasources.md)
- [Earth2Studio IO backends](../../userguide/components/io.md)
