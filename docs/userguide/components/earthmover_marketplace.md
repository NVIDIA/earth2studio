# Earthmover Marketplace { #earthmover_marketplace_userguide }

The [Earthmover Marketplace](https://app.earthmover.io/marketplace) hosts analysis-ready
weather and climate datasets stored as [Arraylake](https://docs.earthmover.io/) /
[Icechunk](https://icechunk.io/) Zarr repositories.
Earth2Studio ships a set of data sources under `earth2studio.data` that read directly
from Marketplace repositories, so datasets can be used as initial conditions or
verification data without writing any custom download or parsing code.

!!! note
    Arraylake-backed Earthmover data sources require **Python 3.12 or newer** and the
    optional `arraylake` dependency, installed with:

    ```bash
    pip install earth2studio[data]
    ```

## Available Marketplace Data Sources

| Data source | Type | Dataset (hosted repo listing) |
| --- | --- | --- |
| [`earth2studio.data.EarthMoverERA5`](../../modules/datasources_analysis.md) | Analysis | [ERA5 0.25° reanalysis](https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2) |
| [`earth2studio.data.EarthMoverBrightBandIFS`](../../modules/datasources_analysis.md) | Analysis | [Brightband ECMWF IFS 0.25° initial conditions](https://app.earthmover.io/marketplace/697162921880507a6587c31b) |
| [`earth2studio.data.EarthMoverBrightBandIFS_FX`](../../modules/datasources_forecast.md) | Forecast | [Brightband ECMWF IFS 0.1° (10 km) 15-day forecast](https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04) |

Each dataset name links to its Marketplace listing page, which is where the Arraylake
repository is hosted and subscribed to — dataset coverage, license and the
**Subscribe** button all live there. Each data source class' docstring links to the
same page.

## How the data is hosted

Marketplace data is **not copied into Earthmover's or your own storage**. Each dataset
provider hosts their data in their own cloud object store (any S3-compatible store, GCS,
or Azure Blob), and [Icechunk](https://icechunk.io/) — the versioned storage format
Arraylake is built on — tracks it with cryptographically-addressed manifests.
Subscribing does not trigger a download:

- **Free listings** give you a direct read-only view: your repo points at the
  provider's object store and reads chunks straight from it, including full commit
  history.
- **Paid ("filtered") listings** store only metadata (which chunks you're entitled to)
  in your own organization's bucket; the chunk data itself is still read live from the
  provider's store, scoped to what your subscription covers.

In both cases, Icechunk's manifests prevent a subscriber from discovering or reading
chunks outside their subscription. Arraylake reads lazily rather than downloading whole
datasets, but (unlike Earth2Studio's other remote data sources) these classes do not
maintain a local on-disk cache — every call re-reads from the provider's object store.

## 1. Subscribe to a dataset

Marketplace datasets require an active subscription before they can be read, even for
open/free listings:

1. Open the dataset's listing page (linked from the data source docstring above, or
   browse [app.earthmover.io/marketplace](https://app.earthmover.io/marketplace)).
2. Click **Subscribe**, and choose which [Arraylake](https://docs.earthmover.io/) organization should house the
   resulting repository (you'll be prompted to create one if you don't have one yet).
   This creates a read-only repository under your organization, typically named
   `<org>/<dataset>-subscription`.
3. Note your **organization name** — it's used to derive the repository path below.

Free listings subscribe instantly and give a complete mirror of the provider's
repository, including full commit history. Paid listings require a
[Professional Arraylake plan](https://docs.earthmover.io/marketplace/faq) and a
request-access step before the subscription repository is created; they may also be
scoped ("filtered subscriptions") to specific variables, time ranges, or regions rather
than mirroring the full dataset.

Attempting to read a repository without an active subscription raises a
`PermissionError` pointing back to the listing page.

!!! note "License & cost"
    Each listing page states its own license and, for paid data, its pricing — set by
    the provider, not Earth2Studio. Storage for a subscription repo lives in the
    provider's bucket (free listings) or your own organization's bucket for the
    subscription metadata (paid listings); either way you are not billed for storing a
    copy of the underlying dataset. Check the listing page before subscribing to a
    paid dataset for the applicable terms.

## 2. Authenticate

Set an [Arraylake](https://docs.earthmover.io/) API key (create one at
[app.earthmover.io](https://app.earthmover.io) under account settings):

```bash
export EARTHMOVER_API_KEY="<your-arraylake-api-key>"
export EARTHMOVER_ORGANIZATION="<your-org-name>"
```

`EARTHMOVER_ORGANIZATION` is only needed if you want the data source to derive the
subscription repository name automatically (see below). Alternatively, pass a
pre-authenticated [`arraylake.AsyncClient`](https://docs.earthmover.io/reference/client)
directly to the data source via the `client` argument.

## 3. Use the data source

With the repo name derived from `EARTHMOVER_ORGANIZATION`:

```python
from earth2studio.data import EarthMoverERA5

ds = EarthMoverERA5()
da = ds(time="2021-06-01T00:00:00", variable=["t2m", "u10m", "z500"])
```

Or pass an explicit `org/repo` to read a specific repository:

```python
from earth2studio.data import EarthMoverBrightBandIFS

ds = EarthMoverBrightBandIFS(repo="my-org/ecmwf-ifs-initial-conditions-open-subscription")
da = ds(time="2024-01-01T00:00:00", variable=["t2m", "z500", "u850"])
```

Forecast sources additionally take a `lead_time`:

```python
import numpy as np
from earth2studio.data import EarthMoverBrightBandIFS_FX

ds = EarthMoverBrightBandIFS_FX()
da = ds(
    time="2024-01-01T00:00:00",
    lead_time=np.array([np.timedelta64(h, "h") for h in [0, 6, 12]]),
    variable=["t2m", "u10m"],
)
```

As with any [data source](datasources.md#datasources_userguide), the returned
`xr.DataArray` can be used directly for postprocessing or moved to the GPU as a model
initial state.

## Variable resolution

[Marketplace](https://app.earthmover.io/marketplace) repositories are not curated by Earth2Studio, so each Earthmover data
source resolves Earth2Studio variable ids (e.g. `t2m`, `z500`) against the repository's
native variable metadata at read time — matching on GRIB `paramId`, GRIB `shortName` /
`cfVarName`, or CF `standard_name`, in that priority order.
If a variable cannot be unambiguously resolved, the data source raises a `ValueError`
listing which variables are available in the repository, rather than silently
returning the wrong field.

## Custom or private Arraylake repositories

The Earthmover data sources are not limited to catalog listings from the
[Marketplace](https://app.earthmover.io/marketplace).
Any [Arraylake](https://docs.earthmover.io/) repository with a compatible CF/GRIB-annotated Zarr layout — including
your own private repositories — can be read by passing `repo="org/repo"` explicitly,
as shown above.

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `ValueError: Pass repo='org/repo' or set EARTHMOVER_ORGANIZATION ...` | No repo could be derived | Set `EARTHMOVER_ORGANIZATION` or pass `repo=` explicitly |
| `ValueError: Set EARTHMOVER_API_KEY ...` | No credentials found | Set `EARTHMOVER_API_KEY` or pass an authenticated `client=` |
| `PermissionError: Access to Arraylake repo '...' was denied` | No active subscription | Subscribe on the dataset's Marketplace listing page |
| `ValueError: Arraylake repo '...' was not found` | Wrong repo name, or subscription not yet active | Double check the `org/repo` name and subscription status |
| `ValueError: Could not resolve Earth2Studio variable '...' ` | Repository lacks GRIB/CF metadata for that variable | Check the error's list of available repository variables |

## Further reading

- [Earthmover Marketplace](https://app.earthmover.io/marketplace) — browse listings
- [Marketplace docs: Data Users](https://docs.earthmover.io/marketplace/data-users) — subscription mechanics
- [Marketplace docs: FAQ](https://docs.earthmover.io/marketplace/faq) — pricing and plan requirements
- [Arraylake client installation](https://docs.earthmover.io/setup/installation) — `arraylake` package setup
