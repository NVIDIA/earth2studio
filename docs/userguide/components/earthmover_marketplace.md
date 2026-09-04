<!-- markdownlint-disable MD013 MD033 MD046 -->

# Earthmover Marketplace { #earthmover_marketplace_userguide }

The [Earthmover Marketplace](https://app.earthmover.io/marketplace) is Earthmover's
["modern, cloud-native data sharing experience for high-velocity gridded weather,
climate, and geospatial datasets"](https://docs.earthmover.io/marketplace/data-users),
stored as [Arraylake](https://docs.earthmover.io/) / [Icechunk](https://icechunk.io/)
Zarr repositories. Earth2Studio ships a set of data sources under `earth2studio.data`
that read directly from Marketplace repositories, so datasets can be used as initial
conditions or verification data without writing any custom download or parsing code.

!!! note
    Arraylake-backed Earthmover data sources require **Python 3.12 or newer** and the
    optional `arraylake` dependency, installed with:

    ```bash
    pip install earth2studio[data]
    ```

## Available Marketplace Data Sources

| Data source | Type | Provider | Dataset (hosted repo listing) |
| --- | --- | --- | --- |
| [`earth2studio.data.EarthMoverERA5`](../../modules/datasources_analysis.md) | Analysis | `earthmover-public` | [ERA5 0.25° reanalysis](https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2) |
| [`earth2studio.data.EarthMoverBrightBandIFS`](../../modules/datasources_analysis.md) | Analysis | `brightband` | [Brightband ECMWF IFS 0.25° initial conditions](https://app.earthmover.io/marketplace/697162921880507a6587c31b) |
| [`earth2studio.data.EarthMoverBrightBandIFS_FX`](../../modules/datasources_forecast.md) | Forecast | `brightband` | [Brightband ECMWF IFS 0.1° (10 km) 15-day forecast](https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04) |

Each dataset name links to its Marketplace listing page, which is where the Arraylake
repository is hosted and subscribed to. Dataset coverage, license and the
**Subscribe** button all live there. Each data source class' docstring links to the
same page. Per the listing pages themselves:

<div class="grid cards" markdown>

- [![ERA5 listing](https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2/opengraph-image-9dzfy2?b1f408a208b63cfd)](https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2)

    "ECMWF ERA5 hourly single and pressure levels starting 1940-01-01, updated every
    three months."

- [![ECMWF IFS Initial Conditions (open) listing](https://app.earthmover.io/marketplace/697162921880507a6587c31b/opengraph-image-9dzfy2?b1f408a208b63cfd)](https://app.earthmover.io/marketplace/697162921880507a6587c31b)

    "Dataset containing variables from the ECMWF IFS atmospheric models necessary for
    initializing MLWP models, available 4x daily."

- [![ECMWF IFS 15-day Forecast (open) listing](https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04/opengraph-image-9dzfy2?b1f408a208b63cfd)](https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04)

    "ECMWF IFS 15-day forecast surface fields, available before ECMWF Open Data."

</div>

## How the data is hosted

The underlying chunk data is **never copied into Earthmover's or your own storage**
(a paid subscription's own bucket holds only metadata about which chunks you can
access, not the data itself - see below). Per
[Earthmover's storage docs](https://docs.earthmover.io/concepts/storage), "Arraylake
works with a wide range of commercial and open-source object storage services,
including any S3-compatible object store as well as Google Cloud Storage and Microsoft
Azure Blob Storage," and providers typically use "BYOB - Bring your own bucket": "all
the data live in your cloud in your own object storage bucket." [Icechunk](https://icechunk.io/)
(the versioned storage format Arraylake is built on) tracks it with
cryptographically-addressed manifests. Subscribing does not trigger a download:

- **Free listings** are a direct subscription. Per
  [Earthmover's provider docs](https://docs.earthmover.io/marketplace/data-providers),
  "subscribers read data directly from [the provider's] object store" and "see your
  full commit history and can access any version" - no data is copied, your repo just
  points at the provider's storage.
- **Paid listings** are "filtered subscriptions": your repo stores only metadata (which
  chunks you're entitled to) in your own organization's bucket, while "the actual chunk
  data is read from [the provider's] object store" - scoped to what your subscription
  covers.

Either way, "due to Icechunk's cryptographically random keys, it is not possible for
the subscriber to discover any data not explicitly included in their manifests."
Arraylake reads lazily rather than downloading whole datasets. Unlike Earth2Studio's
other remote data sources, these classes do not use Earth2Studio's own on-disk cache
(the `cache` constructor argument is accepted for API compatibility but unused); any
caching beyond that is internal to the `arraylake`/Icechunk client and not something
Earth2Studio controls or has verified.

## 1. Subscribe to a dataset

Marketplace datasets require an active subscription before they can be read, even for
open/free listings:

Per [Earthmover's own docs](https://docs.earthmover.io/marketplace/data-users):

1. "Browse the Marketplace - Find a dataset you're interested in at
   app.earthmover.io/marketplace" (or follow the link from the data source docstring
   above).
2. "Subscribe - Click the subscribe button on the listing page."
3. "Select an Organization - Choose which Arraylake organization should house the
   resulting repo" (you'll be prompted to create one if you don't have one yet). Note
   this **organization name**, used to derive the repository path below.

"When you subscribe to a dataset, a read-only repo appears in your Arraylake
organization." The repo name is set by the provider and varies per listing - e.g.
`<org>/era5-subscription` for ERA5, or
`<org>/ecmwf-ifs-initial-conditions-open-subscription` for Brightband's IFS initial
conditions - so check the listing page or the data source's docstring for the exact
name.

"Many datasets on the Marketplace are freely available. Anyone with an Arraylake
account can subscribe to free listings instantly." Free listings "use direct
subscriptions" - "your repo is a complete mirror of the provider's repo," including
full commit history. Paid listings are premium offerings that "use filtered
subscriptions": "instead of mirroring the provider's entire repo, you receive a repo
scoped to specific variables, time ranges, or spatial regions," and require a
[Professional Arraylake plan](https://docs.earthmover.io/marketplace/faq).

Attempting to read a repository without an active subscription raises a
`PermissionError` pointing back to the listing page.

!!! note
    "Subscriptions are not anonymous. When you subscribe to a dataset, the data
    provider can see your organization name and contact email."

!!! note "License & cost"
    Per [Earthmover's FAQ](https://docs.earthmover.io/marketplace/faq): "The
    Marketplace supports both free/open data and paid subscriptions." For free
    listings, "no payment needed to access the data"; "many datasets on the
    Marketplace are completely free" and "anyone with an Arraylake account can
    subscribe to free listings instantly and start querying data right away." For
    paid listings, "custom pricing [is] negotiated between provider and the user," and
    "paid datasets require a Professional plan." Each listing page states its own
    license terms, set by the provider, not Earth2Studio - check it before
    subscribing.

## 2. Authenticate

Set an [Arraylake](https://docs.earthmover.io/) API key. Per
[Earthmover's org-access docs](https://docs.earthmover.io/setup/org-access), "to create
a new API key, click on the big purple 'New API Client' button" in your organization's
settings on [app.earthmover.io](https://app.earthmover.io), "enter a name for the API
key, then select the appropriate permissions (read/write), and the key's lifetime."
The resulting "secret tokens are a single string, prefixed with the `ema_` identifier"
(e.g. `ema_123456789123456789_123456789123456789123456789`) and "expire after 1 year by
default." Tokens "should be considered secret, and should not be shared publicly":

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
from datetime import datetime
from earth2studio.data import EarthMoverERA5

ds = EarthMoverERA5()
da = ds(time=datetime(2021, 6, 1), variable=["t2m", "u10m", "z500"])
```

Or pass an explicit `org/repo` to read a specific repository:

```python
from datetime import datetime
from earth2studio.data import EarthMoverBrightBandIFS

ds = EarthMoverBrightBandIFS(repo="my-org/ecmwf-ifs-initial-conditions-open-subscription")
da = ds(time=datetime(2024, 1, 1), variable=["t2m", "z500", "u850"])
```

Forecast sources additionally take a `lead_time`:

```python
from datetime import datetime

import numpy as np
from earth2studio.data import EarthMoverBrightBandIFS_FX

ds = EarthMoverBrightBandIFS_FX()
da = ds(
    time=datetime(2024, 1, 1),
    lead_time=np.array([np.timedelta64(h, "h") for h in [0, 6, 12]]),
    variable=["t2m", "u10m"],
)
```

!!! note
    Pass `datetime` objects (or a list/array of them) directly to a data source, as
    shown above; passing a raw ISO string is not supported. Workflow entry points such
    as `earth2studio.run.deterministic` convert `list[str]` time inputs for you via
    `earth2studio.utils.time.to_time_array`, but calling a data source directly does not.

As with any [data source](datasources.md#datasources_userguide), the returned
`xr.DataArray` can be used directly for postprocessing or moved to the GPU as a model
initial state.

## Writing output to Arraylake

Earth2Studio's [`IceChunkBackend`](io.md) IO backend is not integrated with Arraylake:
it only accepts a plain `icechunk.Storage | str | None` and always calls
`icechunk.Repository.open_or_create()` itself, so there is no supported way to point it
at an Arraylake-managed org/repo directly.

To write inference output (or any data) into an Arraylake repository, use the
`arraylake` client directly instead of an Earth2Studio IO backend:

```python
import arraylake as al
import zarr

client = al.Client()
repo = client.create_repo("your-org/your-repo")  # or client.get_repo(...)

session = repo.writable_session("main")
root = zarr.group(session.store)
# write with normal zarr/xarray operations against session.store
session.commit(message="Add data")
```

See [Earthmover's version control guide](https://docs.earthmover.io/guide/version-control)
for the full writable-session and commit workflow.

### Publishing your own Marketplace listing

Writing to your own repo is enough to use it as a `repo="org/repo"` data source (see
[Custom or private Arraylake repositories](#custom-or-private-arraylake-repositories)
below), but publishing it as a Marketplace listing so others can subscribe is a
separate, Earthmover-account-level process, per
[Earthmover's provider docs](https://docs.earthmover.io/marketplace/data-providers):

1. Create an organization at
   [app.earthmover.io/orgs/new](https://app.earthmover.io/orgs/new) - "the public face
   for your data."
2. Upgrade to the Professional tier by emailing `support@earthmover.io`; this is
   required to become a provider.
3. "Configure a storage bucket using Credential Vending in your cloud provider of
   choice" (bring-your-own-bucket), or request Earthmover-managed storage from
   support.
4. Prepare the data: "create a new Icechunk repo," import existing Icechunk data, or
   write it with Xarray, using the `writable_session` / `commit` pattern above.
5. In your org settings, open the **Marketplace** tab and click **"+ Create
   Listing"**, then fill in the repository, listing name and description, thumbnail
   URL, README, license terms, and pricing model.
6. Choose a pricing model: for free listings, "subscribers get access to everything in
   the repo"; for paid listings, you "select exactly which variables and groups are
   available to subscribers."
7. Set the listing status to **Published** (requires a repository attached) - it can
   otherwise be left **Unpublished** or **Coming Soon**.

## Variable resolution

[Marketplace](https://app.earthmover.io/marketplace) repositories are not curated by Earth2Studio, so each Earthmover data
source resolves Earth2Studio variable ids (e.g. `t2m`, `z500`) against the repository's
native variable metadata at read time, matching on GRIB `paramId`, GRIB `shortName` /
`cfVarName`, or CF `standard_name`, in that priority order.
If a variable cannot be unambiguously resolved, the data source raises a `ValueError`
listing which variables are available in the repository, rather than silently
returning the wrong field.

## Custom or private Arraylake repositories

The Earthmover data sources are not limited to catalog listings from the
[Marketplace](https://app.earthmover.io/marketplace).
Any [Arraylake](https://docs.earthmover.io/) repository with a compatible CF/GRIB-annotated Zarr layout,
including your own private repositories, can be read by passing `repo="org/repo"` explicitly,
as shown above.

!!! warning
    Each class also fixes which Zarr group(s) it opens, so a custom repo must match
    that layout: `EarthMoverERA5` always opens the `single/spatial` and
    `pressure/spatial` groups, while `EarthMoverBrightBandIFS` and
    `EarthMoverBrightBandIFS_FX` open the repository's root group. A custom repo with a
    different group layout will fail to connect even if its variable metadata is
    otherwise compatible.

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `ValueError: Pass repo='org/repo' or set EARTHMOVER_ORGANIZATION ...` | No repo could be derived | Set `EARTHMOVER_ORGANIZATION` or pass `repo=` explicitly |
| `ValueError: Set EARTHMOVER_API_KEY ...` | No credentials found | Set `EARTHMOVER_API_KEY` or pass an authenticated `client=` |
| `PermissionError: Access to Arraylake repo '...' was denied` | No active subscription | Subscribe on the dataset's Marketplace listing page |
| `ValueError: Arraylake repo '...' was not found` | Wrong repo name, or subscription not yet active | Double check the `org/repo` name and subscription status |
| `ValueError: Could not resolve Earth2Studio variable '...'` | Repository lacks GRIB/CF metadata for that variable | Check the error's list of available repository variables |

## Further reading

- [Earthmover Marketplace](https://app.earthmover.io/marketplace): browse listings
- [Marketplace docs: Data Users](https://docs.earthmover.io/marketplace/data-users): subscription mechanics
- [Marketplace docs: FAQ](https://docs.earthmover.io/marketplace/faq): pricing and plan requirements
- [Arraylake client installation](https://docs.earthmover.io/setup/installation): `arraylake` package setup
