# insitubatch × Earth2Studio: streaming hindcast IO

`earth2studio.data.insitu.InSituForecastFeed` feeds ERA5 into a prognostic **without** building
the dense `(init, lead)` grid that `fetch_data` materializes. It reads the analysis store with
[insitubatch](https://github.com/emfdavid/insitubatch), planning the reads so each stored chunk
is decoded once no matter how many `(init, lead)` pairs need it, and yielding batches as they
are ready.

What that buys an Earth2Studio campaign:

- **Verification reads collapse onto chunks.** A scoring grid needs ERA5 at `valid = init + lead`
  for every pair; consecutive inits share valid times, and a fat time-chunk holds several steps.
- **Bounded memory over a long campaign.** Peak memory tracks the window you stream, not the
  size of the campaign, so a 120-init run costs what a 12-init run costs.
- **No materialized copy of the verification set** — and with `cache_dir`, a re-score reads the
  chunks it touched back from local disk instead of the cloud.

It does **not** replace `recipes/eval`'s `predownload.py`, and structurally cannot: insitubatch's
parallelism lives in one async event loop rather than worker processes, so it does not scale a
bulk fetch across nodes. Predownload buys rank-parallel fetch, resumability, and a durable
pre-regridded artifact. What the feed replaces is the predownload-then-read *cycle* for streaming
consumption on a single box. Rank-parallel training and inference are unaffected — each DDP rank
streams its own shard.

## Install

insitubatch has its own extra, because it needs Python >= 3.12 (zarr-v3's floor, not
insitubatch's choice); folding it into `data` would raise that floor for every other source.

```bash
uv sync --extra data --extra insitu
```

## Getting started

Score a forecast against ERA5 on a public store — no credentials, no predownload pass. The feed
replaces `fetch_data`; everything after it is the ordinary Earth2Studio loop.

```python
from collections import OrderedDict

import numpy as np
from insitubatch import obstore_store

from earth2studio.data.insitu import InSituForecastFeed
from earth2studio.models.px import Persistence
from earth2studio.utils.coords import map_coords

WB2 = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

# Lead 0 is the initial condition; the rest are verification targets.
leads = np.array([np.timedelta64(h, "h") for h in (0, 6, 12, 18)])

feed = InSituForecastFeed(
    obstore_store(WB2, skip_signature=True),  # anonymous public read
    variables=["t2m"],
    var_map={"t2m": "2m_temperature"},  # store array name differs
    lead_times=leads,
    sample_range=(1000, 1016),  # 16 consecutive init times
    batch_size=4,  # 4 inits per batch
    transpose_inner=True,  # WB2 is stored lon-major
)

model = Persistence(
    variable=["t2m"],
    domain_coords=OrderedDict([("lat", feed.lat), ("lon", feed.lon)]),
    history=1,
    dt=np.timedelta64(6, "h"),
)

for x, coords in feed:
    # x is (inits, leads, variables, lat, lon); lead index 0 is the IC.
    ic, ic_coords = map_coords(
        x[:, 0:1],
        OrderedDict(
            [
                ("time", coords["time"]),
                ("lead_time", coords["lead_time"][0:1]),
                ("variable", coords["variable"]),
                ("lat", coords["lat"]),
                ("lon", coords["lon"]),
            ]
        ),
        model.input_coords(),
    )
    for step, (fx, _fc) in enumerate(model.create_iterator(ic, ic_coords)):
        if step == 0:
            continue  # step 0 is the IC
        rmse = (fx[:, -1] - x[:, step]).pow(2).mean().sqrt()
        print(f"init {coords['time'][0]}  lead {6*step:>3}h  RMSE {rmse:.3f}")
        if step == len(leads) - 1:
            break
    break

print(f"chunks decoded: {feed.dataset.cache_misses}")
feed.dataset.close()
```

```text
init 1959-09-08T00:00:00.000000000  lead   6h  RMSE 2.666
init 1959-09-08T00:00:00.000000000  lead  12h  RMSE 3.672
init 1959-09-08T00:00:00.000000000  lead  18h  RMSE 3.211
chunks decoded: 3
```

**Three chunks for 64 requested field reads** (16 inits × 4 leads): the window spans sample
indices 1000–1018, and WB2 chunks the time axis 8 steps deep, so those 64 reads live in 3 chunks.
`feed.dataset` also exposes `cache_hits` and `resident_peak`.

The knobs that matter: `variables` / `var_map` select arrays and name them; `lead_times` takes a
model's `input_coords()["lead_time"]` for a history window (values `<= 0`), verification leads
(`> 0`), or their union; `sample_range` picks the init window; `batch_size` is inits per batch and
sets the streaming memory; `transpose_inner=True` for lon-major stores.

### With a real checkpoint

`Persistence` keeps the example above free of a checkpoint download, but nothing in it is
specific to `Persistence`. `ucast_example.py` runs the same pattern against U-CAST, which takes
83 channels and a two-step history. Most of those channels are one level of a pressure-level
array, and `WB2Lexicon` already names each channel's array and level in the spelling `var_map`
accepts — so the model describes its own inputs and the feed resolves them:

```python
model = UCast.load_model(UCast.load_default_package())
ic = model.input_coords()

feed = InSituForecastFeed(
    obstore_store(WB2, skip_signature=True),
    variables=ic["variable"],                                  # 83 channels
    var_map={v: WB2Lexicon.VOCAB[v] for v in ic["variable"]},  # "geopotential::500"
    lead_times=ic["lead_time"],                                # [-12h, 0h] history
    sample_range=(2000, 2008),
    batch_size=4,
    transpose_inner=True,
)
```

```bash
python ucast_example.py                      # CPU
python ucast_example.py --device cuda        # batches land on the GPU
```

```text
83 channels over 11 arrays; 8 inits x 2 history steps = 1328 requested field reads
  batch 0 (from 1960-05-15T00:00)  lead  12 hours  z500 mean 54147.2 m2/s2
  batch 0 (from 1960-05-15T00:00)  lead  24 hours  z500 mean 54170.4 m2/s2
  batch 1 (from 1960-05-16T00:00)  lead  12 hours  z500 mean 54151.4 m2/s2
  batch 1 (from 1960-05-16T00:00)  lead  24 hours  z500 mean 54171.1 m2/s2
chunks decoded: 22
```

Both batches together cost the same 22 decodes the first one did: the second batch's chunks
were already resident.

**22 chunks for 1328 requested field reads.** A `var_map` value may name a level as
`"array::level"` to select one level of a `(sample, level, …)` array, and channels sharing an
array share one read — a stored chunk holds every level of a step anyway, so U-CAST's 83
channels cost the **11 arrays** that hold them, not 83.

This pulls a 6.7 GB checkpoint on first run. U-CAST is stochastic, so the z500 means move in
the last digit or two between runs while the decode count does not; sanity-check the magnitude
rather than the digits (ERA5's own z500 over this window averages ~54056 m²/s²). With
`--device cuda` the feed lands each batch straight on the GPU: at `batch_size=4` U-CAST peaks
at 6.3 GB, so a 23 GB L4 has room to spare, and the decode count is the same 22 either way.

### Re-scoring the same ground truth

A checkpoint sweep reads one fixed verification set many times. `cache_dir` persists the decoded
chunks a run touches, and a later run over the same store reads them from local disk instead of
the cloud — only the chunks touched, no dense copy, and a reanalysis store never goes stale. For
the many-scorers shape, warm it once and give the scoring jobs `readonly_cache=True`: they take
the directory lock shared, write nothing, and a miss **raises** rather than quietly reaching for
the cloud — so a campaign whose cost model assumes no egress fails loudly instead of surprising
you with a bill.

## Benchmarks

Two runnable benchmarks quantify the two claims. Both read the **same store over obstore anon on
both sides**, so the delta is read planning, not the storage backend.

```bash
export LOGURU_LEVEL=INFO   # these scripts log per-read at DEBUG
export TMPDIR=/mnt/nvme    # put the baseline's cache on your fastest disk
```

All numbers below: **GCP n2-standard-8 (8 vCPU, 31 GB) + local NVMe, us-central1**, reading
in-region public GCS; insitubatch 0.2.0, zarr 3.3.0. Walls are medians over the stated repeats,
measured in one session with both legs interleaved. **Decode counts are deterministic; walls are
not** — quote the counts.

### 1. `bench_hindcast.py` — verification-read de-duplication

BEFORE is Earth2Studio's per-init `fetch_data`; AFTER is the feed over the same window. The
BEFORE leg runs in two configurations, because E2S's sources cache by default (`cache=True`) and
that materially changes the wall.

```bash
python bench_hindcast.py --store wb2 --vars t2m u10m v10m \
  --n-init 48 --max-lead-h 240 --repeats 14 --before-cache
python bench_hindcast.py --store arco --vars t2m --lead-step-h 6 \
  --n-init 24 --max-lead-h 144 --repeats 10 --before-cache
```

| store | chunks | requested → decodes | cache **on** | cache off |
| --- | --- | --- | --- | --- |
| **WB2** 240×121 6-h | `(8,240,121)` fat | 5760 → **33** (174×) | **9.4×** | 14.6× |
| **ARCO** 721×1440 1-h | `(1,721,1440)` chunk-1 | 576 → **162** (3.6×) | **1.6×** | 1.7× |

Medians: WB2 8.72→0.93 s cached (14 repeats), 12.65→0.87 s uncached (12); ARCO 3.86→2.40 s
cached, 4.34→2.57 s uncached (10 repeats each).

Quote the cache-on column — it is how a stock Earth2Studio run behaves; drop `--before-cache` for
the other. `LocalCachingStore` sits at the `Store.get` level and holds *compressed* buffers, so a
redundant read costs a local disk hit instead of a round-trip, but zarr still decodes the chunk
again: **decode counts are identical in both configurations**, and no byte cache addresses decode.
That is why the cache recovers only part of the WB2 baseline — its chunks are ~116 KB, so the
network was never the bottleneck there; the cost is 5760 decodes against 33.

**These ratios are against the live path, not against `predownload.py`.** The BEFORE leg
re-requests every `(init, lead)` pair, which is what the pipelines do
(`fetch_data(time=[item.time], …)` per work item, no memory across items) — predownload
de-duplicates valid times first. Against a valid-time-deduplicated baseline the advantage is
smaller, and it is exactly the sample-axis steps per chunk: WB2 261 unique reads → 33 decodes =
**7.9×**; ARCO 162 → 162 = **1.0×, none**. On a chunk-1 store, chunk granularity *is* timestamp
granularity. That is arithmetic from the geometry, not a measurement.

### 2. `stream_score.py` — streaming vs dense materialization

`create_iterator` already streams the forecast lead-by-lead and scoring is pointwise per
`(init, lead)`, so the verification never needs to be dense. Roll out a window of inits, score
each lead against a just-read slice, discard.

```bash
for m in stream dense e2s; do
  python stream_score.py --mode $m --n-init 120 --n-leads 40
done
```

| mode | wall | **peak RSS** | field reads |
| --- | --- | --- | --- |
| `stream` — feed, `batch_size=W` | 2.8–3.6 s | **1.72 GB** | 60 |
| `dense` — feed, `batch_size=N` | 4.1 s | 7.49 GB | 60 |

Same reads, same backend, same store: the only difference is materialization, and streaming is
both lower-memory and no slower. Peak memory for `stream` tracks the window, so it is flat in
campaign size where `dense` grows with it — that bounded-memory property, more than throughput,
is the point for a long campaign.

The third mode, `e2s`, scores the same campaign through live per-init `fetch_data` with **no
insitubatch in the loop**, and all three agree to three decimals on RMSE at every lead
(3.637 / 5.061 / 5.071 at 24/120/240 h). That agreement is the correctness check — throughput
alone would not catch a loader that silently aliased or double-lent a buffer. It issues **14 760
field reads against the feed's 60**, a real read-count difference, but its wall is **not** quoted
as a speedup: it accumulates into a dense buffer that is this harness's construction rather than
Earth2Studio's, and a real predownload would first collapse those 120 inits × 41 leads onto the
~160 unique valid times they span.

## Where it does not win

De-duplication is the mechanism, so removing the redundancy removes the advantage. Run ARCO with
one init at unit lead spacing — every requested read unique, 0.67 GB moved either way:

```bash
python bench_hindcast.py --store arco --vars t2m --lead-step-h 1 \
  --max-lead-h 162 --n-init 1 --repeats 8
```

162 requested = 162 unique, de-dup ratio **1.0×**, and the feed comes out **~10% slower** than
`fetch_data` (1.45 s baseline against 1.59 s) — at parity per byte, near enough. So read the ARCO
row above as the boundary case it is: de-duplication removes a redundant sample's fetch and
decode but **not its assembly** — the tensor still has one slot per requested `(init, lead)` —
which is why 3.6× fewer decodes nets only ~1.6× wall. Where fields are small (WB2), assembly is
negligible and most of the ratio converts.

A degenerate `batch_size=N` also throws the memory advantage away: `dense` above peaks at 7.49 GB
against `stream`'s 1.72 GB. The large wins land where the chunk layout maps many samples onto
shared chunks — overlapping windows, verification grids, fat time-chunks.

## Scope / caveats

- **Single environment, preliminary.** One n2-standard-8-class box, in-region anonymous GCS.
  Numbers to be cross-posted after NVIDIA-side runs on target infrastructure.
- **Pressure levels are selected, not reduced.** `"array::level"` indexes a store that keeps
  level as a dimension (the WB2 and ARCO layouts). A store that flattens level into array names
  needs no level syntax at all. What the feed does not do is derive fields across variables —
  that belongs upstream of it.
- **ARCO's time axis begins in 1900, its data in 1940.** Chunks outside ~1940–2023 read back as
  NaN fill in ~20 ms with no network request, so `--start` defaults per store (ARCO `1051896` =
  2020-01-01) and both benchmarks fail loudly if a window reads back entirely NaN. Earth2Studio's
  `ARCO` source refuses pre-1940 requests; insitubatch does not.
- **Persistent cache footprint.** The cache stores *decoded* chunks, so per-chunk bytes exceed the
  compressed store — but it is bounded to the unique chunks touched, not the dense grid a
  predownload materializes. The `cache_dir` path is the cache identity.
- **One process per `cache_dir`** (exclusive advisory lock; `readonly_cache=True` openers share
  it). Released by the kernel on process death, so there is no stale lock to clean up.
- **The win is the IO-bound campaign** — many inits, verification-heavy: hindcast scoring, lagged
  ensembles. A single-IC long rollout is compute-bound, where the loader is a rounding error.
- `Persistence` keeps the first example free of a checkpoint download while still exercising the
  real `create_iterator` seam; the U-CAST example above is the same code against a real
  checkpoint, and both run on CPU. A model needing a grid the store does not carry (SFNO wants
  721×1440) needs a store at that resolution, not a change here.
