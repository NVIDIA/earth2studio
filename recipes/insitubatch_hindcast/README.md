# insitubatch × Earth2Studio: streaming hindcast IO

Two runnable benchmarks that feed ERA5 into an Earth2Studio prognostic **without** the dense
`fetch_data` grid — reading the analysis store with [insitubatch](https://github.com/emfdavid/insitubatch)
(`earth2studio.data.insitu.InSituForecastFeed`) instead. They quantify what a streaming,
read-planning loader changes for an IO-bound hindcast / scoring campaign.

The motivation is `recipes/eval`: its `predownload.py` sentinel exists because live `fetch_data`
is too slow for a scoring campaign. That predownload materializes the whole `(init, lead)`
verification grid up front. Both are consequences of a per-`(time, variable)` fetch with no read
de-duplication — exactly what insitubatch removes.

## Setup

insitubatch is declared in earth2studio's `data` extra (needs Python >= 3.12):

```bash
uv sync --extra data
```

Both stores are anonymous public GCS buckets (WeatherBench2 ERA5, ARCO ERA5); no credentials
needed. Every measurement below reads over **obstore anon on both the before and after side**, so
the delta isolates insitubatch's read-planning + streaming rather than the storage backend.

Run the benchmarks with Earth2Studio's per-fetch debug logging suppressed. It emits one line per
`(time, variable)` — 5760 lines in §1, 14 760 in §2, essentially all on the baseline leg — from
inside the timed region, and it is not free. **Every number below is measured with it off:**

```bash
export LOGURU_LEVEL=INFO
```

(`LOGURU_LEVEL` configures loguru's default handler. It gates these benchmarks, but several other
Earth2Studio data modules call `logger.remove()` at import and re-add a handler with no level, for
which the variable has no effect.)

## 1. `bench_hindcast.py` — verification-read de-duplication

A scoring grid needs ERA5 at `valid = init + lead` for every `(init, lead)`. Consecutive init
times share valid times, and a fat time-chunk holds several steps, so the requested reads collapse
onto far fewer stored chunks. BEFORE = E2S's per-init `fetch_data`; AFTER = the insitubatch feed
(each lead a sample-axis `shift` view; each shared chunk decoded once).

```bash
python bench_hindcast.py --store wb2 --vars t2m u10m v10m \
  --n-init 48 --max-lead-h 240 --repeats 5
python bench_hindcast.py --store arco --vars t2m --lead-step-h 6 \
  --n-init 24 --max-lead-h 144 --repeats 10
```

| store | layout | requested reads | unique decodes | **wall speedup** |
|-------|--------|-----------------|----------------|------------------|
| **WB2** 240×121 6-h | `chunks=(8,240,121)` fat | 5760 | **33** (174×) | **12.8×** (10.84→0.84 s) |
| **ARCO** 721×1440 1-h | `chunks=(1,721,1440)` chunk-1 | 576 | 162 (3.6×) | 1.5× (3.91→2.57 s) |

WB2's fat time-chunk amortizes 8 steps per read, so the de-dup ratio is large and the fields are
small — insitubatch dominates. ARCO is the **honest** case (see caveats). ARCO's per-repeat spread
is wide on this box (±15% on both legs); its row is the median of 10 repeats, and the BEFORE leg
converges downward with sample size as HTTP connections warm.

**Separating read elimination from read throughput.** The speedups above conflate two things:
reading *less*, and reading *fast*. To isolate the second, run the same store with no redundancy at
all — one init at unit lead spacing, so every requested read is unique:

```bash
python bench_hindcast.py --store arco --vars t2m --lead-step-h 1 \
  --max-lead-h 162 --n-init 1 --repeats 8
```

162 requested = 162 unique = 0.67 GB moved on both sides. E2S: **1.38 s** (486 MB/s); the feed:
**1.61 s** (416 MB/s) — the feed is **~17% slower per byte**, roughly at parity (medians over three
independent 8-repeat runs). insitubatch's modest ARCO result is therefore *not* a throughput
deficit; see the honest boundary below for where the gap actually comes from.

## 2. `stream_score.py` — streaming vs dense materialization

The model's `create_iterator` already streams the forecast lead-by-lead, and scoring is pointwise
per `(init, lead)` — so the verification never needs to be a dense tensor. Interleave instead:
roll out a window of inits, score each lead against a just-read verification slice, discard. Three
modes, all producing **identical RMSE** (a correctness check):

```bash
for m in e2s dense stream; do
  python stream_score.py --mode $m --n-init 120 --n-leads 40
done
```

| mode | wall | **peak RSS** | field reads |
|------|------|--------------|-------------|
| `e2s` — dense predownload (redundant reads) | 29.0 s | 3.10 GB | 14 760 |
| `dense` — insitubatch, `batch_size=N` | 4.3 s | 7.63 GB | 60 |
| `stream` — insitubatch, `batch_size=W` | 2.9 s | **1.85 GB** | 60 |

All three agree to three decimals on RMSE at every lead (3.637 / 5.061 / 5.071 at 24 h / 120 h /
240 h), and the `e2s` mode computes it through an entirely independent path — dense predownload via
`WB2ERA5_121x240`, no insitubatch in the loop. That agreement is the correctness check; throughput
alone would not catch a loader that silently aliased or double-lent a buffer.

Streaming's peak memory is **flat at ~1.9 GB across N = 120 / 240 / 480**, while the dense grid is
7.63 GB at N = 120 and **OOMs a 15 GB box by ~N = 240**. Dense scales with campaign size; streaming
does not. That bounded-memory property — not just throughput — is the point for a long campaign.

(Persistence is a checkpoint-free model that exercises the real `create_iterator` seam on CPU; a
real NVIDIA checkpoint — SFNO/FCN — is a drop-in with the same code on a GPU.)

## 3. `bench_cache.py` — cross-run persistent cache

The intro's `predownload.py` exists so a re-scored campaign doesn't re-fetch the same ground
truth. `InSituForecastFeed(cache_dir=...)` gives that for free: the first run decodes each shared
chunk once **and** persists it to local disk; a later run over the same store reads those chunks
back as cache hits, touching the cloud zero times. No predownload step, no reshard, only the chunks
actually touched — and because a reanalysis store is static, the cache never goes stale. This is the
common eval shape: many models (or checkpoints) scored against one fixed verification set.

```bash
python bench_cache.py --store wb2 --vars t2m u10m v10m \
  --n-init 48 --max-lead-h 240 --cache-dir /mnt/nvme/insitu_cache
python bench_cache.py --store arco --vars t2m \
  --n-init 12 --max-lead-h 48 --cache-dir /mnt/nvme/insitu_cache
```

| store | field size | cold → warm wall | **cloud fetches (cold → warm)** |
|-------|------------|------------------|----------------------------------|
| **WB2** 240×121 | 116 KB | 1.16 s → 0.81 s (1.4×) | **33 → 0** |
| **ARCO** 721×1440 | 4 MB | 0.81 s → 0.59 s (1.4×) | **54 → 0** |

The deterministic result is **zero cloud fetches on re-score** — the warm run serves every chunk
from local disk. The wall speedup is secondary and modest: 1.4× on both stores, despite a 35×
difference in field size, so it is *not* tracking how IO-bound the cold fetch is. On this box's
cheap same-region reads the cloud fetch simply isn't the bottleneck, so removing it entirely buys
little; the wall win grows under metered egress, requester-pays, or cross-region access, while the
fetch-elimination holds everywhere. The cold wall includes the one-time persist write, so it runs
slightly above the persist-off de-dup figure in §1.

The benchmark wipes and rebuilds `<cache-dir>/bench_cold_warm/<store>/` so the cold leg starts
empty; it never touches the rest of `--cache-dir`.

## How to read these numbers — framing insitubatch

insitubatch is a **streaming batch loader** that trains/infers in place on cloud zarr: all
parallelism lives in one async event loop, the Python hot path is O(chunks) not O(samples), and
memory is bounded by a residency budget rather than the working set. The two benchmarks above
sharpen its positioning into three evidence-backed claims:

1. **Competitive with an optimized parallel loader, at lower memory.** On a well-chunked store and
   for streaming consumption it matches a hand-tuned concurrent fetch's throughput while holding
   *bounded* memory (streaming: flat ~1.9 GB where dense predownload OOMs). Evidence: §2.
2. **Far ahead when the chunking strategy isn't sample-optimized.** When the access pattern maps
   many samples onto shared chunks — overlapping windows, verification grids, fat chunks holding
   several steps — its read planning de-duplicates and a per-sample parallel fetch re-reads.
   Evidence: §1 WB2 (174× fewer decodes, 12.8× wall).
3. **Honest boundary — you can use it sub-optimally.** It is not a universal speed win, and on a
   chunk-1 store with large fields the reason is *not* raw throughput: with redundancy removed from
   both sides the feed is only ~17% slower per byte than E2S's unbounded gather (§1). The gap opens
   because **de-duplication removes the fetch and decode of a redundant sample, but not its
   assembly** — the tensor the model consumes still has one slot per requested `(init, lead)`.
   Subtracting the two §1 ARCO configurations, each of the 414 redundant samples costs E2S 6.1 ms
   (it refetches) and the feed 2.3 ms (it re-assembles from an already-resident chunk): 2.6× cheaper,
   not 3.6×, which is why 3.6× fewer decodes nets only 1.5× wall. Where fields are small (§1 WB2)
   assembly is negligible and nearly the whole de-dup ratio converts. And a degenerate
   `batch_size=N` throws away the memory advantage: §2 `dense` peaks at 7.63 GB, *worse* than the
   predownload it replaces. The tool is **generally optimal for streaming with bounded memory** —
   that is the sweet spot.

One line: *stream training/inference batches from cloud tensors in place, with bounded memory —
competitive with hand-tuned parallel loaders on optimized layouts, and far ahead when the chunking
causes duplicate reads.*

## Caveats / methodology

- **Single environment, preliminary.** One n2-standard-8-class box (15 GB RAM), cold reads,
  anonymous GCS. Numbers to be **cross-posted** after NVIDIA-side runs on the target infrastructure.
- **obstore on both sides.** Earth2Studio's zarr data sources migrated to obstore in
  [#955](https://github.com/NVIDIA/earth2studio/pull/955); the feed uses insitubatch's
  `obstore_store` to match, so neither side carries a backend handicap. The de-duplication ratios
  are backend-independent — swapping both sides to `fsspec_store(url, token="anon")` changes the
  wall clock but not the chunk counts.
- **These numbers supersede an earlier gcsfs measurement.** This recipe was first measured on
  2026-07-04, over gcsfs on both sides and before #955 landed, and reported 15.4× (§1 WB2), ~1.9×
  (§1 ARCO), 39.6 s (§2 `e2s`) and ~2.2× (§3 ARCO). Every headline is lower now. That run predates
  #955, read over gcsfs on both sides, and was measured with the debug logging above still enabled
  — so it is superseded rather than a controlled comparison, and should not be quoted. One
  difference is established independently of all that: the §3 ARCO row was reading an unwritten
  region of the store (below) — all-NaN fills that never touched the network — so it measured
  nothing, and its "~2.2×, scales with field size" result was an artifact.
- **ARCO's time axis begins in 1900, its data in 1940.** The store declares
  `hours since 1900-01-01` over 1 323 648 steps to 2050, but chunks outside ~1940–2023 were never
  written and read back as NaN fill in ~20 ms without a network request. `--start` therefore
  defaults per store (ARCO: `1051896` = 2020-01-01) and both benchmarks now fail loudly if a window
  reads back entirely NaN. Earth2Studio's `ARCO` source validates this independently and refuses
  pre-1940 requests; insitubatch does not, so a window outside the populated range returns fill
  data rather than raising.
- **Surface variables only** (`t2m`, `u10m`, `v10m`); pressure-level variables need level indexing,
  not yet wired in the adapter.
- **Persistent cache footprint.** The cache stores *decoded* chunks, so per-chunk bytes exceed the
  compressed store — but it is bounded to the unique chunks touched (decode-once), not the dense
  grid a predownload materializes. The `cache_dir` path is the cache identity; use a fresh one when
  the store or variable set changes.
- **The win is the IO-bound campaign** (many inits, verification-heavy — hindcast scoring, lagged
  ensembles). A single-IC long rollout is compute-bound, where the loader is a rounding error.
