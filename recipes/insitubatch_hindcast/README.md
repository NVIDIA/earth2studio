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

```bash
# insitubatch is declared in earth2studio's `data` extra (needs Python >= 3.12):
uv sync --extra data
```

Both stores are anonymous public GCS buckets (WeatherBench2 ERA5, ARCO ERA5); no credentials
needed. Every measurement below reads over **gcsfs anon on both the before and after side**, so the
delta isolates insitubatch's read-planning + streaming — it is *not* an obstore-vs-gcsfs artifact.

## 1. `bench_hindcast.py` — verification-read de-duplication

A scoring grid needs ERA5 at `valid = init + lead` for every `(init, lead)`. Consecutive init
times share valid times, and a fat time-chunk holds several steps, so the requested reads collapse
onto far fewer stored chunks. BEFORE = E2S's per-init `fetch_data`; AFTER = the insitubatch feed
(each lead a sample-axis `shift` view; each shared chunk decoded once).

```bash
python bench_hindcast.py --store wb2  --vars t2m u10m v10m --n-init 48 --max-lead-h 240 --repeats 5
python bench_hindcast.py --store arco --vars t2m --n-init 24 --lead-step-h 6 --max-lead-h 144 --repeats 3
```

| store | layout | requested reads | unique decodes | **wall speedup** |
|-------|--------|-----------------|----------------|------------------|
| **WB2** 240×121 6-h | `chunks=(8,240,121)` (fat) | 5760 | **33** (174×) | **15.4×** (14.0 s → 0.91 s) |
| **ARCO** 721×1440 1-h | `chunks=(1,721,1440)` (chunk-1) | 576 | 162 (3.6×) | ~1.9× |

WB2's fat time-chunk amortizes 8 steps per read, so the de-dup ratio is large and the fields are
small — insitubatch dominates. ARCO is the **honest** case (see caveats).

## 2. `stream_score.py` — streaming vs dense materialization

The model's `create_iterator` already streams the forecast lead-by-lead, and scoring is pointwise
per `(init, lead)` — so the verification never needs to be a dense tensor. Interleave instead:
roll out a window of inits, score each lead against a just-read verification slice, discard. Three
modes, all producing **identical RMSE** (a correctness check):

```bash
for m in e2s dense stream; do python stream_score.py --mode $m --n-init 120 --n-leads 40; done
```

| mode | wall | **peak RSS** | field reads |
|------|------|--------------|-------------|
| `e2s` — dense predownload (redundant reads) | 39.6 s | 3.04 GB | 14 760 |
| `dense` — insitubatch, `batch_size=N` | 4.8 s | 7.53 GB | 60 |
| `stream` — insitubatch, `batch_size=W` | 3.3 s | **1.81 GB** | 60 |

Streaming's peak memory is **flat at ~1.9 GB across N = 120 / 240 / 480**, while the dense grid is
7.53 GB at N = 120 and **OOMs a 15 GB box by ~N = 240**. Dense scales with campaign size; streaming
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
python bench_cache.py --store wb2  --vars t2m u10m v10m --n-init 48 --max-lead-h 240 --cache-dir /mnt/nvme/insitu_cache
python bench_cache.py --store arco --vars t2m --n-init 12 --max-lead-h 48 --cache-dir /mnt/nvme/insitu_cache
```

| store | field size | cold → warm wall | **cloud fetches (cold → warm)** |
|-------|------------|------------------|----------------------------------|
| **WB2** 240×121 | 116 KB | 1.41 s → 1.05 s (~1.4×) | **33 → 0** |
| **ARCO** 721×1440 | 4 MB | 1.22 s → 0.55 s (~2.2×) | **54 → 0** |

The deterministic result is **zero cloud fetches on re-score** — the warm run serves every chunk
from local disk. The wall speedup is secondary and scales with how IO-bound the cold fetch is (tiny
WB2 fields ~1.4×; 4 MB ARCO fields ~2.2×); it is *understated* on this box's cheap same-region reads
and grows under metered egress, requester-pays, or cross-region access. The cold wall includes the
one-time persist write, so it runs slightly above the persist-off de-dup figure in §1.

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
   Evidence: §1 WB2 (174× fewer decodes, 15× wall).
3. **Honest boundary — you can use it sub-optimally.** It is not a universal speed win. On a
   chunk-1 store with large fields, against an *unbounded* concurrent gather, its bounded-inflight
   scheduling trails per byte (ARCO ~2×; the reads are already minimal — verified — but the dense
   output the model consumes must still be assembled, and E2S's flat gather saturates bandwidth on
   4 MB chunks). And a degenerate `batch_size=N` throws away the memory advantage. The tool is
   **generally optimal for streaming with bounded memory** — that is the sweet spot.

One line: *stream training/inference batches from cloud tensors in place, with bounded memory —
competitive with hand-tuned parallel loaders on optimized layouts, and far ahead when the chunking
causes duplicate reads.*

## Caveats / methodology

- **Single environment, preliminary.** One n2-standard-8-class box (15 GB RAM), cold reads, gcsfs
  anon. Numbers to be **cross-posted** after NVIDIA-side runs on the target infrastructure.
- **gcsfs on both sides.** Isolates the loader's contribution from the store backend; obstore would
  raise the AFTER throughput further but is not what these numbers measure.
- **Surface variables only** (`t2m`, `u10m`, `v10m`); pressure-level variables need level indexing,
  not yet wired in the adapter.
- **Persistent cache footprint.** The cache stores *decoded* chunks, so per-chunk bytes exceed the
  compressed store — but it is bounded to the unique chunks touched (decode-once), not the dense
  grid a predownload materializes. The `cache_dir` path is the cache identity; use a fresh one when
  the store or variable set changes.
- **The win is the IO-bound campaign** (many inits, verification-heavy — hindcast scoring, lagged
  ensembles). A single-IC long rollout is compute-bound, where the loader is a rounding error.
