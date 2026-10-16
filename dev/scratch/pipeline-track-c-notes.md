# Track C notes — `earth2studio/run/output.py` (`OutputManager`)

Implemented against `dev/spec/EXECUTION_CONTRACT_SPEC.md` ("Output", "Shard
ownership", `X13`/`X14`/`X15`) and `dev/spec/IO_SPEC.md` (`output_template`,
`I4`, `I9`, `I11`).

## Spec ambiguities and the decisions taken

- **Where the ownership check runs.** The spec says `prepare` checks known
  backends, without saying whether before or after `add_array`. Decided:
  *before*, so a rejected layout never creates a store, and every rank raises
  the same `ValueError` before entering the `rank0_first` barrier (no rank is
  left waiting).
- **One check per backend row.** `IceChunkBackend` subclasses `ZarrBackend`,
  so a single `isinstance(backend, ZarrBackend)` covers the spec's first row.
- **"Warn once" granularity** (`X15`). Interpreted as once per *backend type*
  per `OutputManager`, not once per stream and not process-global, so a run
  with four `XarrayBackend` streams logs one warning. Recorded in
  `OutputManager._warned` keyed by class name.
- **`ensemble` chunks "within member groups".** Formalized as: for each
  reference time, map each item's `member_ids` onto positions of the sorted
  union, and reject if any chunk/shard block of positions intersects two
  different member groups of that time. This permits a chunk of 2 with groups
  `(0,1)`/`(2,3)` and rejects a chunk of 4, and tolerates groupings that differ
  between times.
- **`time` chunks of 1.** Implemented as "effective block ≤ 1 when there is
  more than one item time", where the effective block is
  `min(backend.chunks.get("time", size), size)`. A store with a single time is
  therefore always fine. A `ZarrBackend` with `chunks={}` (nothing requested)
  is *rejected*, because an unchunked `time` axis is one chunk; only the
  default `chunks` (`ensemble`/`time`/`lead_time` = 1) and equivalents pass.
- **Empty `items`.** `prepare` raises `ValueError` when it has backends and no
  items (`output_template` would otherwise build a zero-sized `time`, which
  `add_array` rejects with a less helpful message, `I2`). With no backends,
  `prepare` is a no-op, so a scorer-only `Pipeline` can call it unconditionally.
- **A backend naming a stream with no schema.** Spec is silent here (`X10` is a
  `Pipeline` rule). `prepare` raises `ValueError` listing the unknown stream
  names, so typos are caught even if `Pipeline` ever forgets to.
- **`prepare` called twice.** Not guarded: calling it again just re-runs
  `add_array`, which is a no-op (`I4`). `Pipeline` calls it once.
- **Attribute name.** The backends mapping is exposed as the public attribute
  `OutputManager.backends` (a copy of the caller's mapping), which `Pipeline`
  may read; `prepare`/`write`/`flush`/`close` are the supported entry points.

## Where I believe the spec needs an amendment

- **A failing `close` can still hang a run.** The spec has `close` barrier and
  `Pipeline` call `raise_if_any_failed(error)` *after* `outputs.close()`.
  `OutputManager.close()` closes every backend, barriers, and then re-raises
  the first error (silently dropping an Icechunk commit failure would lose
  data). If only one rank's commit fails, that rank raises from `close` and
  skips `raise_if_any_failed`, so the other ranks block in that collective.
  Proposed wording for the "Output"/"Run loop" sections: *"`close` returns the
  errors it collected rather than raising; `Pipeline` folds them into the
  `error` it passes to `raise_if_any_failed`, so a close failure is reported
  through the same collective as a run failure."* Until that is resolved,
  `OutputManager.close()` raises after its barrier, and the `Pipeline` track
  should prefer to call it inside the same `try` that feeds
  `raise_if_any_failed` (see below).
- **`AsyncZarrBackend` and `chunked_coords`.** The spec's row only mentions
  `parallel_coords` and shards, but `AsyncZarrBackend._compute_chunks` lets
  `chunked_coords` override a parallel coordinate's chunk of 1
  (`earth2studio/io/async_zarr.py:838-845`), so `chunked_coords={"time": 4}`
  re-creates exactly the race the row is meant to exclude. The check therefore
  reads `chunked_coords` as well as `shard_coords`. Suggest the spec row read:
  *"`time`/`ensemble` in `parallel_coords`, not re-chunked by `chunked_coords`,
  and no shard spanning items."*

## IO-backend gaps (no IO change made, as required)

- **No portable chunk/shard introspection.** The check is backend-specific by
  construction: `ZarrBackend.chunks` (requested chunks, already overwritten
  from the arrays of a reopened store by `_read_store_state`),
  `AsyncZarrBackend.parallel_coords` / `chunked_coords` / `shard_coords`, and
  class identity for everything else. `NetCDF4Backend` exposes nothing about
  its layout, hence the `X15` warning. A future (optional) protocol member such
  as `storage_blocks(template) -> Mapping[str, int]` would replace the whole
  `isinstance` ladder; not added here.
- **`ZarrBackend` shards.** `ZarrBackend` never creates shards, and
  `_read_store_state` folds only *chunks* of existing arrays into
  `self.chunks`, so a store sharded by another writer would go unnoticed. The
  check therefore also reads `array.shards` from the arrays already in the
  group (best effort, `_stored_shards`), which needs no IO change.

## What the `Pipeline` track must know

- `OutputManager(output or {})` — an empty mapping is valid; `prepare`,
  `flush`, `write` and `close` are all safe with it, and every rank must call
  `prepare` and `close` because both are collective.
- `prepare(schemas, items)` takes **stored** schemas (post-filter,
  post-regrid) and the **full** item list. It builds the templates itself with
  `output_template(signature, leading={"ensemble": members, "time": times})`;
  do not pre-concretize `time`/`ensemble`, and keep them out of the signature's
  fixed dims (`X1`). `lead_time` extent must already be in the signature.
- `prepare` raises `ValueError` for a backend whose stream has no schema and
  for an empty item list, both before any collective, so all ranks raise
  together given identical arguments (`X7`).
- `write(stream, x)` silently ignores streams with no backend, so `_emit` does
  not need to check `self.output`.
- `close()` is idempotent and must be called on every rank, including after a
  local failure. It raises the first backend error *after* its barrier — see
  the amendment above; calling `close()` inside the `try` whose error goes to
  `raise_if_any_failed` is the safer arrangement today.
- The caller's backends are never replaced or dropped, so `run.*` wrappers can
  return the backend they passed in after `close`.
- Ensemble templates use the sorted union of `member_ids` across *all* items,
  so the `ensemble` axis is identical at any world size and on resume (`X8`).
