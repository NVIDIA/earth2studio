# Proposed amendments to `dev/spec/EXECUTION_CONTRACT_SPEC.md` (wave 1)

These issues surfaced while implementing tracks A (`run/work.py`), B
(`run/runner.py`), and C (`run/output.py`) against the spec. They are
collective, not per-track, and should be applied to the spec PR before wave 2
(`run/pipeline.py`) begins.

## 1. `work_items` ordering and time precision

**Spec now:** "Items are ordered time-major."

**Issue:** Does the function sort caller-provided `times`? Does it normalize
datetime precision? A 1 ns difference between two otherwise-identical
reference times would produce distinct items and distinct progress units.

**Implementation choice:** Keep the caller's `times` order and normalize to
`datetime64[s]`, so items built from `datetime64[ns]`/`datetime64[h]` inputs
hash/compare equal and match `member_seed` granularity (already second
resolution).

**Proposed spec wording:** *"Items are ordered time-major, in the caller's
`times` order. Times and horizons are normalized to second resolution for item
identity and progress keying."*

## 2. How non-zero ranks learn of a manifest mismatch

**Spec now:** "Rank 0 writes the manifest if none exists; if one differs, every
rank raises `ValueError` naming the differing keys."

**Issue:** It is not stated how the non-zero ranks learn of the mismatch.

**Implementation choice:** Rank 0 writes or leaves the manifest, then a
barrier, then every rank reads and compares the same file, so every rank raises
its own `ValueError`. No broadcast of the error is needed.

**Proposed spec wording:** *"After rank 0 writes the manifest, all ranks load it
and compare it to the manifest they were given; every rank raises `ValueError`
naming the differing keys."*

## 3. Run-attempt id in log naming

**Spec now:** "Each rank appends to its own log under `path`, named by run
attempt and rank."

**Issue:** "Run attempt" is undefined.

**Implementation choice:** Rank 0 computes the next attempt id after
compaction and `broadcast`s it. Because compaction deletes previous per-attempt
logs, the id usually resets to 0, but the broadcast keeps all ranks in sync.

**Proposed spec wording:** *"Rank 0 picks the run attempt id in `open` and
broadcasts it so each rank names its log uniquely."*

## 4. `ProgressStore.tags` and the manifest

**Spec now:** The manifest table lists `tags` as a key, but the interaction
between `ProgressStore(tags=...)` and the manifest is not described.

**Implementation choice:** Merge `ProgressStore.tags` into the caller-provided
manifest under `manifest["tags"]`. A tag change therefore becomes a manifest
mismatch. Store tags win over caller tags on collision.

**Proposed spec wording:** *"The store merges its `tags` argument into the
manifest under the `"tags"` key before writing; a tag change is a manifest
mismatch."*

## 5. Torn-line tolerance

**Spec now:** "A torn last line is ignored, so no record outlives its data."

**Issue:** A corrupted middle line would crash resume.

**Implementation choice:** Skip any line that fails to parse (warning once), not
only the final fragment.

**Proposed spec wording:** *"A torn final line and any other unparseable line
are skipped with a warning; only valid records count."*

## 6. `OutputManager.close()` can still hang a distributed run

**Spec now (Output):** `close` is collective and idempotent; it closes every
backend, then barriers. **Spec now (Run loop):** `raise_if_any_failed(error)` is
called *after* `outputs.close()`.

**Issue:** If a backend commit fails on one rank inside `close`, that rank
raises from `close` and never reaches `raise_if_any_failed`. The other ranks
block in the close barrier.

**Proposed fix:** *"`OutputManager.close()` returns the first backend error it
encountered (or `None`) instead of raising. `Pipeline` folds that error into the
`error` it passes to `raise_if_any_failed`, so a close failure is reported on
every rank via the same failure-flag reduction used for run failures."*

## 7. `AsyncZarrBackend` `chunked_coords` override — FIXED IN BACKEND

**Issue:** `AsyncZarrBackend._compute_chunks` let `chunked_coords` override a
parallel coordinate's chunk of 1, so `chunked_coords={"time": 4}` re-created
the same shared-chunk race the ownership row is meant to prevent. This already
contradicted the documented meaning of `chunked_coords` ("chunk sizes for
coordinates that are *not* in `parallel_coords`").

**Resolution:** Fixed in the backend rather than worked around in supervision.
`AsyncZarrBackend.__init__` now raises `ValueError` when `chunked_coords` names
a parallel coordinate with a size other than 1 (and for any non-positive chunk
size), and `_compute_chunks` no longer lets such an entry through. The parallel
chunk of 1 is now an invariant of the backend, not a convention supervision has
to re-derive. `OutputManager` keeps reading `chunked_coords` only as a cheap
safeguard; it still catches a distributed dimension that is chunked but absent
from `parallel_coords`. Covered by
`test_async_zarr_chunked_coords_validation`; no in-tree caller passes a
parallel coordinate in `chunked_coords`, including the eval recipe, which
already excludes its iteration dims.

**Spec now reads:** *"`time`/`ensemble` in `parallel_coords`; no shard spans
items. The backend chunks a parallel coordinate at 1 and rejects a
`chunked_coords` entry that would widen it."*

## 8. Default Zarr `time` chunking

**Observation:** The ownership check treats a `ZarrBackend` constructed with
`chunks={}` as unsafe for distributed runs, because an unchunked `time` axis
becomes one shared chunk. Only the default `chunks={"ensemble": 1, "time": 1,
"lead_time": 1}` (or equivalent) passes. This matches the intent of `X14`; worth
being explicit.

## 9. `member_seed` / `WorkItem` location — FIXED

**Issue:** `runner.py` defined `WorkItem` while `work.py` imported it, so
`runner.py` had to lazy-import `member_seed` inside `_seed_model` to break a
circular import. Fragile, and invisible until someone moved the import.

**Resolution:** `WorkItem` now lives in `run/work.py`, which imports nothing
else from `run`. `runner.py` imports `WorkItem` and `member_seed` from it at
module level and the lazy import is gone. `earth2studio.run.WorkItem` and
`earth2studio.run.work.WorkItem` are the same object, so no public import path
changed. The spec's module table was updated to match.

`DataRequest` deliberately stayed in `runner.py`: it holds a `Source`
(`DataSource | ForecastSource`), so moving it would pull `earth2studio.data`
into the work layer and make distribution and progress depend on the data
layer. `work.py` needs only numpy.

## 10. DLESyM initial-condition publication (deferred by owner)

**Issue:** The face-grid DLESyM model's `input_coords` variables are not a subset
of its `output_coords`, so publishing `initial_condition(x)` into the forecast
stream would write labels not present in the stream. The fused DLESyM test works
because the lead-0 row simply lacks those variables and is fill-valued.

**Resolution:** Deferred, not a wave-1 concern. Planned data-source support for
auto-computing requested derived variables is expected to subsume it; revisit
during the eval integration track (wave 3, E1) if it does not.
