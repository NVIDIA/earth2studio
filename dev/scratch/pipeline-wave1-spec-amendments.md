# Wave-1 amendments to `dev/spec/EXECUTION_CONTRACT_SPEC.md`

Issues found while implementing tracks A (`run/work.py`), B (`run/runner.py`) and C
(`run/output.py`). Settled items are now in the spec; this note keeps only what is
still open for wave 2 (`run/pipeline.py`) and later.

## Applied to the spec

1. `work_items` keeps the caller's `times` order and normalizes to second
   resolution.
2. Every rank reads and compares the stored manifest after rank 0 writes it.
3. Rank 0 picks the progress run-attempt id in `open` and broadcasts it.
4. `ProgressStore.tags` merge into the manifest under `"tags"`.
5. Any unparseable progress line is skipped with a warning, not only a torn last
   line.
6. `OutputManager.close()` returns its first backend error instead of raising, and
   `Pipeline` passes it to `raise_if_any_failed`. Raising on one rank skipped the
   failure-flag reduction and hung the others. Code and tests changed with the spec.
7. `AsyncZarrBackend` rejects a `chunked_coords` entry that widens a parallel
   coordinate.
8. `ZarrBackend(chunks={})` fails the distributed ownership check.
9. `WorkItem` and `member_seed` live in the leaf module `run/work.py`.

## Open

- **DLESyM initial-condition publication (deferred by owner).** DLESyM lat/lon
  `input_coords` variables are not a subset of its `output_coords`, so publishing
  `initial_condition(x)` into the forecast stream writes labels the stream lacks.
  The fused face-grid test passes only because the lead-0 row lacks those
  variables and stays fill-valued. Planned data-source support for derived
  variables should subsume this; revisit in E1 if it does not. Until then, the
  `variable` part of `X4` is "a subset of", not "equal to".
- **Setup collectives can hang on a rank-0-only failure.** `ProgressStore.open`
  compacts and writes the manifest on rank 0 only, and `OutputManager.prepare`
  runs `add_array` on rank 0 inside `rank0_first`. An IO error there raises on
  rank 0 while the others wait at the barrier until the process-group timeout.
  The spec's run loop calls both before its `try`. Wave 2 should decide whether
  to guard them with the same failure-flag reduction or accept the timeout.
