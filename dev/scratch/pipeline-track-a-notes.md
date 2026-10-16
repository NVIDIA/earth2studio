# Pipeline track A notes (`earth2studio/run/work.py`)

Non-normative. Ambiguities hit while implementing the "Work items",
"Seeding", "Progress" and "Distribution" sections of
`dev/spec/EXECUTION_CONTRACT_SPEC.md`, the decision taken for each, and
what the `Pipeline` track needs to know.

## Decisions where the spec is silent

- **Time-major ordering does not sort.** `work_items` keeps the caller's
  `times` order and only makes time the outer loop. Sorting would silently
  reorder a caller's campaign; `OutputManager.prepare` sorts times itself.
  Proposed wording: "Items are ordered time-major, in the caller's time
  order."
- **Normalization to seconds.** `work_items` stores `time` as
  `datetime64[s]` and `horizon` as `timedelta64[s]`, so items built from
  `datetime64[ns]` and `datetime64[h]` inputs compare *and hash* equal, and
  so item identity matches the progress unit key and `member_seed`
  granularity (both already second-resolution). Sub-second reference times
  are therefore not representable; nothing in the spec needs them.
- **Input validation in `work_items`:** `member_batch >= 1`; negative
  `horizon` rejected; `members` as an int must be `>= 1`; explicit member
  IDs must be non-empty, non-negative and unique; repeated times rejected.
  All raise `ValueError`.
- **`_validate_items` is private.** The "repeats no unit / all-or-no
  members" check lives in `work.py` as `_validate_items(items)` and runs at
  the end of `work_items`. `Pipeline._validate` (track C) may import it
  rather than reimplementing the rule — say the word and I will make it
  public (`validate_work_items`), the spec lists no public name for it.
- **Progress directory layout:** `manifest.json`, the compacted
  `done.log`, and per-attempt per-rank logs `done-<attempt:04d>-<rank:04d>.log`.
  Module constants `MANIFEST_NAME` and `COMPACT_LOG_NAME` are exported for
  tests/tools.
- **Run attempt id.** The spec says logs are "named by run attempt and
  rank" without defining the attempt. Rank 0 computes
  `max(existing attempt) + 1` *after* compaction and `broadcast`s it inside
  `open`, so every rank agrees and no file ever gets two writers. Because
  compaction removes the per-attempt logs, the attempt number usually
  resets to 0 — that is safe, since the files it would collide with are
  gone.
- **Log record format:** one JSON object per line,
  `{"time": "<ISO seconds>", "horizon": <int seconds>, "members": [ints]}`.
  A deterministic item writes `"members": []` and maps to a unit with
  member `None`. Torn-line tolerance = drop a final fragment with no
  trailing newline, plus skip (with a `logger.warning`) any line that fails
  to parse.
- **Compaction is skipped when there is at most one log**, so a freshly
  created store does not grow a `done.log`. The bound in the spec
  (manifest + compacted log + one log per rank) still holds.
- **`clear()` is rank-local**, not collective; `open` calls it on rank 0
  only (under `overwrite=True`) and then barriers. It tolerates missing
  files, so eval's `_remove_progress_dir` retry dance is not needed: we
  delete files, never a shared tree, and only from one rank.
- **`ProgressStore.close()` / context manager** added so a long-lived
  append handle can be released (tests, and `Pipeline` may call it after
  `run`). Not in the spec's method list; purely additive.
- **Lazy load.** `remaining`/`is_done`/`mark_done` work without `open`
  (they read the logs on first use), which keeps the class usable outside
  `Pipeline`. `open` is still required for the manifest check and
  compaction.
- **`raise_if_any_failed`** raises the local error when there is one, else
  `RuntimeError` naming this rank when some other rank failed. Flag tensor
  is `int32` with `ReduceOp.MAX`.
- **NCCL device choice:** `cuda:$LOCAL_RANK` if `LOCAL_RANK` is set, else
  `cuda:{torch.cuda.current_device()}`. `broadcast_object_list` gets the
  same `device=`.

## Possible spec wording fixes

- "Items are ordered time-major" should say whether `times` is sorted (see
  above). Likewise, whether `work_items` normalizes time precision.
- The "Progress" section says `open` is collective and rank 0 writes the
  manifest, but does not say how the non-zero ranks learn of a mismatch.
  Implemented as: rank 0 writes if absent, barrier, then *every* rank reads
  the file and compares, so each raises its own `ValueError` naming the
  differing keys (no broadcast of the error needed). Suggest stating this.
- The "run attempt" in the log naming rule is undefined; suggest adding
  "rank 0 picks the attempt in `open` and broadcasts it".
- `tags` interaction with the manifest is implied but unstated. Implemented
  as: `ProgressStore` merges its `tags` into `manifest["tags"]` (store tags
  win), so a tag change is a manifest mismatch. The `Pipeline` manifest
  table already lists `tags`, so `Pipeline` may pass them through instead —
  either way the stored file is the same.
- "A torn last line is ignored" — the implementation also tolerates any
  unparsable line (warning + skip) rather than failing a resume on a
  corrupted middle line. Worth saying explicitly.

## For the `Pipeline` track

- Import: `from earth2studio.run.work import work_items, distribute_work,
  member_seed, ProgressStore, rank_info, rank0_first, broadcast, barrier,
  raise_if_any_failed`. `run/__init__.py` is untouched (track C owns the
  exports).
- `progress.open(manifest)` is collective and must be called on **every**
  rank, including idle ones, and before any `mark_done`.
- `progress.remaining(items)` is pure and rank-agnostic; per `X18`
  `Pipeline` should call it on rank 0 only and `broadcast` the result —
  `broadcast` pickles `WorkItem`s fine (frozen dataclass of numpy scalars).
- The manifest mapping must be JSON-serializable and **identical on every
  rank**; non-JSON values are stringified (`default=str`) before
  comparison, so e.g. numpy dtypes compare by `str(dtype)`.
- `distribute_work` raises `ValueError` for an out-of-range rank or
  `world_size < 1`; it never exits or logs above DEBUG.
- `work_items` returns items with `datetime64[s]` times — if `Pipeline`
  builds store templates from `item.time`, labels will be second
  resolution unless it converts.
