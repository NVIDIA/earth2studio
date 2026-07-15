# GFS Cache Integrity and Loop-Safe Coalescing

## Goal

Fix two cache regressions in `earth2studio.data.GFS`:

1. A reused `GFS` instance must not retain synchronization objects bound to an old
   event loop.
2. A cached byte range must be reused only when it matches the complete requested
   range.

The implementation must preserve atomic cache writes and coalesce concurrent
same-range downloads within one event loop.

## Cache Identity and Validation

Cache filenames will use a SHA-256 digest of an unambiguous serialization of the
remote path, byte offset, and byte length. Including `byte_length` prevents two
requests that begin at the same offset but request different ranges from sharing a
cache entry. This intentionally invalidates filenames produced by the old
path-and-offset-only scheme.

For a finite byte-range request, an existing cache file is valid only when its size
equals the requested byte length. A wrong-size file is removed and treated as a
cache miss. Full-file requests have no expected size available locally, so they rely
on the atomic-write guarantee and the new cache namespace.

Downloads continue to write to a unique temporary file and use `os.replace` only
after the complete payload has been written. A downloaded finite range whose size
does not match the request raises `IOError` and is never promoted into the cache.

## Loop-Safe Miss Coalescing

The persistent `dict[str, asyncio.Lock]` will be replaced with a registry of
in-flight download tasks keyed by `(current_event_loop, cache_path)`.

When a cache miss occurs:

1. The current event loop checks the registry for an existing task for that cache
   path.
2. If present, the caller awaits the shared task through `asyncio.shield` so one
   cancelled waiter cannot cancel the download for every waiter.
3. If absent, the loop creates and registers a download task.
4. A completion callback removes that exact task from the registry on success,
   failure, or cancellation.

Tasks from different event loops are never shared or awaited across loops. Separate
loops may perform duplicate downloads, but unique temporary paths and atomic replace
keep the final cache entry consistent. Because completed tasks are removed, the
registry does not retain event loops or grow with every historical cache path.

## Error Handling

- Wrong-size existing ranges are deleted and refetched.
- Wrong-size downloads raise `IOError` without leaving a final or temporary file.
- Download exceptions propagate to all same-loop waiters and remove the in-flight
  registry entry so a later request can retry.
- Temporary files are removed in a `finally` block.

## Tests

Focused unit tests will verify:

- an exact-size existing range is reused without a remote call;
- a wrong-size existing range is removed and refetched;
- different byte lengths produce different cache paths;
- concurrent same-loop misses make one remote call;
- the same `GFS` instance can coalesce misses in two successive event loops after
  cache eviction;
- atomic writes, wrong-size download rejection, and failed-download cleanup remain
  intact.

The affected tests, Black, Ruff, MyPy for the changed surface, and `git diff --check`
must pass before handoff.
