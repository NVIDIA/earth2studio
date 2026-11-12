# Shared RMM Allocator Implementation Plan

> **For agentic workers:** Use executing-plans to implement these tasks inline.

**Goal:** Automatically share per-device RMM resources between Torch, CuPy, and cuDF.

**Architecture:** A private, locked initializer installs RMM's Torch and CuPy
allocators once per process. Import-time initialization for auto/cupy precedes
CUDA tensor allocations. Backend scopes and explicit CuPy conversions also
initialize, with an actionable error if Torch's allocator is already initialized.
Default CUDA resources gain grow-on-demand pools; custom resources are preserved.
Allocator configuration persists after backend scope exit.

**Tech Stack:** PyTorch CUDAPluggableAllocator, CuPy, RMM, pytest subprocesses, uv.

## Tasks

- [x] Add `rmm-cu13>=26.2.0` on Linux with `uv add --no-sync`, then `uv sync --locked`.
- [x] Add subprocess tests in `test/utils/test_rmm.py`: automatic initialization,
  released-memory reuse, preserved custom resources, idempotence, late selection,
  explicit conversions, CPU-only import, and missing dependencies. Run
  `uv run pytest test/utils/test_rmm.py -q` and verify the missing behavior fails.
- [x] Create `earth2studio/utils/rmm.py` with a lock and successful-initialization
  flag. Load RMM only when CUDA is available; prepare zero-initial-size pools for
  each default CudaMemoryResource, change Torch's allocator before publishing the
  resources or CuPy allocator, then mark initialization successful.
- [x] Wire `earth2studio/utils/cupy.py` import policy, backend scopes, `_tensor_data`,
  and `as_cupy` into initialization. Keep type detection free of configuration.
- [x] Document process-global configuration, early import ordering, dependency
  installation, per-device pooling, and pluggable-allocator limitations in
  `docs/userguide/about/overview.md` and conversion docstrings.
- [ ] Run allocator tests and existing backend tests, then related utility tests.
  Verify GPU allocation reuse, DLPack sharing, and cuDF compatibility in fresh
  processes. Run `make lint`, formatting checks, and `git diff --check`.
- [ ] Review the complete base diff, commit intended files, confirm the PR title,
  push `origin rmm`, and open a `[E2S 1.0]` PR against `1.0.0-rc` with label `1.0.0`.

## Acceptance criteria

No setup call is introduced. `EARTH2STUDIO_DISABLE_RMM=1` before import disables
setup while retaining CuPy conversions, as subsequently requested. NumPy/Torch initial
policies and CUDA-unavailable imports do not configure RMM. Failure to switch
Torch leaves the CuPy allocator and installed RMM resources unchanged. Existing
custom RMM resources remain installed. Configuration is serialized and repeated
conversion/scoping does not replace live pools. GPU tests run in subprocesses
because allocator replacement is irreversible within a process.

## Accepted upstream limitation

RMM 26.4 and upstream main register only allocation/deallocation callbacks with
Torch. They do not register a `record_stream` callback. The new subprocess
regression `test_record_stream_prevents_premature_reuse` reproduces silent data
corruption on H100: a delayed cross-stream read returns 100 rather than 2 after
the shared pool reuses the allocation. Torch's native allocator returns 2.
The user approved proceeding with direct RMM integration with this limitation.
Document the required cross-stream lifetime handling and retain the reproduction
as a strict expected-failure test so an upstream fix is detected.
