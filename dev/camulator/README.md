# CAMulator port: verification scripts and cluster instructions

Work-in-progress port of NCAR's CAMulator (CREDIT) to an Earth2Studio prognostic
model. Branch `worktree-camulator-px`. Wrapper code lives in
`earth2studio/models/px/camulator.py`, `earth2studio/models/nn/camulator*.py`,
`earth2studio/data/camulator.py`, `earth2studio/lexicon/camulator.py`.
Design notes and everything established about the shipped model: `DESIGN.md`.

## Status (2026-09-14, laptop, CPU)

| check | result |
|---|---|
| `compare.py` architecture vs vendored CREDIT `Camulator` (random weights, small dims) | EQUIVALENT, max abs diff 0 |
| `compare_physics.py` tracer clip + wind filter + mass/water/energy fixers vs CREDIT `_postblock` + `WindPP` (real stats files) | EQUIVALENT, max abs diff 0 |
| `smoke_px.py` wrapper forward + 3-step ensemble iterator (tiny random net, full grid) | runs, but see below |
| unit tests | none written yet |
| real checkpoint rollout (`run_real.py`) | never run |

Open problems found by `smoke_px.py`:

1. **Peak RSS 31 GB** for tensors that total well under 1 GB. Something in the
   wrapper (forcing fetch, padding, wind filter, fixers) is allocating ~100x
   what it should. Find this before trusting any GPU memory numbers.
2. **NaNs at rollout step 2** (step 1 is clean). May be the random unphysical
   state blowing up through the feedback loop; unverified.
3. **Conservation fixers return non-finite output** on the synthetic
   "physical" state at the end of the script. Fixers match CREDIT bit-for-bit
   on real stats (`compare_physics.py`), so this is likely the synthetic
   center/scale, but it is unverified.

## Cluster setup

```bash
git clone -b worktree-camulator-px git@github.com:negin513/earth2studio.git
cd earth2studio
uv sync --extra camulator          # wrapper itself has no extra deps
uv run python dev/camulator/fetch_assets.py   # mean/std/statics/IC (~50 MB) -> dev/camulator/assets/
```

Set `CAMULATOR_ASSETS=/some/shared/path` before `fetch_assets.py` if you want the
assets outside the repo (compare_physics.py and run_real.py read the same var).
The HF checkpoint (4.8 GB) is fetched by `CAMulator.load_default_package()` on
first use into the Earth2Studio cache; set `EARTH2STUDIO_CACHE` to control where.
The HF revision is pinned in `earth2studio/data/camulator.py`.

## Runs

Peak-memory report on Linux: prefix with `/usr/bin/time -v` and read
"Maximum resident set size". The scripts also print RSS (and CUDA peak) themselves.

```bash
# 1. equivalence checks (CPU is fine, a few minutes)
uv run --with einops python dev/camulator/compare.py          # einops only for the vendored CREDIT reference
uv run python dev/camulator/compare_physics.py

# 2. smoke test, first CPU to reproduce the 31 GB, then GPU
/usr/bin/time -v uv run python dev/camulator/smoke_px.py
DEVICE=cuda uv run python dev/camulator/smoke_px.py

# 3. real checkpoint, 8 steps from the shipped 1981-01-01 IC
DEVICE=cuda STEPS=8 uv run python dev/camulator/run_real.py
DEVICE=cuda STEPS=8 FIXERS=0 WIND=0 uv run python dev/camulator/run_real.py   # isolate the post-processing chain
```

`run_real.py` assumes the shipped IC `.pth` is the normalized 136-channel model
input in south-to-north latitude order (channels 0..129 = prognostic state). That
assumption has not been verified against a CREDIT run; the script asserts surface
pressure lands in 400..1100 hPa after denormalization and aborts otherwise.

What "good" looks like for step 3: all steps finite with fixers and wind filter on,
PS staying in roughly 500..1050 hPa, top-level T not drifting by tens of K over 8
steps. For a real validation, compare against the CREDIT `Quick_Climate.py` output
from the same IC and forcing (HF repo `willychap/camulator`, branch
`camulator_huggingface` of `WillyChap/miles-credit`).

## Files

- `smoke_px.py` wrapper smoke test with a tiny random `CamulatorNet`.
- `compare.py` architecture equivalence vs `credit_camulator.py` / `credit_boundary_padding.py` (vendored from CREDIT, need einops).
- `compare_physics.py` post-processing equivalence vs `stub/credit/` (CREDIT postblock + physics core, vendored) and `stub/WindPP.py`.
- `run_real.py` real checkpoint rollout.
- `fetch_assets.py` downloads the small HF assets.
- `DESIGN.md` design and facts about the model.

This directory is dev scaffolding and should not be merged upstream as is.
