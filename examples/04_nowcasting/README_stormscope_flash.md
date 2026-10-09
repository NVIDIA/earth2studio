# StormScope Flash

StormScope Flash is a variant of the existing `StormScopeGOES` and
`StormScopeMRMS` classes: GOES uses five backbone calls and **MRMS/GLM** uses
seven per forecast step. Both variants use the same coupled forecasting API.
The baseline retains its Heun sampler, checkpoints, default model names, and
GLM preprocessing.

## Install and run

Install this branch with `uv pip install '.[stormscope-flash]'`, using a NATTEN
build compatible with your PyTorch/CUDA installation. GPU execution is required
for NATTEN.

The example automatically downloads the selected native PhysicsNeMo `.mdlus`
checkpoints from `nvidia/stormscope-goes-mrms` on Hugging Face and caches them.
The download is pinned to a verified revision. GOES and MRMS/GLM Flash weights
are selected from their respective `checkpoints/*/3km_10min_flash/` directories;
the existing teacher checkpoints are not downloaded by the Flash loader.
Each Flash directory contains `expert_0.mdlus` (low sigma), `expert_1.mdlus`
(middle sigma), and `expert_2.mdlus` (high sigma), matching the baseline naming.

Run the example using the installed branch environment. It uses the **full
domain by default**. The same script runs the baseline or Flash:

```bash
# Baseline StormScope (default)
uv run --no-sync python examples/04_nowcasting/03_stormscope_goes_example.py

# StormScope Flash
uv run --no-sync python examples/04_nowcasting/03_stormscope_goes_example.py \
  --model 3km_10min_flash
```

For regional Flash inference, add geographic bounds:

```bash
uv run --no-sync python examples/04_nowcasting/03_stormscope_goes_example.py \
  --model 3km_10min_flash --lat 36.0195 44.4629 --lon -100.9688 -89.2842
```

Supply both bounds together, in degrees. Edit the small **Forecast configuration**
section in Python for `start_date`, `n_steps`, `padding`, `seed` and `output_dir`. The
example defaults to 18 ten-minute leads (three hours). Both variants enable
`amp=True` and `compile=True`. Flash selects FP16 AMP; the baseline retains its
existing AMP dtype and Heun settings. There are no satellite, package, date, step,
seed, output, padding or compilation flags.

The baseline runs the full domain; regional bounds require the Flash variant.
Regional inputs receive `padding = 25` real pixels per side, **once**. If only
15 pixels remain at an edge, only those 15 are used. Inputs expand inward as
needed to reach **200 × 200 pixels**, then align outward to four-pixel patches.
Minimum-size expansion and alignment may add extra real context; no synthetic
padding is generated. Edit `padding` in Python to request a different value.

The example saves **all forecast leads in a Zarr**, a **final-lead JPG**, and a
separate **animated GIF of all leads** with fixed color scales. Each invocation
reserves its own numbered run directory, including the initialization and bounds:

```text
outputs/stormscope_3km_10min_flash_20240313T2330_lat_39.075_41.075_lon_-97_-95/run_001/
    stormscope.zarr
    stormscope.jpg
    stormscope.gif
```

Repeating the same request creates `run_002`, then `run_003`; changing the region
uses a separate parent directory. Full-domain runs use `full` instead of bounds.
Previous forecasts are preserved, and the complete output directory is logged.
Geographic bounds are checked against the package's native grid before loading
checkpoints or reserving an output directory. Requests outside the model domain
produce a geographic error instead of an unrelated existing-output error.
The Zarr records requested bounds, model-input dimensions, actual context padding
and the satellite used for every observation frame.

The example automatically selects GOES-16/19 for each input-history timestamp,
using the baseline data source's operational dates. A history spanning the
handover uses the appropriate source for each frame. GLM retains the baseline
`satellite="east"` source. This does not add persistence filling.

The explicit `python` invocation uses the installed branch environment rather
than resolving the gallery script's inline dependencies from upstream. The
package is portable: loading uses relative asset paths. Set
`EARTH2STUDIO_VERIFY_CHECKPOINT_HASH=1` for full weight-hash verification on load.

## Models and region selection

```python
from earth2studio.models.px import StormScopeGOES, StormScopeMRMS
from earth2studio.data import GOESGLMGrid

package = StormScopeGOES.load_default_package(model_name="3km_10min_flash")
region = {"lat": (35.0, 41.0), "lon": (-102.0, -94.0)}
goes = StormScopeGOES.load_model(
    package, model_name="3km_10min_flash", region=region, padding=25,
)
mrms = StormScopeMRMS.load_model(
    package, model_name="3km_10min_flash", region=region, padding=25,
    glm_data_source=GOESGLMGrid(satellite="east"),
)
```

Omit `region` for full-domain 1024×1792 inference. Regions are geographic
rectangles within the native curvilinear footprint. Signed or 0–360 longitude
bounds work. Non-overlapping or partially out-of-domain requests raise clear
errors. Missing **context padding** at an edge is allowed: each side gets up to
25 real pixels, independently. No synthetic padding, resizing, or persistence
fill is added. Small requests expand to model inputs of at least 200 pixels on
each axis; inputs align outward to the four-pixel patch grid. Rectangles need
not be square. Alignment and minimum-size expansion may add extra real context.

`model.region_info` reports requested geographic bounds, half-open native input
and output indices, input dimensions, and actual padding in array order
(top, bottom, left, right). Padding is relative to the output rectangle.
Each model instance owns its geometry and caches; construct a new instance to
change regions. `model.crop_output(prediction, coords)` returns an independent
output copy with pixels outside the exact geographic rectangle masked. Keep
uncropped predictions for `next_input`; context must persist across every lead.

The Flash extra requires PhysicsNeMo 2.2.2 or newer. Flash reuses its rotary
neighborhood attention, DiT block components, MLP, patch projection, and timestep
embedding. The baseline StormScope dependency remains unchanged.

## Forecast contract

- Six input histories, T−50 through T at ten-minute intervals; one +10-minute output.
- GOES: eight ABI channels. MRMS: `refc`, `refc_base`, `glm_density`.
- Both models use 16/32/80 interval heads. GOES calls 1/1/3; MRMS calls 1/1/5.
- Low-region GOES blocks: 0–32, 32–64, 64–80. MRMS uses five 16-head blocks.
- Flash fuses heads with their flow-time interval widths. It is deterministic after
  the initial Gaussian latent; there is no EDM churn or Heun correction.
- Closed-form preconditioning uses data time `t = 1/(1+sigma)` and sigma_data 0.5.
  State/coefficient/fused-projection calculations use FP32. Network AMP defaults
  to BF16; pass `amp_dtype=torch.float16` to both loaders for FP16.
- RoPE is parameter-free, but Q/K normalization has learned scale/bias. All
  selected weights are retained and frozen for inference.
- In coupled forecasts, MRMS uses the current GOES history. Advance both
  histories only after both forecasts finish. GLM predictions feed back.
- GLM follows the E2S baseline: bilinear raw counts **then** log1p. This differs
  from the historical AGA preprocessing order, so identical raw-data forecasts
  and historical skill scores are not guaranteed by numerical model parity.
- GOES NaNs use learned missing-patch tokens and normalized-zero observations.
  MRMS GOES-conditioning NaNs are normalized-zero, matching training. Infinite
  inputs and non-finite radar/GLM values inside valid coverage still fail checks.

Regional skill must be assessed separately from full-domain skill; additional
context can materially affect forecasts.

## Compilation

Load either model with `compile=True` to compile its experts, or call
`model.compile_experts()` before the first forecast. Network AMP stays enabled.
The first forecast includes compilation; no separate warm-up is required.
Compilation can increase first-forecast latency, especially for small regions.
Keep model instances and input shapes stable to reuse compiled graphs across
forecasts. `compile=False` remains the default for occasional forecasts.
The example enables compilation for both variants and FP16 AMP for Flash. The
programmatic model API retains its existing `compile` option; other clients
can still choose their own execution policy.

The `.mdlus` package stores weights and reconstructible architecture metadata;
it does not store compiled kernels or remove compilation costs. The installed
Flash implementation and its matching dependencies are still required.

## Source layout

- `earth2studio/models/px/stormscope.py`: existing GOES and MRMS classes;
  `model_name="3km_10min_flash"` selects Flash loading and sampling.
- `earth2studio/models/nn/stormscope_flash.py`: checkpoint adapters and
  packed interval heads around PhysicsNeMo components. Adapters preserve learned
  Q/K normalization, AMP dtype, residual arithmetic, and cached conditioning.
- `earth2studio/models/px/_stormscope_flash/`: regional geometry and private
  loading/sampling helpers. `preconditioner.py` only re-exports the deployment
  class to keep previously published `.mdlus` import metadata loadable.
- `examples/04_nowcasting/03_stormscope_goes_example.py`: one customer entry point
  for baseline and Flash coupled forecasts.
- `examples/04_nowcasting/pack_stormscope_flash_package.py`: developer-only
  conversion utility. Customers download ready-to-run weights automatically;
  they do not need training checkpoints or a conversion step.
- `test/models/px/test_stormscope_flash.py`: sampler, geometry, compatibility,
  and real-checkpoint tests. Baseline tests remain in `test_stormscope.py`.

Training jobs, case manifests, benchmark reports, checkpoints, and generated
forecasts are external to the model integration.

## Validation

Run the baseline and Flash regression tests in the installed branch environment:

```bash
uv run --no-sync pytest test/models/px/test_stormscope.py \
  test/models/px/test_stormscope_flash.py -m "not package"

# Real checkpoint tests require GPU access and model downloads.
uv run --no-sync pytest test/models/px/test_stormscope.py \
  test/models/px/test_stormscope_flash.py -m package --package
```

The tests cover the shared model API, Flash schedules, trained normalization
parameters, regional geometry, and checkpoint loading. Forecast-skill comparisons
require real observations and matching verification domains; they are separate
from unit tests and checkpoint serialization checks.
