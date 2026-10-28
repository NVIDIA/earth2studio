# Diffusion solvers

Earth2Studio diffusion and flow-matching wrappers use numerical integration to transform
initial noise into a generated weather field. The numerical method integrates the
sampling ODE in reverse time for diffusion wrappers. Stochastic methods also inject
noise during integration. Experimenting with that method can help you compare forecast
quality and computational cost at the sampling budget you intend to use.

The wrappers expose [PhysicsNeMo solvers](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/diffusion/samplers.html#solvers)
through optional `solver` and `solver_options` arguments. They continue to handle
conditioning, normalization, guidance, and output coordinates. Solver overrides
are opt-in: omitting them preserves the model's default solver selection.

Install the selected model's optional dependencies before loading its checkpoint.
The loading examples use the [AutoModel interface](auto.md) and download model
assets when needed.

## Select a solver when loading

Start with the normal `CorrDiffTaiwan` loading workflow:

```python
from earth2studio.models.dx import CorrDiffTaiwan

package = CorrDiffTaiwan.load_default_package()
model = CorrDiffTaiwan.load_model(package)
```

To experiment with DPM++(2M), change only the loading line:

```python
model = CorrDiffTaiwan.load_model(package, solver="dpmpp_2m")
```

The API name remains `dpmpp_2m`. To select DPM++(2M) with UniC2 correction,
use the separate name `dpmpp_2m_unic2`.

## Configure the selected method

Pass options to the selected solver through `solver_options` to customize its
behavior or override defaults.
PhysicsNeMo provides general ODE integration methods that you can also configure
for classical diffusion sampling. The options let you adjust a method without
changing the wrapper's conditioning or output processing. Supported options
depend on the method. Consult the [PhysicsNeMo solver API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/diffusion/samplers.html#solvers)
for constructor arguments and their mathematical meaning.

For example, select stochastic Heun and increase noise injection during its steps:

```python
model = CorrDiffTaiwan.load_model(
    package,
    solver="edm_stochastic_heun",
    solver_options={"S_churn": 10.0},
)
```

The stochastic Heun solver defaults to `S_churn=0.0`. A positive value adds noise
during eligible integration steps. Initial latent noise also contributes to
sample variability. Unknown names or incompatible constructor options raise an
error.

## Set the sampling step count

A larger sampling step count can improve generation quality and increases
generation time. Tune the step count for your model, solver, and forecast
metrics. Quality does not improve at every larger step count.

For `CorrDiffEra5Hrrr`, select the step count when loading:

```python
from earth2studio.models.dx import CorrDiffEra5Hrrr

corrdiff = CorrDiffEra5Hrrr.load_model(
    CorrDiffEra5Hrrr.load_default_package(),
    number_of_steps=32,
)
```

For other already loaded models, set their sampling controls:

```python
stormcast.sampler_steps = 32  # StormCast and StormCastSDA
stormcast_conus.num_diffusion_steps = 32
stormscope.sampler_args = dict(stormscope.sampler_args, num_steps=32)
meteosat.sampler_args["num_steps"] = 32
cbottle.sampler_steps = 32  # Non-distilled C-Bottle sampling
```

Here, `stormscope` is a loaded `StormScopeGOES` or `StormScopeMRMS`, and
`meteosat` is a loaded `StormScopeMeteosatEU`. For `StormScopeDxNSRDB`, set
`model.num_steps` instead.
Other CorrDiff wrappers expose `number_of_steps`.
For `CorrDiffCosmoEra5SDA`, set `model.sampler_steps`.
Distilled `CBottleSR` keeps its fixed two-step schedule.

Do not pass `num_steps` in `solver_options`. Use the wrapper's sampling control
so the step count and sampling time grid stay consistent.

## Compare solver choices

Use the model's default as your baseline. Compare forecast metrics, ensemble
spread, and Neural Function Evaluation (NFE) counts. Equal step counts can
have different computational costs. Keep the model's documented seed controls
when comparing solvers, and use paired ensemble members where possible.
Call `model.set_rng(seed)` after loading to reset the sampling stream. Model
calls accept `xarray.DataArray` inputs and return fields with their coordinates.
Methods that consume random numbers differently can produce different samples
from the same seed.

The installed PhysicsNeMo version must provide the selected method. Defaults
vary across models, and some loaders read them from package metadata.
For `CorrDiffCosmoEra5`, select `mode="diffusion"` to enable sampling.
`CorrDiffEra5Hrrr` supports EDM and rectified-flow packages.

## Sampling with `StormCast`

`StormCast`, `StormCastCONUS`, `StormCastSDA`, and `StormCastConusSDA` accept
the same solver loader options.
For example, compare DPM++(2M) at a specified sampling budget:

```python
from earth2studio.models.px import StormCast

model = StormCast.load_model(
    StormCast.load_default_package(),
    sampler_steps=18,
    solver="dpmpp_2m",
)
```

Use `sampler_steps` to change StormCast's sampling grid. Solver selection keeps
the wrapper's conditioning and guidance in place. For `StormCastConusSDA`,
`num_diffusion_steps` controls unguided forecasts and `num_sda_diffusion_steps`
controls guided analyses. Supply conditioning fields and observations explicitly
to the model, and call `set_rng(seed)` before starting a reproducible trajectory.

## Coupled forecast with `StormScope`

StormScope Flash variants use a fixed distilled rollout and do not accept
solver overrides. The examples below use the teacher variants.

`StormScopeGOES`, `StormScopeGOESConditioned`, `StormScopeMRMS`,
`StormScopeMeteosatEU`, and
`StormScopeDxNSRDB` accept `solver` and `solver_options` when loading.
The conditioned GOES wrapper also requires an explicit conditioning field at
initialization and each later step.
The GOES and MRMS wrappers keep their noise-dependent expert selection.

A coupled GOES/MRMS forecast illustrates why solver choices need evaluation
through the complete forecast. Both models advance in ten-minute steps.
The GOES model predicts the satellite state, while the MRMS model predicts
radar fields conditioned on the satellite history. After the initial step,
that history includes the GOES model's own predictions.

The following example uses the published benchmark's `3km_10min` checkpoints.
Run it on a CUDA machine with both models' optional dependencies installed.
Prepare `satellite` and `radar_history` as `xarray.DataArray` objects with
one batch and one initialization. Match each model's `input_coords()` signature,
including its native grid, ordered variables, and required observation history.
The radar history contains MRMS and GLM observations.
Earth2Studio's `GOES`, `GOESGLMGrid`, and `MRMS` data sources supply these
observations. Data preparation and model loading can require downloads.

```python
import torch

from earth2studio.models.px import StormScopeGOES, StormScopeMRMS

solver_kwargs = {"solver": "dpmpp_2m_unic2"}
goes = StormScopeGOES.load_model(
    StormScopeGOES.load_default_package(),
    model_name="3km_10min",
    amp=True,
    compile=False,
    **solver_kwargs,
).to("cuda").eval()
radar = StormScopeMRMS.load_model(
    StormScopeMRMS.load_default_package(),
    model_name="3km_10min",
    amp=True,
    compile=False,
    **solver_kwargs,
).to("cuda").eval()

for model in (goes, radar):
    model.sampler_args = dict(model.sampler_args, num_steps=8)
    model.set_rng(42)

with torch.no_grad():
    satellite_forecast, satellite_state = goes.initialize(satellite)
    output, radar_state = radar.initialize(radar_history, satellite)
    for _ in range(2):
        output, radar_state = radar.step(output, satellite_forecast, radar_state)
        satellite_forecast, satellite_state = goes.step(
            satellite_forecast, satellite_state
        )
```

Each radar step receives the satellite forecast at its current lead time.
The explicit states keep the histories needed to advance both models.
The final `output` contains the radar forecast at 30 minutes.
Repeat the complete trajectory for each ensemble member, resetting the input
histories and calling `set_rng` before each run.

To compare DPM++(2M) without correction, reload both models with:

```python
solver_kwargs = {"solver": "dpmpp_2m"}
```

For the published stochastic Heun comparison, reload both models with these
arguments and choose the sampling budget through `model.sampler_args`:

```python
solver_kwargs = {
    "solver": "edm_stochastic_heun",
    "solver_options": {"S_churn": 10.0},
}
```

### Published exploratory comparison

Reproducing the plotted Heun results requires the PhysicsNeMo precision fix
described in [the benchmark prerequisites](https://github.com/NVIDIA/earth2studio/pull/1250).
The pinned dependency revision does not include that fix.

The [published benchmark](https://github.com/NVIDIA/earth2studio/pull/1250)
compares coupled composite reflectivity at 30 minutes across three dates,
using four paired ensemble members per date. NFE includes both networks over
all three forecast steps. Fair continuous ranked probability score (CRPS)
measures ensemble quality against observed MRMS reflectivity. Lower is better.
The reference uses nearest-neighbor grid alignment from the
[Earth2Studio evaluation workflow](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval).
Scoring excludes missing reference cells and clips other negative reflectivity
values to zero.

![Coupled StormScope quality versus NFE](https://raw.githubusercontent.com/CharlelieLrt/earth2studio/fa8a0a683e90cc73c847db81232b6b56cc4b0d24/figures/1250-stormscope-quality.png)

The curve reports the mean score across dates. Vertical bars show the range.
At the illustrated low budget, DPM++(2M) with UniC2 scores 2.668 dBZ at 54 NFE.
Stochastic Heun with churn set to 10 scores 13.21 dBZ at 48 NFE.
The comparison changes churn as well as the integration method. The two-step
and three-step Heun budgets produced non-finite forecasts, which the plot
excludes from mean scores. These exploratory results describe this setup.
Test the method at your own forecast horizon and sampling budget.

![Coupled StormScope fields and errors](https://raw.githubusercontent.com/CharlelieLrt/earth2studio/fa8a0a683e90cc73c847db81232b6b56cc4b0d24/figures/stormscope-fields.png)

The maps show the first date and paired realization zero at NFE budgets within
25% of each other. The top row shows the reference and generated fields.
The bottom row shows each realization minus the reference. Fields share one
color scale, and errors share a separate zero-centered scale. A single map
illustrates spatial behavior. The ensemble scores summarize all three dates.

## Infill, guidance, and video sampling

`CBottleInfill`, `CBottleTCGuidance`, and `CBottleVideo` accept `solver` and
`solver_options` when loading. Super-resolution has specific restrictions.

## Super-resolution sampling

Use `CBottleSR.load_model` when selecting a PhysicsNeMo solver for SR. For
example, this explicitly selects Euler for the distilled checkpoint:

```python
from earth2studio.models.dx import CBottleSR

package = CBottleSR.load_default_package()
model = CBottleSR.load_model(
    package,
    distilled_model=True,
    solver="euler",
)
```

For distilled SR, `solver=None` keeps the native distilled sampling method.
An explicit alternative uses the checkpoint's two-step schedule through the
PhysicsNeMo adapter and can change predictions and randomness. It does not
turn the distilled checkpoint into a model with an arbitrary sampling grid.
Distilled `solver_options` require an explicit `solver`.

If constructing `CBottleSR` with a supplied external SR helper, that helper keeps
its own sampler. The wrapper rejects solver overrides for such helpers.
The standard SR helper that `load_model` creates supports named PhysicsNeMo
methods.
