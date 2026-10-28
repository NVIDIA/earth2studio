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
```

Other CorrDiff wrappers expose `number_of_steps`.
For `CorrDiffCosmoEra5SDA`, set `model.sampler_steps`.

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
