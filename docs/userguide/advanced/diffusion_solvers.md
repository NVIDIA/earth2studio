# Diffusion solvers

Earth2Studio diffusion wrappers expose PhysicsNeMo solvers through `solver` and
`solver_options`. These controls change how a model generates a sample from its
initial noise. The wrapper continues to handle conditioning, normalization,
guidance, and output coordinates.

Install the selected model's optional dependencies before loading its checkpoint.
The examples below download model assets through the usual
[AutoModel interface](auto.md). They configure sampling without running a
forecast.

## Select a solver when loading

Pass a solver name to a supported model's `load_model` method. For example,
this selects the `dpmpp_2m` method for StormCast:

```python
from earth2studio.models.px import StormCast

package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    sampler_steps=18,
    solver="dpmpp_2m",
)
```

Use `solver_options` for arguments of the selected solver. This example keeps
stochastic Heun but disables noise injection during its steps:

```python
from earth2studio.models.px import StormCast

package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    sampler_steps=18,
    solver="edm_stochastic_heun",
    solver_options={"S_churn": 0.0},
)
```

Initial latent noise still makes the generated forecast stochastic. For the
`edm_stochastic_*` methods, `S_churn`, `S_min`, `S_max`, and `S_noise` control
noise injection. These options do not apply to every solver. For example,
`solver="heun"` accepts `alpha`, while `S_churn` belongs to the stochastic
methods. Unknown names or incompatible constructor options raise an error.

The wrapper supplies the noise-schedule coefficients required by exponential
Euler and DPM-Solver++ methods. Users do not need to construct `bias_fn`,
`bias_int_fn`, `slope_fn`, or `lambda_fn` for these wrappers. Explicit options
override wrapper defaults, so changing these callbacks also changes the
mathematical formulation.

Set the sampling step count through the model's own parameter, such as
`sampler_steps` on StormCast or `number_of_steps` on CorrDiffEra5Hrrr.
The solver constructor's `num_steps` option controls churn scaling for methods
that use it. It does not change the wrapper's sampling time grid.

## Solver names

The reverse-time wrappers accept these eight tested PhysicsNeMo solver names:

| Name | Method |
| --- | --- |
| `euler` | Explicit Euler. |
| `heun` | Second-order predictor/corrector, with an Euler update at a zero endpoint. |
| `edm_stochastic_euler` | Euler with EDM noise injection. |
| `edm_stochastic_heun` | Heun with EDM noise injection. |
| `exponential_euler` | Exponential Euler with the wrapper's schedule coefficients. |
| `edm_stochastic_exponential_euler` | Exponential Euler with EDM noise injection. |
| `dpmpp_2m` | DPM-Solver++ with a 2M update. |
| `dpmpp_2m_unic2` | DPM-Solver++ with a 2M update and UniC-2 correction. |

The installed PhysicsNeMo version must provide the selected method. Keep the
model's default as the comparison baseline. A solver change can improve or worsen
forecast quality and runtime. Compare forecast metrics, ensemble
spread, runtime, and model evaluations at the intended step count.
Equal step counts do not imply equal model evaluation counts.

PhysicsNeMo Heun can call the predictor or guidance callback on the final step
even though a mask discards that step's correction. For $N$ steps ending at zero,
this means $2N$ denoiser calls where a legacy loop that skipped the terminal
correction made $2N-1$. Model evaluation counts can increase even when the
default sampled outputs agree. The extra call can also trigger callback side
effects.

## Wrapper coverage

| Models | Configuration |
| --- | --- |
| `CorrDiff`, `CorrDiffTaiwan`, `CorrDiffCMIP6` | Loader solver options. |
| `CorrDiffCosmoEra5` | Select `mode="diffusion"` to enable sampling. |
| `CorrDiffEra5Hrrr` | Supports EDM and rectified-flow packages. |
| `CorrDiffCosmoEra5SDA` | Loader options preserve guidance. |
| `StormCast`, `StormCastCONUS`, `StormCastSDA` | Preserves guidance. |
| `StormScopeGOES`, `StormScopeMRMS` | Preserves noise-dependent expert selection. |
| `StormScopeMeteosatEU`, `StormScopeDxNSRDB` | Loader solver options. |
| `CBottleInfill`, `CBottleTCGuidance`, `CBottleVideo` | Loader solver options. |
| `CBottleSR` | Loader solver options, with the SR exceptions below. |
| `Atlas` | Uses `model.solver` and the factory interface below. |
| `GenCastMini` | Native JAX sampler, without these overrides. |

Omitting `solver` preserves each wrapper's default selection. Those defaults
differ across models, and some loaders read them from package metadata.
Keep the model's documented RNG or seed controls when comparing solvers.
Methods that consume random numbers differently can generate different samples
from an identical seed.

Ensemble generation alone does not imply diffusion sampling. Models such as
`AIFSENS`, `AIFS2ENS`, `AtlasCRPS`, `FCN3`, `FuXiS2S`, `UCast`, the Aurora
ensemble variants, and the WeatherNext2 wrappers use other stochastic prediction
paths. They do not gain these controls from their ensemble behavior.

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
The standard SR helper that `load_model` creates supports the named methods
above.

## Atlas solver factories

Atlas uses `solver=None` by default. Its native SDE update participates in the
PhysicsNeMo solver protocol while keeping Atlas's diffusion and score-correction
terms. Pass a solver factory to `Atlas.load_model` for experiments:

```python
from functools import partial

from physicsnemo.diffusion.samplers import HeunSolver

from earth2studio.models.px import Atlas

model = Atlas.load_model(
    Atlas.load_default_package(),
    solver=partial(HeunSolver, alpha=1.0),
)
```

This factory receives the drift callable bound to the current condition and
creates a fresh solver for each sampling trajectory. Use the same pattern for a
custom PhysicsNeMo solver factory. Construct any history or caches inside
the factory. Passing a solver instance instead keeps caller-owned state across
sampling trajectories and can keep an old condition bound to its drift.

Atlas also accepts `"euler"` and `"heun"` directly. Its lightweight interface does
not accept the general `solver_options` mapping or the other named methods in
the table. A factory is the extension point for custom configuration.
Forward additive-noise SDE solvers require matching drift, diffusion, and time
direction. Use them only through a compatible Atlas factory. The reverse-time
wrappers above reject them.

The Heun example integrates Atlas's drift as an ODE. It omits the native SDE
diffusion and score correction, so it changes the generative dynamics.
A custom solver must supply those terms to preserve them. Set `model.solver =
None` to restore the native path. For forecasts beyond one step, use Atlas's
`create_iterator` interface so the model carries its latent history forward.
