---
name: earth2studio-create-coupling
version: 0.1.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, coupler, prognostic-model, python]
description: >
  Use when coupling two or more Earth2Studio prognostic models so one feeds
  another's forcing (e.g. StormScope GOES to MRMS, atmosphere to ocean at a
  different cadence): writes the small coupling function and wires it into a
  CoupledRunner. Not for single-model forecasts, new models, or data sources.
argument-hint: The models to couple, and which output feeds which forcing (optional)
---

# Create a Coupling

`earth2studio.coupler.CoupledRunner` is only a clock plus a place to plug in
exchange logic. The exchange for one model pair is a **coupling**: a plain
function, written by you, that turns the other models' outputs into one model's
first forcing slot. Do not extend the runner; write the function.

```python
def coupling(outputs: Mapping[str, xr.DataArray], leads: np.ndarray) -> xr.DataArray: ...
```

- `outputs[name]`: that model's outputs from the previous window's last frame to
  now, along `lead_time`.
- `leads`: the lead times the forcing must be valid at (the slot's `lead_time`
  coordinate shifted to the model's current lead). Return one entry per lead.
- The result must carry the forcing slot's spatial coordinates. A wrong lead
  count or grid raises at call time; a field on a different grid is cropped or nearest-regridded automatically (as in `fetch_data`; needs the `utils` extra), otherwise regrid *inside* the function; convert units there too.

## Workflow

1. **Read the models, not the docs.** For each model print `input_coords()`,
   `output_coords(input_coords())`, `forcing_coords()` and `recommended_sources`.
   Record steps (`output lead - input lead`), the variables the producer makes
   versus the consumer's forcing variables, grids, units, and the forcing slot's
   `lead_time` offsets (e.g. `[-10min, 0]` needs two frames).
2. **Pick the semantics from the cadences.**
   - Same cadence, same time: `hold(producer)`; the producer must be listed first.
   - Slow consumer, fast producer: `window_mean(producer, consumer_step)` or any
     xarray reduction over `slice(lead + eps, lead + step)`; producer listed first.
   - Fast consumer, slow producer: `hold(producer)`; the producer's value is
     constant across the consumer's sub-steps.
   - Model order in `models` decides who sees whose *new* output; the other sees
     the end of the previous window. Say which you chose and why.
3. **Write the function** in `earth2studio/coupler/couplings.py` as a factory
   returning a `Coupling`, with a full docstring (see
   [template](references/coupling-template.py)). Compose `hold`/`window_mean`
   with a function for regridding (`earth2studio.grids`), unit conversion or
   masking rather than copying them. Select variables by name (`.sel(variable=...)`)
   and reorder to the slot's variable order.
4. **Wire and run** with `CoupledRunner(models, sources, couplings={name: fn})` and
   `Pipeline(runner, output={model_name: backend, ...})`. Streams are named after
   the models. Uncoupled forcing slots and initial conditions come from sources.
5. **Test** in `test/coupler/` with mock models (see `test/coupler/test_coupler.py`
   `_Mock`): each step adds a known increment, so assert the exact forcing values
   and lead times each model received at every step, and that a wrong order or
   grid fails loudly. Real-weight checks are marked `package`.
6. **Verify**: `uv run pytest test/coupler -q && make format && make lint`.
   Report what you did not run.

## Common Mistakes

- Averaging over the wrong window: it is `(lead, lead + step]`, not `[lead, ...)`.
- Listing the models in the wrong order, so a mean is requested before the
  producer has run (`window_mean` raises; fix the dict order).
- Returning a regridded field without the slot's coordinate values.
- Mutating `outputs`; return new arrays.
- Putting pair-specific logic in `runner.py`. It belongs in `couplings.py`.

## Ensembles

`CoupledRunner` runs one member per work item (`member_batch=1`). To perturb
initial conditions use `EnsembleCoupledRunner(..., perturbations={name: lambda seed: Gaussian(0.05, seed=seed)})`.
New runner behavior belongs in a subclass that overrides a hook such as `_start`,
not in new `CoupledRunner` parameters.
