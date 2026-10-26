# Development Examples

Runnable tutorials for proposed APIs and workflows.

- Use the example-gallery script format with concise narrative sections.
- Keep examples focused, reproducible, and inexpensive to run.
- Do not commit generated notebooks, outputs, or artifacts.
- Update examples when their demonstrated API changes.

## Coordinate and model tutorials

- `01_grid_tutorial.py`: grid definitions, registration, inference, and selection.
- `02_model_contract_tutorial.py`: legacy tensor execution, hooks, and conformance.
- `03_coordinate_signatures.py`: new allocation-free DataArray signatures,
  grid-backed regional geometry, output planning, and precipitation statistics.
- `04_xarray_model_execution.py`: FCN DataArray stepping, leading-dimension batching,
  iterator hooks, and the Torch bridge with a small CPU-only synthetic core.
- `05_array_backends.py`: environment defaults, scoped backend overrides, and
  gradient-preserving Torch handoffs with optional CUDA/CuPy examples.
- `07_io_backends.py`: IO templates planned from model signatures, label-based
  DataArray writes, and metadata round trips with the in-memory XarrayBackend.
- `08_regridding.py`: `fetch_data` onto a target grid, explicit regridders,
  `RegriddedSource`, coverage, and `CompositeSource` on a shared grid.

## Execution contracts

- `06_runner.py`: the built-in `PrognosticRunner` with a diagnostic stream,
  `DiagnosticRunner` on source data, and a hand-written runner that stops early.
- `09_pipeline.py`: `Pipeline` writing one store per stream, per-stream variable
  filtering and regridding, and resume from a `ProgressStore` after a failure.

## Coupling

- `10_coupling.py`: a `Coupler` over toy atmosphere/ocean models with a windowed
  `z1000:mean:48h` exchange, lagged versus same-step coupling read off the
  forcing lead, a windowed diagnostic, and hybrid-to-pressure interpolation in a
  connector transform.
- `11_coupled_pipeline.py`: the coupler as a `Runner` under `Pipeline`, stream
  schemas and data requests, and in-memory initial conditions.

Run the signature tutorial from the repository root with
`.venv/bin/python dev/examples/03_coordinate_signatures.py`. It uses CPU-only
synthetic coordinates and requires no model weights or downloads.
